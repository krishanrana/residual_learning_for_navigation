#! /usr/bin/env python

from __future__ import print_function, division
import numpy as np
import os
import copy
from APF_ros import *
import rospy
from std_msgs.msg import Bool, String, Int32, Int32MultiArray, MultiArrayLayout, MultiArrayDimension
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import torch, torch.nn as nn
import tf2_ros
import ray
import scipy.stats as stats
import collections
from torch.distributions import Normal
import math
import matplotlib.pyplot as plt


MAP_FRAME = 'map'
PATH = os.path.dirname(os.path.realpath(__file__))
METHOD = "prior"  # Options: 1.) residual_switch 2.) residual_no_switch 3.) policy 4.) prior
GOAL_COMPLETE_THRESHOLD = 0.2
SUB_GOAL_FREQUENCY = 2.5
num_agents = 5
VIS_GRAPH = False

def _dist(p1, p2):
    return ((p2[0] - p1[0])**2 + (p2[1] - p2[1])**2)**0.5

# SAC Policy Model

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi, mu, std


def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma):
    # The policy mu and sigma are from the stochastic SAC output
    # The sigma from prior is fixed
    w1 = 1
    w2 = 1
    mu = (np.power(policy_sigma, 2) * w1 * prior_mu + np.power(prior_sigma,2) * w2 * policy_mu)/(np.power(policy_sigma,2) * w1 + np.power(prior_sigma,2) * w2)
    sigma = np.sqrt((np.power(prior_sigma,2) * np.power(policy_sigma,2))/(np.power(policy_sigma,2) * w1 + np.power(prior_sigma,2) * w2))
    return mu, sigma


def fuse_ensembles_deterministic(ensemble_actions):
    # Takes in the ensemble actions and computes the mean and variance of the data
    global num_agents
    actions = torch.tensor([ensemble_actions[i][0] for i in range (num_agents)])
    mu = torch.mean(actions, dim=0)
    var = torch.var(actions, dim=0)
    sigma = np.sqrt(var)
    return mu, sigma

@ray.remote
def get_action(state, policy):
    # This function can be run across multiple processes
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    a, _, mu, std = policy(state, False, False)
    return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]

def get_action_simple(state, policy):
    # Use this function when evaluating the policy alone
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    a, _, mu, std = policy(state, False, False)
    return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]




class ObstacleAvoiderROS(object):

    def __init__(self):
        global num_agents
        self.prior = PotentialFieldsController()

        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.sub_laser = rospy.Subscriber(
            '/scan_laser_fixed', LaserScan, self.cb_laser, queue_size=1)

        #self.sub_odom = rospy.Subscriber('/odom', Odometry, queue_size=1)
        #self.last_odom = None
        # The listener recieves tf2 tranformations over the wire and buffers them up for 10 seconds
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # self.goal_loc = np.array([-5.230626, -0.7525351])
        self.goal_loc = None
        self.goal_list = None
        self.sub_global_plan = rospy.Subscriber(
            '/move_base/GlobalPlanner/plan',
            Path,
            self.cb_global_plan,
            queue_size=1)

        self.pub_mode = rospy.Publisher('/using_residual', Bool, queue_size=1)
        self.method = METHOD

        ensemble_file_name = "trained_ensemble_for_robot_new_5/1582454594.5134897_PointGoalNavigation_sparse_SAC_spinup_long_horizon_hybridFOR_ROBOT10_"

        # Load in the network weights
        self.policy_net_ensemble = [torch.load(ensemble_file_name + str(i) + "_.pth").cpu() for i in range(num_agents)]

        self.policy_net = self.policy_net_ensemble[0]

        if VIS_GRAPH:
            self.fig = plt.gcf()
            self.fig.show()
            self.fig.canvas.draw()
            plt.axis([-10,10,0,2])


    def get_next_goal(self):
        self.goal_loc = (self.goal_list.pop(0) if self.goal_list else None)


    def process_data(self, t, data):
        # Convert the quaternion to euler angles
        q = (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w)
        euler = euler_from_quaternion(q)
        robot_yaw = euler[2]
        # Figure out the relative angle of the goal from the robot's POSE

        robot_loc = np.array(
            [t.transform.translation.x, t.transform.translation.y])
        robot_angle = np.arctan2(np.sin(robot_yaw), np.cos(robot_yaw))
        #print(np.rad2deg(robot_angle))
        to_goal = self.goal_loc - robot_loc
        #print(np.rad2deg(np.arctan2(to_goal[1], to_goal[0])))
        angle_to_goal = robot_angle - np.arctan2(to_goal[1], to_goal[0])
        angle_to_goal = np.arctan2(
            np.sin(angle_to_goal), np.cos(angle_to_goal))
        #print(np.rad2deg(angle_to_goal))
        dist_to_goal = np.linalg.norm(to_goal)
        print('Distance to Goal: ', dist_to_goal)
        # Remove the last element of laser scan array to clip it to 180 samples
        laser_scan = np.array(data.ranges[:180])
        # Deal with the spurious data
        laser_scan[laser_scan < 0.25] = 1.5
        # Clip laser scan data to 1.5 meters
        laser_scan[laser_scan > 1.5] = 1.5

        return laser_scan, angle_to_goal, dist_to_goal

    def process_observation(self, angle_to_goal, dist_to_goal, laser_scan):

        num_bins = 15
        laser_scan_binned = np.zeros(num_bins)
        div_factor = int(180 / num_bins)
        laser_scan[laser_scan < 0.25] = np.nan
        for i in range(num_bins):
            laser_scan_binned[i] = np.nanmean(
                laser_scan[i * div_factor:(i * div_factor + div_factor)])

        obs = np.concatenate([
            laser_scan_binned, self.actions_prev, [dist_to_goal],
            [angle_to_goal]
        ])

        return obs

    def cb_global_plan(self, data):
        # TODO some processing logic to get list of goals
        print("NEW GLOBAL PLAN RECEIVED")
        raw_goal_list = [
            [p.pose.position.x, p.pose.position.y] for p in data.poses
        ]
        print("PLAN STARTS AT: %s, & FINISHES AT: %s" % (raw_goal_list[0],
                                                         raw_goal_list[-1]))
        self.goal_list = []
        for g in raw_goal_list:
            if (not self.goal_list or
                    _dist(g, self.goal_list[-1]) > SUB_GOAL_FREQUENCY):
                self.goal_list.append(g)
        if self.goal_list[-1] != raw_goal_list[-1]:
            self.goal_list.append(raw_goal_list[-1])

        print("GOT THE FOLLOWING SUB_GOALS:")
        for g in self.goal_list:
            print("\t%s" % g)

        # Take the first goal from the goal list & make it our current goal
        self.get_next_goal()

    def cb_laser(self, data):
        # Bail if we don't have a current goal
        if self.goal_loc is None:
            print("No goal, exiting early")
            return

        # Get robot's position through TF @ same timestamp as laser message
        try:
            t = self._tf_buffer.lookup_transform(MAP_FRAME, 'base_link',
                                                 data.header.stamp,
                                                 rospy.Duration(1.0))
        except Exception as e:
            rospy.logerr(e)
            return

        # Decide whether we should bail on performing any actions
        if dist_to_goal < GOAL_COMPLETE_THRESHOLD:
            self.get_next_goal()
        if self.goal_loc is None:
            print("No goal, exiting early")
            return

        if self.method == "hybrid":
            laser_scan, angle_to_goal, dist_to_goal = self.process_data(t, data)
            mu_prior = self.prior.computeResultant(angle_to_goal, laser_scan)

            obs = self.process_observation(angle_to_goal, dist_to_goal, laser_scan)
            ensemble_actions = ray.get([get_action.remote(obs, p) for p in self.policy_net_ensemble])
            mu_ensemble, std_ensemble = fuse_ensembles_deterministic(ensemble_actions)
            mu_hybrid, std_hybrid = fuse_controllers(mu_prior, std_prior, mu_ensemble, std_ensemble)

            vmu_policy = mu_ensemble[0]
            vsigma_policy = std_ensemble[0]

            vmu_prior = mu_prior[0]
            vsigma_prior = std_prior

            wmu_policy = mu_ensemble[1]
            wsigma_policy = std_ensemble[1]

            wmu_prior = mu_prior[1]
            wsigma_prior = std_prior  

            vmu_combined = mu_hybrid[0]
            vsigma_combined = std_hybrid[0]
            wmu_combined = mu_hybrid[1]
            wsigma_combined = std_hybrid[1]

            if VIS_GRAPH:
                    x = np.linspace(-3, 3, 100)
                    plt.subplot(211)
                    plt.plot(x, stats.norm.pdf(x, vmu_policy, vsigma_policy))
                    plt.plot(x, stats.norm.pdf(x, vmu_prior, vsigma_prior))
                    plt.plot(x, stats.norm.pdf(x, vmu_combined, vsigma_combined))
                    plt.legend(['Policy', 'Prior', 'Combined'], loc="upper right")
                    plt.xlabel('Linear Velocity')
                    plt.subplot(212)
                    plt.plot(x, stats.norm.pdf(x, wmu_policy, wsigma_policy))
                    plt.plot(x, stats.norm.pdf(x, wmu_prior, wsigma_prior))
                    plt.plot(x, stats.norm.pdf(x, wmu_combined, wsigma_combined))
                    plt.xlabel('Angular Velocity')
                    self.fig.canvas.draw()
                    self.fig.clf()

            dist_combined  = Normal(torch.tensor(mu_hybrid), torch.tensor(std_hybrid))
            act = dist_combined.rsample()
            act = torch.tanh(act).numpy()

            linear_vel = act[0] * 0.25
            angular_vel = act[1] * 0.25
            twist_msg = Twist(
            linear=Vector3(linear_vel, 0, 0),
            angular=Vector3(0, 0, angular_vel))
            self.pub_vel.publish(twist_msg)
            self.pub_mode.publish(Bool(True))
            
        elif self.method == "policy":
            action_dist = get_action_simple(state, policy_net)
            dist_policy = Normal(torch.tensor(action_dist[0]), torch.tensor(action_dist[1]))
            act = dist_policy.sample()
            policy_action = torch.tanh(act).numpy()
            linear_vel = policy_action[0] * 0.25
            angular_vel = policy_action[1] * 0.25
            twist_msg = Twist(
                linear=Vector3(linear_vel, 0, 0),
                angular=Vector3(0, 0, angular_vel))
            self.pub_vel.publish(twist_msg)

        elif self.method == "prior":
            laser_scan, angle_to_goal, dist_to_goal = self.process_data(t, data)
            prior_action = self.prior.computeResultant(angle_to_goal, laser_scan)
            linear_vel = prior_action[0] * 0.25
            angular_vel = prior_action[1] * 0.25
            twist_msg = Twist(
                linear=Vector3(linear_vel, 0, 0),
                angular=Vector3(0, 0, angular_vel))
            self.pub_vel.publish(twist_msg)

        print('Linear Vel: ', linear_vel)
        print('Angular Vel: ', angular_vel)


if __name__ == "__main__":
    rospy.init_node('env_ros')
    oar = ObstacleAvoiderROS()
    rospy.spin()

    # while not rospy.is_shutdown():
    #     plt.show(block=False)
    #     plt.pause(0.001)