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

MAP_FRAME = 'map'
PATH = os.path.dirname(os.path.realpath(__file__))
METHOD = "residual_switch" # Options: 1.) residual_switch 2.) residual_no_switch 3.) policy 4.) prior 
GOAL_COMPLETE_THRESHOLD = 0.2
SUB_GOAL_FREQUENCY = 5


def _dist(p1, p2):
    return ((p2[0] - p1[0])**2 + (p2[1] - p2[1])**2)**0.5


class ActorNetwork_residual(nn.Module):

    def __init__(self, obs_size, act_size):
        super(ActorNetwork_residual, self).__init__()
        self.a1 = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(300, act_size),
            nn.Tanh())

    def forward(self, obs):
        return self.a1(obs)


class ActorNetwork_policy(nn.Module):

    def __init__(self, obs_size, act_size):
        super(ActorNetwork_policy, self).__init__()
        self.a1 = nn.Sequential(
            nn.Linear(obs_size, 400), nn.ReLU(), nn.Linear(400, 300),
            nn.ReLU(), nn.Linear(300, act_size), nn.Tanh())

    def forward(self, obs):
        return self.a1(obs)


class ObstacleAvoiderROS(object):

    def __init__(self):
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
        if self.method == "residual_switch" or "residual_no_switch":
            self.actor = ActorNetwork_residual(21, 2)
        else:
            self.actor = ActorNetwork_policy(19, 2)
        self.actions_prev = [0, 0]

    def get_next_goal(self):
        self.goal_loc = (self.goal_list.pop(0) if self.goal_list else None)

    def load_weights(self):
        if self.method == "residual_switch" or "residual_no_switch":

            model_name = "1567642880.05_PointGoalNavigation_residual_EnvType_4_sparse_Dropout_vhf_ROBOT_FINAL"
            self.actor.load_state_dict(
                torch.load(PATH + '/residual_policy_weights/' + model_name +
                           'pi.pth'))
        else:
            #model_name = "1567656455.73_PointGoalNavigation_policy_EnvType_4_dense_Dropout_vhf_ROBOT_FINAL"
            model_name = "1567741358.18_PointGoalNavigation_policy_EnvType_5_sparse_Dropout_vhf_ROBOT_FINAL"
            self.actor.load_state_dict(
                torch.load(PATH + '/policy_only_weights/' + model_name +
                           'pi.pth'))
        return

    def extract_network_uncertainty(self, state):
        action = self.actor(state.repeat(100, 1))
        mean = torch.mean(action, dim=0).detach().numpy()
        var = torch.var(action, dim=0).detach().numpy()
        return mean, var

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

    def process_observation(self, angle_to_goal, dist_to_goal, laser_scan,
                            prior_action):
        num_bins = 15
        laser_scan_binned = np.zeros(num_bins)
        div_factor = int(180 / num_bins)
        laser_scan[laser_scan < 0.25] = np.nan
        for i in range(num_bins):
            laser_scan_binned[i] = np.nanmean(
                laser_scan[i * div_factor:(i * div_factor + div_factor)])
        if self.method == "residual":
            obs = np.concatenate([
                prior_action, laser_scan_binned, self.actions_prev,
                [dist_to_goal], [angle_to_goal]
            ])
        else:
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

        # Compute what our actions should be
        eps = np.random.random()
        laser_scan, angle_to_goal, dist_to_goal = self.process_data(t, data)
        prior_action = self.prior.computeResultant(angle_to_goal, laser_scan)
        obs = self.process_observation(angle_to_goal, dist_to_goal, laser_scan,
                                       prior_action)
        policy_action, var = self.extract_network_uncertainty(
            torch.as_tensor(obs).float())
        hybrid_action = (policy_action + prior_action).clip(-1, 1)

        # Decide whether we should bail on performing any actions
        if dist_to_goal < GOAL_COMPLETE_THRESHOLD:
            self.get_next_goal()
        if self.goal_loc is None:
            print("No goal, exiting early")
            return

        # Send actions to the robot
        if self.method == "residual_switch":
            if eps > var[0] or eps > var[1]:
                linear_vel = hybrid_action[0] * 0.25
                angular_vel = hybrid_action[1] * 0.25
                twist_msg = Twist(
                    linear=Vector3(linear_vel, 0, 0),
                    angular=Vector3(0, 0, angular_vel))
                self.pub_vel.publish(twist_msg)
                self.pub_mode.publish(Bool(True))
                print('Residual')
            else:
                linear_vel = prior_action[0] * 0.25
                angular_vel = prior_action[1] * 0.25
                twist_msg = Twist(
                    linear=Vector3(linear_vel, 0, 0),
                    angular=Vector3(0, 0, angular_vel))
                self.pub_vel.publish(twist_msg)
                self.pub_mode.publish(Bool(False))
                print('Prior')
        elif self.method == "residual_no_switch":
                linear_vel = hybrid_action[0] * 0.25
                angular_vel = hybrid_action[1] * 0.25
                twist_msg = Twist(
                    linear=Vector3(linear_vel, 0, 0),
                    angular=Vector3(0, 0, angular_vel))
                self.pub_vel.publish(twist_msg)
        elif self.method == "prior":
            linear_vel = prior_action[0] * 0.25
            angular_vel = prior_action[1] * 0.25
            twist_msg = Twist(
                linear=Vector3(linear_vel, 0, 0),
                angular=Vector3(0, 0, angular_vel))
            self.pub_vel.publish(twist_msg)
        elif self.method == "policy":
            policy_action = policy_action.clip(-1, 1)
            linear_vel = policy_action[0] * 0.25
            angular_vel = policy_action[1] * 0.25
            twist_msg = Twist(
                linear=Vector3(linear_vel, 0, 0),
                angular=Vector3(0, 0, angular_vel))
            self.pub_vel.publish(twist_msg)

        self.actions_prev = np.array([linear_vel, angular_vel])

        print('Linear Vel: ', linear_vel)
        print('Angular Vel: ', angular_vel)


if __name__ == "__main__":
    rospy.init_node('env_ros')
    oar = ObstacleAvoiderROS()
    oar.load_weights()
    rospy.spin()

    # while not rospy.is_shutdown():
    #     plt.show(block=False)
    #     plt.pause(0.001)
