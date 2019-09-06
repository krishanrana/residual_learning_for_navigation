#! /usr/bin/env python

from __future__ import print_function, division
import numpy as np
import os
import copy
from APF_ros import *
import rospy
from std_msgs.msg import String
from std_msgs.msg import String, Int32, Int32MultiArray, MultiArrayLayout, MultiArrayDimension
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import torch, torch.nn as nn
import tf2_ros

MAP_FRAME = 'map'
PATH = os.path.dirname(os.path.realpath(__file__))
model_name = "1567642880.05_PointGoalNavigation_residual_EnvType_4_sparse_Dropout_vhf_ROBOT_FINAL"


class ActorNetwork(nn.Module):

    def __init__(self, obs_size, act_size):
        super(ActorNetwork, self).__init__()
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
        self.goal_loc = np.array([-5.230626, -0.7525351])
        self.actor = ActorNetwork(21, 2)
        self.actions_prev = [0, 0]
        self.method = "prior"

    def load_weights(self):
        self.actor.load_state_dict(
            torch.load(PATH + '/residual_policy_weights/' + model_name +
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

        obs = np.concatenate([
            prior_action, laser_scan_binned, self.actions_prev, [dist_to_goal],
            [angle_to_goal]
        ])

        return obs

    def cb_laser(self, data):
        # Get robot's position through TF @ same timestamp as laser message
        try:
            t = self._tf_buffer.lookup_transform(MAP_FRAME, 'base_link',
                                                 data.header.stamp,
                                                 rospy.Duration(1.0))
        except Exception as e:
            rospy.logerr(e)
            return

        eps = np.random.random()
        laser_scan, angle_to_goal, dist_to_goal = self.process_data(t, data)
        prior_action = self.prior.computeResultant(angle_to_goal, laser_scan)
        obs = self.process_observation(angle_to_goal, dist_to_goal, laser_scan,
                                       prior_action)
        policy_action, var = self.extract_network_uncertainty(
            torch.as_tensor(obs).float())
        hybrid_action = (policy_action + prior_action).clip(-1, 1)

        if self.method == "residual":
            if eps > var[0] or eps > var[1]:
                linear_vel = hybrid_action[0] * 0.25
                angular_vel = hybrid_action[1] * 0.25
                twist_msg = Twist(
                    linear=Vector3(linear_vel, 0, 0),
                    angular=Vector3(0, 0, angular_vel))
                self.pub_vel.publish(twist_msg)
                print('Residual')
            else:
                linear_vel = prior_action[0] * 0.25
                angular_vel = prior_action[1] * 0.25
                twist_msg = Twist(
                    linear=Vector3(linear_vel, 0, 0),
                    angular=Vector3(0, 0, angular_vel))
                self.pub_vel.publish(twist_msg)
                print('Prior')
        elif self.method == "prior":
            linear_vel = prior_action[0] * 0.25
            angular_vel = prior_action[1] * 0.25
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
