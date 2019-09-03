#! /usr/bin/env python

from __future__ import print_function, division
import numpy as np
import os
import copy
from APF_ros import *
import rospy
from std_msgs.msg import String
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler


import tf2_ros

MAP_FRAME='map'

class ObstacleAvoiderROS(object):

    def __init__(self):
        self.prior = PotentialFieldsController()

        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.sub_laser = rospy.Subscriber('/scan_laser_fixed', LaserScan, self.cb_laser, queue_size=1)

        #self.sub_odom = rospy.Subscriber('/odom', Odometry, queue_size=1)
        #self.last_odom = None
        # The listener recieves tf2 tranformations over the wire and buffers them up for 10 seconds
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self.goal_loc = np.array([-5.667,0.593])


    def cb_laser(self, data):
        # Get robot's position through TF @ same timestamp as laser message
        try:
            t = self._tf_buffer.lookup_transform(MAP_FRAME, 'base_link', data.header.stamp, rospy.Duration(1.0))
        except Exception as e:
            rospy.logerr(e)
            return 

        # Convert the quaternion to euler angles
        q = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
        euler = euler_from_quaternion(q)
        robot_yaw = euler[2]
        # Figure out the relative angle of the goal from the robot's POSE 
        
        robot_loc = np.array([t.transform.translation.x, t.transform.translation.y])
        robot_angle = np.arctan2(np.sin(robot_yaw), np.cos(robot_yaw))
        #print(np.rad2deg(robot_angle))
        to_goal = self.goal_loc - robot_loc
        #print(np.rad2deg(np.arctan2(to_goal[1], to_goal[0])))
        angle_to_goal = robot_angle - np.arctan2(to_goal[1], to_goal[0])
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))
        #print(np.rad2deg(angle_to_goal))
        dist_to_goal  = np.linalg.norm(to_goal)
        print('Distance to Goal: ', dist_to_goal)
        # Remove the last element of laser scan array to clip it to 180 samples
        laser_scan = np.array(data.ranges[:180])
        laser_scan[laser_scan < 0.23] = 15
        action = self.prior.computeResultant(angle_to_goal, laser_scan)
        linear_vel = action[0] * 0.25
        angular_vel = action[1] * 0.25
        twist_msg = Twist(linear=Vector3(linear_vel, 0, 0), angular=Vector3(0,0,angular_vel))
        self.pub_vel.publish(twist_msg)
        print('Linear Vel: ', linear_vel)
        print('Angular Vel: ', angular_vel)
        # rospy.sleep(0.01666667)
        # twist_msg = Twist(linear=Vector3(0, 0, 0), angular=Vector3(0,0,0))
        # self.pub_vel.publish(twist_msg)
        #print(twist_msg)



        

        # Figure out the desired action
        

        # From the action, form & publish a velocity to /cmd_vel
        # TODO
        # twist_msg = Twist(linear=Vector3(0, 0, 0), angular=Vector3(0,0,0))
        # self._pub_vel.publish(twist_msg)

        # state_info = data.data
        # from this data compute the angle to goal and process the laser scan data for inputs to the prior



if __name__ == "__main__":
    rospy.init_node('env_ros')

    oar = ObstacleAvoiderROS()
    # rospy.spin()

    while not rospy.is_shutdown():
        plt.show(block=False)
        plt.pause(0.001)


