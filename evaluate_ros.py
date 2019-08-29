#! /usr/bin/env python3

import torch, torch.nn as nn
from env import *
import numpy as np
import os
from tensorboardX import SummaryWriter
import copy
from APF_ros import *

prior = PotentialFieldsController()

def _get_data():
    #get laser scan data from robot and process
    #get robot angle to goal in robot frame and process
    laser_scan = 0
    angle_to_goal = 0
    return angle_to_goal, laser_scan

def _step_robot(action):
    #actions are between -1 and 1
    linear_vel = action[0] 
    angular_vel = action[1]

    #execute these action on the robot using ros for a fixed timestep
    #scale them appropriately for the robot
    return



while(True):

    angle_to_goal, laser_scan = _get_data()
    action = prior.computeResultant(angle_to_goal, laser_scan)
    _step_robot(action)