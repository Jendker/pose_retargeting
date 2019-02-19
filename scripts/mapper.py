#!/usr/bin/env python

import vrep
import numpy as np


class Mapper:
    def __init__(self):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        if self.clientID != -1:
            print('Connected to remote API server')

    def mapFinger(self, finger_pose):
        self.__getJacobian()
        for pose in finger_pose:
            pass

    def __getJacobian(self):
        empty_buff = bytearray()
        _, dimension, jacobian_vect, _, _ = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                                                  vrep.sim_scripttype_childscript,
                                                                  'jacobianIKGroup', [], [],
                                                                  ['IK_Group'], empty_buff,
                                                                  vrep.simx_opmode_blocking)
        jacobian = np.mat(jacobian_vect).reshape(dimension)
        print(dimension)
        print(jacobian)

    def callback(self, data):
        finger_indices = [5, 6, 7, 8]
        finger_pose = []
        for index in finger_indices:
            finger_pose.append(data.joints_position[index])
        self.mapFinger(finger_pose)
