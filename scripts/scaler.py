#!/usr/bin/env python

import numpy as np
import vrep


class Scaler:
    def __init__(self):
        self.fingers = [[2, 9, 10, 11], [3, 12, 13, 14], [4, 15, 16, 17], [5, 18, 19, 20], [1, 6, 7, 8]]
        self.vrep_points_knuckles = [np.array([0.033, -.0099, 0.352]), np.array([0.011, -0.0099, .356]), np.array([-.011, -.0099, .352]),
                            np.array([-0.033, -.0099, .3436]), np.array([0.033, -0.0189, 0.286])]
        self.knuckle_first_poses = {}
        finger_handles = ['IMCP_front_joint', 'IPIP_joint', 'IDIP_joint', 'ITIP_tip', 'MMCP_front_joint', 'MPIP_joint', 'MDIP_joint', 'MTIP_tip',
                          'RMCP_front_joint', 'RPIP_joint', 'RDIP_joint', 'RTIP_tip', 'PMCP_front_joint', 'PPIP_joint', 'PDIP_joint', 'PTIP_tip',
                          'TMCP_front_joint', 'TPIP_front_joint', 'TDIP_joint', 'TTIP_tip']

        for index in [item for sublist in self.fingers for item in sublist]:
            _, self.knuckle_first_poses[index] = vrep.

    def scalePoints(self, original_points_to_scale):
        new_points = original_points_to_scale[:]  # copy
        for finger_index, finger in enumerate(self.fingers):
            for this_index, point_index in enumerate(finger):
                if finger_index == 0:
                    new_points[point_index] = self.vrep_points_knuckles[finger_index]
                else:
                    previous_index = finger[this_index - 1]
                    vector = original_points_to_scale[point_index] - original_points_to_scale[previous_index]
                    new_points[point_index] = new_points[previous_index] + vector / np.linalg.norm(vector) * length  # length do wyliczenia jeszcze!!!!!!!!!!
