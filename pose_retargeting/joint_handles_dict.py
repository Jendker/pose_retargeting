#!/usr/bin/env python

import pose_retargeting.vrep as vrep
import rospy


class JointHandlesDict:
    def __init__(self, simulator):
        body_handle_names = ['IMCP_side_joint', 'IMCP_front_joint', 'IPIP_joint', 'IDIP_joint', 'ITIP_tip',
                              'MMCP_side_joint', 'MMCP_front_joint', 'MPIP_joint', 'MDIP_joint', 'MTIP_tip',
                              'RMCP_side_joint', 'RMCP_front_joint', 'RPIP_joint', 'RDIP_joint', 'RTIP_tip',
                              'metacarpal_joint', 'PMCP_side_joint', 'PMCP_front_joint', 'PPIP_joint',
                              'PDIP_joint', 'PTIP_tip', 'TMCP_rotation_joint', 'TMCP_front_joint', 'TPIP_side_joint',
                              'TPIP_front_joint', 'TDIP_joint', 'TTIP_tip', 'ShadowRobot_base_target',
                              'ShadowRobot_base_tip']
        if simulator.name == 'vrep':
            self.body_handles_dict = {}
            for handle_name in body_handle_names:
                result, handle = simulator.getObjectHandle(handle_name)
                if result != vrep.simx_return_ok:
                    rospy.logerr("Handle %s does not exist! Exiting.", handle_name)
                    exit(1)
                self.body_handles_dict[handle_name] = handle

        elif simulator.name == 'mujoco':
            joint_body_pairs_dict = {'rh_FFJ4': 'rh_ffknuckle', 'rh_FFJ3': 'rh_ffproximal',
                                          'rh_FFJ2': 'rh_ffmiddle', 'rh_FFJ1': 'rh_ffdistal',
                                          'rh_fftip': 'rh_fftip', 'rh_MFJ4': 'rh_mfknuckle',
                                          'rh_MFJ3': 'rh_mfproximal', 'rh_MFJ2': 'rh_mfmiddle',
                                          'rh_MFJ1': 'rh_mfdistal', 'rh_mftip': 'rh_mftip',
                                          'rh_RFJ4': 'rh_rfknuckle', 'rh_RFJ3': 'rh_rfproximal',
                                          'rh_RFJ2': 'rh_rfmiddle', 'rh_RFJ1': 'rh_rfdistal',
                                          'rh_rftip': 'rh_rftip',
                                          'rh_LFJ5': 'rh_lfmetacarpal', 'rh_LFJ4': 'rh_lfknuckle',
                                          'rh_LFJ3': 'rh_lfproximal', 'rh_LFJ2': 'rh_lfmiddle',
                                          'rh_LFJ1': 'rh_lfdistal', 'rh_lftip': 'rh_lftip',
                                          'rh_THJ5': 'rh_thbase', 'rh_THJ4': 'rh_thproximal',
                                          'rh_THJ3': 'rh_thhub', 'rh_THJ2': 'rh_thmiddle',
                                          'rh_THJ1': 'rh_thdistal', 'rh_thtip': 'rh_thtip',
                                          'rh_forearm': 'rh_forearm'}

            joint_handles_dict = {'IMCP_side_joint': 'rh_FFJ4', 'IMCP_front_joint': 'rh_FFJ3',
                                       'IPIP_joint': 'rh_FFJ2', 'IDIP_joint': 'rh_FFJ1',
                                       'ITIP_tip': 'rh_fftip', 'MMCP_side_joint': 'rh_MFJ4',
                                       'MMCP_front_joint': 'rh_MFJ3', 'MPIP_joint': 'rh_MFJ2',
                                       'MDIP_joint': 'rh_MFJ1', 'MTIP_tip': 'rh_mftip',
                                       'RMCP_side_joint': 'rh_RFJ4', 'RMCP_front_joint': 'rh_RFJ3',
                                       'RPIP_joint': 'rh_RFJ2', 'RDIP_joint': 'rh_RFJ1', 'RTIP_tip': 'rh_rftip',
                                       'metacarpal_joint': 'rh_LFJ5', 'PMCP_side_joint': 'rh_LFJ4',
                                       'PMCP_front_joint': 'rh_LFJ3', 'PPIP_joint': 'rh_LFJ2',
                                       'PDIP_joint': 'rh_LFJ1', 'PTIP_tip': 'rh_lftip',
                                       'TMCP_rotation_joint': 'rh_THJ5', 'TMCP_front_joint': 'rh_THJ4',
                                       'TPIP_side_joint': 'rh_THJ3', 'TPIP_front_joint': 'rh_THJ2',
                                       'TDIP_joint': 'rh_THJ1', 'TTIP_tip': 'rh_thtip',
                                       'ShadowRobot_base_tip': 'rh_forearm'}

            self.body_handles_dict = {k: joint_body_pairs_dict[v] for k, v in joint_handles_dict.items()}
            self.body_joint_pairs_dict = {v: k for k, v in joint_body_pairs_dict.items()}
        else:
            raise ValueError

    def getHandle(self, handle_name):
        return self.body_handles_dict[handle_name]

    def getBodyJointName(self, body_name):
        return self.body_joint_pairs_dict[body_name]
