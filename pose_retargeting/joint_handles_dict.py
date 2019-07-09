#!/usr/bin/env python

import pose_retargeting.vrep as vrep
import rospy


class JointHandlesDict:
    def __init__(self, simulator):
        joint_handle_names = ['IMCP_side_joint', 'IMCP_front_joint', 'IPIP_joint', 'IDIP_joint', 'ITIP_tip',
                              'MMCP_side_joint', 'MMCP_front_joint', 'MPIP_joint', 'MDIP_joint', 'MTIP_tip',
                              'RMCP_side_joint', 'RMCP_front_joint', 'RPIP_joint', 'RDIP_joint', 'RTIP_tip',
                              'metacarpal_joint', 'PMCP_side_joint', 'PMCP_front_joint', 'PPIP_joint',
                              'PDIP_joint', 'PTIP_tip', 'TMCP_rotation_joint', 'TMCP_front_joint', 'TPIP_side_joint',
                              'TPIP_front_joint', 'TDIP_joint', 'TTIP_tip', 'ShadowRobot_base_target',
                              'ShadowRobot_base_tip']
        if simulator.name == 'vrep':
            self.joint_handles_dict = {}
            for handle_name in joint_handle_names:
                result, handle = simulator.getObjectHandle(handle_name)
                if result != vrep.simx_return_ok:
                    rospy.logerr("Handle %s does not exist! Exiting.", handle_name)
                    exit(1)
                self.joint_handles_dict[handle_name] = handle

        elif simulator.name == 'mujoco':
            self.joint_body_pairs_dict = {'IMCP_side_joint': 'rh_ffknuckle', 'IMCP_front_joint': 'rh_ffproximal',
                                          'IPIP_joint': 'rh_ffmiddle', 'IDIP_joint': 'rh_ffdistal',
                                          'ITIP_tip': 'rh_fftip', 'MMCP_side_joint': 'rh_mfknuckle',
                                          'MMCP_front_joint': 'rh_mfproximal', 'MPIP_joint': 'rh_mfmiddle',
                                          'MDIP_joint': 'rh_mfdistal', 'MTIP_tip': 'rh_mftip',
                                          'RMCP_side_joint': 'rh_rfknuckle', 'RMCP_front_joint': 'rh_rfproximal',
                                          'RPIP_joint': 'rh_rfmiddle', 'RDIP_joint': 'rh_rfdistal',
                                          'RTIP_tip': 'rh_rftip',
                                          'metacarpal_joint': 'rh_lfmetacarpal', 'PMCP_side_joint': 'rh_lfknuckle',
                                          'PMCP_front_joint': 'rh_lfproximal', 'PPIP_joint': 'rh_lfmiddle',
                                          'PDIP_joint': 'rh_lfdistal', 'PTIP_tip': 'rh_lftip',
                                          'TMCP_rotation_joint': 'rh_thbase', 'TMCP_front_joint': 'rh_thproximal',
                                          'TPIP_side_joint': 'rh_thhub', 'TPIP_front_joint': 'rh_thmiddle',
                                          'TDIP_joint': 'rh_thdistal', 'TTIP_tip': 'rh_thtip',
                                          'ShadowRobot_base_tip': 'rh_forearm'}

            self.joint_handles_dict = {'IMCP_side_joint': 'rh_FFJ4', 'IMCP_front_joint': 'rh_FFJ3',
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
        else:
            raise ValueError

    def getHandle(self, handle_name):
        return self.joint_handles_dict[handle_name]
    
    def getJointBodyHandle(self, handle_name):
        return self.joint_body_pairs_dict[handle_name]
