#!/usr/bin/env python

import rospy
import pose_retargeting.vrep as vrep
from pose_retargeting.simulator.simulator import SimulatorType


class JointHandlesDict:
    def __init__(self, simulator):
        body_handle_names = ['IMCP_side_joint', 'IMCP_front_joint', 'IPIP_joint', 'IDIP_joint', 'ITIP_tip',
                              'MMCP_side_joint', 'MMCP_front_joint', 'MPIP_joint', 'MDIP_joint', 'MTIP_tip',
                              'RMCP_side_joint', 'RMCP_front_joint', 'RPIP_joint', 'RDIP_joint', 'RTIP_tip',
                              'metacarpal_joint', 'PMCP_side_joint', 'PMCP_front_joint', 'PPIP_joint',
                              'PDIP_joint', 'PTIP_tip', 'TMCP_rotation_joint', 'TMCP_front_joint', 'TPIP_side_joint',
                              'TPIP_front_joint', 'TDIP_joint', 'TTIP_tip', 'ShadowRobot_base_tip',
                              'ShadowRobot_base_target']
        if simulator.type == SimulatorType.VREP:
            self.body_handles_dict = {}
            for handle_name in body_handle_names:
                result, handle = simulator.getObjectHandle(handle_name)
                if result != vrep.simx_return_ok:
                    rospy.logerr("Handle %s does not exist! Exiting.", handle_name)
                    exit(1)
                self.body_handles_dict[handle_name] = handle

        elif simulator.type == SimulatorType.MUJOCO:
            if simulator.env_name == 'relocateSR-v0':
                joint_names = ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1', 'rh_fftip', 'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2',
                               'rh_MFJ1', 'rh_mftip', 'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1', 'rh_rftip', 'rh_LFJ5',
                               'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1', 'rh_lftip', 'rh_THJ5', 'rh_THJ4', 'rh_THJ3',
                               'rh_THJ2', 'rh_THJ1', 'rh_thtip', 'rh_forearm']
                body_names = ['rh_ffknuckle', 'rh_ffproximal', 'rh_ffmiddle', 'rh_ffdistal', 'rh_fftip', 'rh_mfknuckle',
                              'rh_mfproximal', 'rh_mfmiddle', 'rh_mfdistal', 'rh_mftip', 'rh_rfknuckle', 'rh_rfproximal',
                              'rh_rfmiddle', 'rh_rfdistal', 'rh_rftip', 'rh_lfmetacarpal', 'rh_lfknuckle', 'rh_lfproximal',
                              'rh_lfmiddle', 'rh_lfdistal', 'rh_lftip', 'rh_thbase', 'rh_thproximal', 'rh_thhub',
                              'rh_thmiddle', 'rh_thdistal', 'rh_thtip', 'rh_forearm']
            elif simulator.env_name == 'relocate-v0':
                joint_names = ['FFJ3', 'FFJ2', 'FFJ1', 'FFJ0', 'fftip', 'MFJ3', 'MFJ2', 'MFJ1',
                               'MFJ0', 'mftip', 'RFJ3', 'RFJ2', 'RFJ1', 'RFJ0', 'rftip', 'LFJ4',
                               'LFJ3', 'LFJ2', 'LFJ1', 'LFJ0', 'lftip', 'THJ4', 'THJ3', 'THJ2',
                               'THJ1', 'THJ0', 'thtip', 'forearm']

                body_names = ['ffknuckle', 'ffproximal', 'ffmiddle', 'ffdistal', 'fftip', 'mfknuckle',
                              'mfproximal', 'mfmiddle', 'mfdistal', 'mftip', 'rfknuckle', 'rfproximal',
                              'rfmiddle', 'rfdistal', 'rftip', 'lfmetacarpal', 'lfknuckle', 'lfproximal',
                              'lfmiddle', 'lfdistal', 'lftip', 'thbase', 'thproximal', 'thhub',
                              'thmiddle', 'thdistal', 'thtip', 'forearm']
            else:
                raise ValueError

            self.body_handles_dict = dict(zip(body_handle_names, body_names))
            self.body_joint_pairs_dict = dict(zip(body_names, joint_names))
        else:
            raise ValueError

    def getHandle(self, handle_name):
        return self.body_handles_dict[handle_name]

    def getBodyJointName(self, body_name):
        return self.body_joint_pairs_dict[body_name]
