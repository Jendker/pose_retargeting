#!/usr/bin/env python

import pose_retargeting.vrep as vrep
import numpy as np
import rospy
from pose_retargeting.error_calculation import ErrorCalculation
from pose_retargeting.jacobians.jacobian_calculation_vrep import ConfigurationType
from pose_retargeting.simulator.simulator import SimulatorType
import math
import time


def degToRad(angle):
    return angle / 180.0 * math.pi


class HandPart:
    def __init__(self, list_joint_handles_names, tip_handle_name, task_descriptor_base_handles_and_indices,
                 joints_limits, configuration_type, name, simulator):
        self.initialized = False
        self.task_prioritization = False
        self.simulator = simulator
        self.hand_base_handle = self.simulator.getHandle('ShadowRobot_base_tip')
        self.name = name
        self.tip_handle = self.simulator.getHandle(tip_handle_name)
        self.list_joints_handles = []
        for joint_handle_name in list_joint_handles_names:
            handle = self.simulator.getHandle(joint_handle_name)
            self.list_joints_handles.append(handle)

        handle_name_dict = dict(zip(list_joint_handles_names, self.list_joints_handles))
        handle_name_dict[tip_handle_name] = self.tip_handle
        self.task_descriptor_handles = [handle_name_dict[joint_handle_name] for joint_handle_name in
                                        task_descriptor_base_handles_and_indices[0]]
        self.base_handles = [handle_name_dict[joint_handle_name] for joint_handle_name in
                             task_descriptor_base_handles_and_indices[1]]
        self.task_descriptor_equivalent_hpe_indices = task_descriptor_base_handles_and_indices[2]
        self.DOF_count = len(self.list_joints_handles)
        self.tasks_count = len(self.task_descriptor_handles)

        if self.task_prioritization:
            self.K_matrix = np.identity(3) * 2  # for prioritization we use just single error
        else:
            self.K_matrix = np.identity(3 * len(self.task_descriptor_handles)) * 2
        self.weight_matrix_inv = np.identity(self.DOF_count)
        if self.task_prioritization:
            self.damping_matrix = np.identity(3) * 0.00001  # for prioritization we use just single error
        else:
            self.damping_matrix = np.identity(3 * len(self.task_descriptor_handles)) * 0.00001

        self.human_hand_vel = np.zeros(3 * self.tasks_count)
        self.last_human_hand_part_pose = self.simulator.simulationObjectsPose(
            self.task_descriptor_handles, mode=vrep.simx_opmode_blocking)
        all_handles_for_jacobian_calc = self.list_joints_handles[:]
        all_handles_for_jacobian_calc.append(self.tip_handle)
        self.jacobian_calculation = self.simulator.jacobianCalculation(
            all_handles_for_jacobian_calc, zip(self.task_descriptor_handles, self.base_handles),
            self.simulator, configuration_type=configuration_type)

        for joint_handle in self.list_joints_handles:  # initialize streaming
            simulator.getJointPosition(joint_handle, mode=vrep.simx_opmode_streaming)
        for handle in self.task_descriptor_handles:
            simulator.getObjectPosition(handle, self.hand_base_handle, mode=vrep.simx_opmode_streaming)

        self.joint_velocity = np.zeros(self.DOF_count)
        self.joints_limits = []
        for joint_limits in joints_limits:
            max_angle, min_angle = joint_limits
            self.joints_limits.append([degToRad(max_angle), degToRad(min_angle)])
        self.dummy_targets_handles = self.__createTargetDummies()
        self.first_inverse_calculation = True
        self.errors_in_connection = 0
        self.last_callback_time = 0  # 0 means no callback yet
        self.initialized = True

    def __del__(self):
        zero_velocities = np.zeros(np.shape(self.list_joints_handles))
        # TODO: Here take care of mujoco as well (get velocities from setJointTargetVelocities and set them before exit
        self.__setJointsTargetVelocity(zero_velocities)
        if self.initialized:
            for dummy_handle in self.dummy_targets_handles:
                self.simulator.removeObject(dummy_handle)

    def __createTargetDummies(self):
        dummy_targets = []
        for i in range(0, self.tasks_count):
            dummy_target = self.simulator.createDummy(0.02,
                                                      [255 * (i % 3), 255 * ((i + 1) % 3), 255 * ((i + 2) % 3), 255])
            dummy_targets.append(dummy_target)
        return dummy_targets

    def __updateWeightMatrixInverse(self):
        weight_matrix = np.identity(self.DOF_count)
        for index, joint_handle in enumerate(self.list_joints_handles):
            result, joint_position = self.simulator.getJointPosition(joint_handle)
            if not result:  # failed
                continue
            joint_velocity = self.joint_velocity[index]
            joint_max, joint_min = self.joints_limits[index]
            joint_middle = (joint_max + joint_min) / 2.0
            going_away = bool((joint_position > joint_middle and joint_velocity < 0) or
                              (joint_position < joint_middle and joint_velocity > 0))
            if going_away:
                w = 1.0
            else:
                performance_gradient = (((joint_max - joint_min) ** 2) * (2.0 * joint_position - joint_max - joint_min)
                                        ) / float(
                    4.0 * ((joint_max - joint_position) ** 2) * ((joint_position - joint_min) ** 2)
                    + 0.0000001)
                w = 1.0 + abs(performance_gradient)
            weight_matrix[index, index] = w
        self.weight_matrix_inv = np.linalg.inv(weight_matrix)

    def __updateTargetDummiesPoses(self):
        if self.simulator.type != SimulatorType.VREP:
            return
        last_pose = self.last_human_hand_part_pose.copy()
        for index, dummy_handle in enumerate(self.dummy_targets_handles):
            start_index = index * 3
            end_index = start_index + 3
            dummy_position_list = last_pose[start_index:end_index].tolist()
            self.simulator.setObjectPosition(dummy_handle, self.hand_base_handle, dummy_position_list)

    def __setJointsTargetVelocity(self, joints_velocities):
        if self.simulator.type == SimulatorType.MUJOCO:
            return joints_velocities
        elif self.simulator.type == SimulatorType.VREP:
            for index, velocity in enumerate(joints_velocities):
                self.simulator.setJointTargetVelocity(self.list_joints_handles[index], velocity,
                                                      self.first_inverse_calculation)
            return None
        else:
            raise ValueError

    def __getError(self, index=None):
        if index is None:
            current_pose = self.simulator.simulationObjectsPose(self.task_descriptor_handles)
            return self.last_human_hand_part_pose - current_pose
        else:
            current_pose = self.simulator.simulationObjectsPose([self.task_descriptor_handles[index]])
            return self.last_human_hand_part_pose[index * 3:index * 3 + 3] - current_pose

    def getAllTaskDescriptorsErrors(self):
        error = 0.
        for index, _ in enumerate(self.task_descriptor_handles):
            error = error + np.linalg.norm(self.__getError(index))
        return error

    def taskPrioritization(self):
        self.__updateWeightMatrixInverse()
        pseudo_inverse_jacobians, jacobians = self.__getPseudoInverseForTaskPrioritization()
        q_vel = np.zeros(self.DOF_count)
        multiplier = np.identity(self.DOF_count)
        for index, task_handle in enumerate(self.task_descriptor_handles):
            error = self.__getError(index)
            q_vel = q_vel + np.dot(np.dot(multiplier, pseudo_inverse_jacobians[index]),
                                   (self.human_hand_vel[index * 3:index * 3 + 3] + np.dot(self.K_matrix, error)))
            multiplier = np.dot(multiplier,
                                np.identity(self.DOF_count) - np.dot(pseudo_inverse_jacobians[index], jacobians[index]))
        self.joint_velocity = q_vel
        return self.__setJointsTargetVelocity(self.joint_velocity)

    def taskAugmentation(self):
        # TODO: the same thing as in taskPrioritization to do here
        error = self.__getError()
        self.__updateWeightMatrixInverse()
        pseudo_inverse_jacobian = self.__getPseudoInverseForTaskAugmentation()
        q_vel = np.dot(pseudo_inverse_jacobian, (self.human_hand_vel + np.dot(self.K_matrix, error)))
        self.joint_velocity = q_vel
        return self.__setJointsTargetVelocity(self.joint_velocity)

    def __getPseudoInverseForTaskPrioritization(self):
        whole_jacobian = self.jacobian_calculation.getJacobian()
        jacobians = []
        pseudo_jacobian_inverses = []
        for task_index, _ in enumerate(self.task_descriptor_handles):
            this_jacobian = whole_jacobian[..., task_index * 3:task_index * 3 + 3].T
            jacobians.append(this_jacobian)
            this_pseudo_jacobian_inverse = np.linalg.multi_dot([self.weight_matrix_inv, this_jacobian.T, np.linalg.inv(
                np.linalg.multi_dot([this_jacobian, self.weight_matrix_inv, this_jacobian.T]) + self.damping_matrix)])
            pseudo_jacobian_inverses.append(this_pseudo_jacobian_inverse)
        return pseudo_jacobian_inverses, jacobians

    def __getPseudoInverseForTaskAugmentation(self):
        if len(self.task_descriptor_handles) != 2:
            rospy.logerr("Task augmentation works currently only with 2 target handles. Current count: %d. Exiting.",
                         len(self.task_descriptor_handles))
            exit(1)
        jacobian = self.jacobian_calculation.getJacobian()
        jacobian = np.concatenate((jacobian[..., 0:3].T, jacobian[..., 3:6].T), axis=0)
        return np.linalg.multi_dot([self.weight_matrix_inv, jacobian.T, np.linalg.inv(
            np.linalg.multi_dot([jacobian, self.weight_matrix_inv, jacobian.T]) + self.damping_matrix)])

    def executeControl(self):
        if self.task_prioritization:
            joints_velocities = self.taskPrioritization()
        else:
            joints_velocities = self.taskAugmentation()
        self.first_inverse_calculation = False
        if self.simulator.type == SimulatorType.VREP:
            return None
        elif self.simulator.type == SimulatorType.MUJOCO:
            joint_velocity_dict = {}
            for index, velocity in enumerate(joints_velocities):
                joint_velocity_dict[self.simulator.getJointIndex(self.list_joints_handles[index])] = velocity
            return joint_velocity_dict
        else:
            raise ValueError

    def newPositionFromHPE(self, new_data, alpha):
        current_time = time.time()
        hand_part_poses = []
        for index in self.task_descriptor_equivalent_hpe_indices:
            hand_part_poses.append(new_data.joints_position[index])
        HPE_hand_part_poses = np.concatenate(hand_part_poses)
        temp_new_HPE_hand_pose = HPE_hand_part_poses * alpha + self.last_human_hand_part_pose * (1 - alpha)
        if self.last_callback_time != 0:  # TODO: maybe here we can make better with calculating with first iteration
            self.human_hand_vel = (temp_new_HPE_hand_pose - self.last_human_hand_part_pose) / (
                    current_time - self.last_callback_time)
            self.last_callback_time = current_time
        else:
            self.last_callback_time = current_time
        self.last_human_hand_part_pose = temp_new_HPE_hand_pose
        self.__updateTargetDummiesPoses()

    def getName(self):
        return self.name

    def taskDescriptorsCount(self):
        return len(self.task_descriptor_handles)


class Hand:
    def __init__(self, alpha, simulator):
        self.simulator = simulator

        index_finger = HandPart(['IMCP_side_joint', 'IMCP_front_joint', 'IPIP_joint', 'IDIP_joint'],
                                'ITIP_tip',
                                [['ITIP_tip', 'IPIP_joint'], ['IMCP_side_joint', 'IMCP_side_joint'], [11, 9]],
                                [[10., -10.], [100., 0.], [90., 0.], [90., 0.]], ConfigurationType.finger, 'index',
                                self.simulator)
        middle_finger = HandPart(['MMCP_side_joint', 'MMCP_front_joint', 'MPIP_joint', 'MDIP_joint'],
                                 'MTIP_tip',
                                 [['MTIP_tip', 'MPIP_joint'], ['MMCP_side_joint', 'MMCP_side_joint'], [14, 12]],
                                 [[10., -10.], [100., 0.], [90., 0.], [90., 0.]], ConfigurationType.finger, 'middle',
                                 self.simulator)
        ring_finger = HandPart(['RMCP_side_joint', 'RMCP_front_joint', 'RPIP_joint', 'RDIP_joint'],
                               'RTIP_tip',
                               [['RTIP_tip', 'RPIP_joint'], ['RMCP_side_joint', 'RMCP_side_joint'], [17, 15]],
                               [[10., -10.], [100., 0.], [90., 0.], [90., 0.]], ConfigurationType.finger, 'ring',
                               self.simulator)
        pinkie_finger = HandPart(['metacarpal_joint', 'PMCP_side_joint', 'PMCP_front_joint', 'PPIP_joint',
                                  'PDIP_joint'], 'PTIP_tip',
                                 [['PTIP_tip', 'PPIP_joint'], ['metacarpal_joint', 'metacarpal_joint'], [20, 18]],
                                 [[45., 0.], [10., -10.], [100., 0.], [90., 0.], [90., 0.]], ConfigurationType.pinkie,
                                 'pinkie', self.simulator)
        thumb_finger = HandPart(['TMCP_rotation_joint', 'TMCP_front_joint', 'TPIP_side_joint', 'TPIP_front_joint',
                                 'TDIP_joint'], 'TTIP_tip',
                                [['TTIP_tip', 'TPIP_front_joint'], ['TMCP_rotation_joint', 'TMCP_rotation_joint'],
                                 [8, 6]],
                                [[60., -60.], [70., 0.], [30., -30.], [12., -12.], [90, 0]], ConfigurationType.thumb,
                                'thumb', self.simulator)

        self.alpha = alpha
        self.hand_parts_list = (index_finger, middle_finger, ring_finger, pinkie_finger, thumb_finger)
        self.error_calculation = ErrorCalculation(list(self.hand_parts_list),
                                                  [['IPIP_joint', 'IDIP_joint', 'ITIP_tip'],
                                                   ['MPIP_joint', 'MDIP_joint', 'MTIP_tip'],
                                                   ['RPIP_joint', 'RDIP_joint', 'RTIP_tip'],
                                                   ['PPIP_joint', 'PDIP_joint', 'PTIP_tip'],
                                                   ['TPIP_front_joint', 'TDIP_joint', 'TTIP_tip']],
                                                  [[9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20],
                                                   [6, 7, 8]], 10, alpha, self.simulator)

    def controlOnce(self):
        for hand_part in self.hand_parts_list:
            hand_part.executeControl()
        self.error_calculation.calculateError()

    def getControlOnce(self, frequency):
        action_dict = {}
        for hand_part in self.hand_parts_list:
            action_dict.update(hand_part.executeControl())  # given as velocities
        # for key, value in action_dict.items():
        #     action_dict[key] = value / frequency * 5  # integrate the velocity

        complete_action_vector = self.simulator.getHandBaseAction()
        complete_action_vector = np.pad(complete_action_vector, (0, self.simulator.getNumberOfJoints() -
                                                                 complete_action_vector.size), 'constant',
                                        constant_values=0)
        for k, v in action_dict.items():
            complete_action_vector[k] = v  #+ self.simulator.getJointIndexPosition(k)  # add position step to current
        return complete_action_vector

    def newPositionFromHPE(self, new_data):
        for hand_part in self.hand_parts_list:
            hand_part.newPositionFromHPE(new_data, self.alpha)
        self.error_calculation.newPositionFromHPE(new_data)
