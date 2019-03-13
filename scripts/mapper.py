#!/usr/bin/env python

import vrep
import numpy as np
from numpy.linalg import inv
import time
import math
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import geometry_msgs.msg
import tf_conversions
from jacobian_calculation import JacobianCalculation, ConfigurationType

def degToRad(angle):
    return angle / 180.0 * math.pi


class Mapper:
    def __init__(self):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        self.tasks_count = 3
        if self.clientID != -1:
            rospy.loginfo('Connected to remote API server')
        self.K_matrix = np.identity(3 * self.tasks_count)
        self.human_hand_vel = np.zeros(3 * self.tasks_count)
        self.sampling_time = 0.03  # in seconds
        _, self.finger_tip_handle = vrep.simxGetObjectHandle(self.clientID, 'ITIP_tip', vrep.simx_opmode_blocking)
        _, self.IDIP_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'IDIP_joint', vrep.simx_opmode_blocking)
        _, self.IPIP_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'IPIP_joint', vrep.simx_opmode_blocking)
        _, self.IMCP_front_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'IMCP_front_joint',
                                                                   vrep.simx_opmode_blocking)
        _, self.IMCP_side_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'IMCP_side_joint',
                                                                  vrep.simx_opmode_blocking)
        self.list_joints_handles = [self.IMCP_side_joint_handle, self.IMCP_front_joint_handle,
                                    self.IPIP_joint_handle, self.IDIP_joint_handle]
        self.finger_pose_handles = [self.finger_tip_handle, self.IDIP_joint_handle, self.IPIP_joint_handle]
        self.last_human_hand_pose = self.__simulationObjectsPose(
            self.finger_pose_handles, mode=vrep.simx_opmode_blocking)  # initialize with simulation pose
        all_handles_for_jacobian_calc = self.list_joints_handles[:]
        all_handles_for_jacobian_calc.append(self.finger_tip_handle)
        self.jacobian_calculation = JacobianCalculation(self.clientID, all_handles_for_jacobian_calc, ConfigurationType.finger)
        for joint_handle in self.list_joints_handles:  # initialize streaming
            _, _ = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_streaming)
        for handle in self.finger_pose_handles:
            _, _ = vrep.simxGetObjectPosition(self.clientID, handle, -1, vrep.simx_opmode_streaming)
        joints_limits = [[10., -10.], [100., 0.], [90., 0.], [90., 0.]]
        self.joints_limits = []
        for joint_limits in joints_limits:
            max_angle, min_angle = joint_limits
            self.joints_limits.append([degToRad(max_angle), degToRad(min_angle)])
        self.last_callback_time = 0  # 0 means no callback yet
        self.weight_matrix_inv = np.identity(4)  # with size of the count of DOF
        self.damping_matrix = np.identity(3 * self.tasks_count) * 0.001  # with size of the task descriptor dimension
        self.last_data = []
        self.node_frame_name = "hand_vrep"
        self.first_inverse_calculation = True
        self.dummy_targets_handles = self.__createTargetDummies()
        self.last_update = time.time()

        self.simulationFingerLength = 0.096

        self.marker_pub = rospy.Publisher('pose_mapping_vrep/transformed_hand', Marker, queue_size=10)

    def cleanup(self):
        for dummy_handle in self.dummy_targets_handles:
            vrep.simxRemoveObject(self.clientID, dummy_handle, vrep.simx_opmode_blocking)

    def __createTargetDummies(self):
        dummy_targets = []
        for i in range(0, self.tasks_count):
            _, dummy_target = vrep.simxCreateDummy(self.clientID, 0.02, [255 * (i % 3), 255 * ((i+1) % 3), 255 * ((i+2) % 3), 255], vrep.simx_opmode_blocking)
            dummy_targets.append(dummy_target)
        return dummy_targets

    def __updateTargetDummiesPoses(self):
        last_pose = self.last_human_hand_pose.copy()
        for index, dummy_handle in enumerate(self.dummy_targets_handles):
            start_index = index * 3
            end_index = start_index + 3
            dummy_position_list = last_pose[start_index:end_index].tolist()
            vrep.simxSetObjectPosition(self.clientID, dummy_handle, -1, dummy_position_list,
                                       vrep.simx_opmode_oneshot)

    def __updateWeightMatrixInverse(self):
        weight_matrix = np.identity(4)
        for index, joint_handle in enumerate(self.list_joints_handles):
            result, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_buffer)
            if result != vrep.simx_return_ok:
                continue
            joint_max, joint_min = self.joints_limits[index]
            if joint_max == joint_position or joint_min == joint_position:
                performance_gradient = float("inf")
            else:
                performance_gradient = ((joint_max - joint_min) ** 2 * (
                        2.0 * joint_position - joint_max - joint_min)) / float(
                    4.0 * (joint_max - joint_position) ** 2 * (joint_position - joint_min) ** 2)
            if performance_gradient > 1.0:
                w = 1.0 + performance_gradient
            else:
                w = 1.0
            weight_matrix[index, index] = w
        self.weight_matrix_inv = inv(weight_matrix)

    def __simulationObjectsPose(self, handles, mode=vrep.simx_opmode_buffer):
        current_pos = []
        for handle in handles:
            _, this_current_pos = vrep.simxGetObjectPosition(self.clientID, handle, -1, mode)
            current_pos.extend(this_current_pos)
        return np.array(current_pos)

    def __setJointsTargetVelocity(self, joints_velocities):
        for index, velocity in enumerate(joints_velocities):
            result = vrep.simxSetJointTargetVelocity(self.clientID, self.list_joints_handles[index], velocity,
                                                     vrep.simx_opmode_oneshot)
            if result != 0 and not self.first_inverse_calculation:
                rospy.logwarn("vrep.simxSetJointTargetVelocity return code: %d", result)

    def __getError(self):
        current_pose = self.__simulationObjectsPose(self.finger_pose_handles)
        return self.last_human_hand_pose - current_pose

    def __getPseudoInverseJacobian(self):
        jacobian = self.jacobian_calculation.getJacobian()
        jacobian = np.concatenate((jacobian[..., 0:3].T, jacobian[..., 3:6].T, jacobian[..., 6:9].T), axis=0)
        return np.linalg.multi_dot([self.weight_matrix_inv, jacobian.T, inv(
            np.linalg.multi_dot([jacobian, self.weight_matrix_inv, jacobian.T]) + self.damping_matrix)])

    def __getJacobian(self):
        empty_buff = bytearray()
        _, dimension, jacobian_vect, _, _ = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                                                        vrep.sim_scripttype_childscript,
                                                                        'jacobianIKGroup', self.list_joints_handles, [],
                                                                        ['IK_Index'], empty_buff,
                                                                        vrep.simx_opmode_blocking)
        jacobian = np.array(jacobian_vect).reshape(dimension)
        return jacobian

    def skew(self, vector):
        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])

    def __transformationVector(self, from_point, to_point):
        return to_point - from_point

    def __getRotationMatrixFromVectors(self, desired_vector, given_vector):
        desired_vector = desired_vector / np.linalg.norm(desired_vector)
        given_vector = given_vector / np.linalg.norm(given_vector)
        rotation_axis = np.cross(desired_vector, given_vector)
        c = np.dot(given_vector, desired_vector)
        if c == -1:
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        skew_mat = self.skew(rotation_axis)
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        rotation_matrix = np.identity(3) + skew_mat + np.dot(skew_mat, skew_mat) * (1.0 / (1.0 + c))
        return rotation_matrix

    def __euclideanTransformation(self, rotationMatrix, transformationVector):
        top = np.concatenate((rotationMatrix, transformationVector[:, np.newaxis]), axis=1)
        return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)

    def __transformDataWithTransform(self, data, transformation_matrix):
        rotation_matrix = transformation_matrix[0:3, 0:3]
        translation_vector = transformation_matrix[0:3, 3]
        inverse_transformation_matrix = self.__euclideanTransformation(rotation_matrix.T,
                                                                       np.dot(-rotation_matrix.T, translation_vector))
        points_return = []
        for index, _ in enumerate(data.joints_position):
            vector_transformed = np.dot(inverse_transformation_matrix,
                                        np.append(self.__getPositionVectorForDataIndex(data, index), [1]))[0:3]
            points_return.append(vector_transformed)
        data_return = data
        data_return.joints_position = points_return
        data_return.header.frame_id = self.node_frame_name
        data_return.header.stamp = rospy.Time.now()
        return data_return

    def __getFingerLength(self, data, indices):
        if len(indices) != 4:
            rospy.logerr("Error. Fingers point count does not match 4!")
            return 0.0
        length = 0.0
        for i, index in enumerate(indices):
            if i == 0:
                continue
            length += np.linalg.norm(data.joints_position[index] - data.joints_position[indices[i - 1]])
        return length

    def __scaleHandData(self, data):
        fingers_lenths = []
        fingers_lenths.append(self.__getFingerLength(data, [2, 9, 10, 11]))
        fingers_lenths.append(self.__getFingerLength(data, [4, 15, 16, 17]))
        mean_length = sum(fingers_lenths) / float(len(fingers_lenths))
        scaling_ratio = self.simulationFingerLength / mean_length
        center_point = data.joints_position[2]  # index knuckle as transformation center
        new_data = data
        for i, point in enumerate(data.joints_position):
            new_data.joints_position[i] = (point - center_point) * scaling_ratio + center_point
        return new_data

    def __getPositionVectorForDataIndex(self, data, index):
        joint = data.joints_position[index]
        position = [joint.x, joint.y, joint.z]
        return np.array(position)

    def __dataToPointsList(self, data):
        pointList = []
        for point in data.joints_position:
            this_point = [point.x, point.y, point.z]
            pointList.append(np.array(this_point))
        return pointList

    def __publishMarkers(self, target_frame):
        message = Marker()
        message.header.frame_id = target_frame
        message.header.stamp = rospy.Time.now()
        message.color.r = 1.0
        # message.color.g = 1.0
        message.color.a = 1.0
        message.scale.x = message.scale.y = message.scale.z = 0.01
        message.type = message.LINE_STRIP

        index_finger_points = [self.last_data.joints_position[2]]
        index_finger_points.extend(self.last_data.joints_position[9:12])
        for point in index_finger_points:
            message_point = Point()
            message_point.x = point[0]
            message_point.y = point[1]
            message_point.z = point[2]
            message.points.append(message_point)
        self.marker_pub.publish(message)

    def __publishTransformation(self, data):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = data.header.frame_id
        t.child_frame_id = self.node_frame_name

        position_knuckle_index_finger = data.joints_position[2]
        transform_to_position_base = np.array([-0.033, +0.0099, -0.352])
        vectorized_palm_base = self.__getPositionVectorForDataIndex(data, 0)
        vectorized_middle_knuckle = self.__getPositionVectorForDataIndex(data, 3)
        vector_hand = vectorized_middle_knuckle - vectorized_palm_base

        rotation_matrix = self.__getRotationMatrixFromVectors(np.array([0, 0, 1]), vector_hand)
        rotated_transform_to_position_base = np.dot(rotation_matrix, transform_to_position_base)

        translation_vector = np.array([position_knuckle_index_finger.x + rotated_transform_to_position_base[0],
                                       position_knuckle_index_finger.y + rotated_transform_to_position_base[1],
                                       position_knuckle_index_finger.z + rotated_transform_to_position_base[2]])

        transformation_matrix = self.__euclideanTransformation(rotation_matrix, translation_vector)

        t.transform.translation.x = translation_vector[0]
        t.transform.translation.y = translation_vector[1]
        t.transform.translation.z = translation_vector[2]

        q = tf_conversions.transformations.quaternion_from_matrix(transformation_matrix)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)
        return transformation_matrix

    def callback(self, data):
        current_time = time.time()
        transformation_matrix = self.__publishTransformation(data)
        data = self.__transformDataWithTransform(data, transformation_matrix)
        self.last_data = self.__scaleHandData(data)  # ready to save after scaling
        HPE_finger_tip_pose = np.concatenate((self.last_data.joints_position[11], self.last_data.joints_position[10],
                                              self.last_data.joints_position[9]))
        new_HPE_finger_pose = HPE_finger_tip_pose * 0.2 + self.last_human_hand_pose * 0.8
        if self.last_callback_time != 0:
            self.human_hand_vel = (new_HPE_finger_pose - self.last_human_hand_pose) / (
                    current_time - self.last_callback_time)
            self.last_callback_time = current_time
        else:
            self.last_callback_time = current_time
        self.last_human_hand_pose = new_HPE_finger_pose
        self.__updateTargetDummiesPoses()
        self.__publishMarkers(self.node_frame_name)

    def __executeInverseOnce(self):
        error = self.__getError()
        # self.__updateWeightMatrixInverse()
        pseudo_inverse_jacobian = self.__getPseudoInverseJacobian()
        q_vel = np.dot(pseudo_inverse_jacobian, (self.human_hand_vel + np.dot(self.K_matrix, error)))
        self.__setJointsTargetVelocity(q_vel)
        self.first_inverse_calculation = False

    def execute(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            self.__executeInverseOnce()
            time.sleep(self.sampling_time - ((time.time() - start_time) % self.sampling_time))
            self.last_update = time.time()
        self.cleanup()
