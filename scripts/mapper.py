#!/usr/bin/env python

import vrep
import numpy as np
from numpy.linalg import inv
import sched, time
import math
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import geometry_msgs.msg
import tf_conversions


class Mapper:
    def __init__(self):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        if self.clientID != -1:
            print('Connected to remote API server')
        self.K_matrix = np.identity(3)
        self.human_hand_vel = np.zeros(3)
        self.sampling_time = 0.03  # in seconds
        _, self.finger_tip_handle = vrep.simxGetObjectHandle(self.clientID, 'ITIP_tip', vrep.simx_opmode_blocking)
        _, self.IDIP_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'IDIP_joint', vrep.simx_opmode_blocking)
        _, self.IPIP_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'IPIP_joint', vrep.simx_opmode_blocking)
        _, self.IMCP_front_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'IMCP_front_joint',
                                                                   vrep.simx_opmode_blocking)
        _, self.IMCP_side_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'IMCP_side_joint',
                                                                  vrep.simx_opmode_blocking)
        self.last_human_hand_pose = self.__initializeHumanHandPose(self.finger_tip_handle)
        self.last_callback_time = 0
        self.weight_matrix_inv = np.identity(3)
        self.damping_matrix = np.identity(3) * 0.1
        self.last_data = []
        self.node_frame_name = "hand_vrep"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.marker_pub = rospy.Publisher('pose_mapping_vrep/transformed_hand', Marker, queue_size=10)

    def __initializeHumanHandPose(self, handle):
        _, current_pos = vrep.simxGetObjectPosition(self.clientID, handle, -1,
                                                    vrep.simx_opmode_blocking)
        return current_pos

    def __setJointsTargetVelocity(self, joints_velocities):
        list_joints_handles = [self.IDIP_joint_handle, self.IPIP_joint_handle, self.IMCP_front_joint_handle,
                               self.IMCP_side_joint_handle]
        for index, velocity in enumerate(joints_velocities):
            vrep.simxSetJointTargetVelocity(self.clientID, list_joints_handles[index], velocity,
                                            vrep.simx_opmode_oneshot)

    def __getError(self):
        _, current_pos = vrep.simxGetObjectPosition(self.clientID, self.finger_tip_handle, -1,
                                                    vrep.simx_opmode_blocking)
        current_pos_vec = np.array(current_pos)
        return self.last_human_hand_pose - current_pos_vec

    def __getPseudoInverseJacobian(self):
        jacobian = self.__getJacobian()
        return np.linalg.multi_dot([self.weight_matrix_inv, jacobian.T, inv(
            np.linalg.multi_dot([jacobian, self.weight_matrix_inv, jacobian.T]) + self.damping_matrix)])

    def __setFingerTarget(self, scheduler):
        error = self.__getError()
        pseudo_inverse_jacobian = self.__getPseudoInverseJacobian()
        q_vel = np.dot(pseudo_inverse_jacobian, (self.human_hand_vel + np.dot(self.K_matrix, error)))
        self.__setJointsTargetVelocity(q_vel)
        # continue with loop
        scheduler.enter(self.sampling_time, 1, self.__setFingerTarget, (scheduler,))

    def __getJacobian(self):
        empty_buff = bytearray()
        _, dimension, jacobian_vect, _, _ = vrep.simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',
                                                                        vrep.sim_scripttype_childscript,
                                                                        'jacobianIKGroup', [], [],
                                                                        ['IK_Group'], empty_buff,
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
        rotation_matrix = np.identity(3) + skew_mat + np.dot(skew_mat, skew_mat) * (1.0/(1.0+c))
        return rotation_matrix

    def __euclideanTransformation(self, rotationMatrix, transformationVector):
        top = np.concatenate((rotationMatrix, transformationVector[:, np.newaxis]), axis=1)
        return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)

    def __getRotationMatrixFromTransform(self, transform):
        q = transform.transform.rotation
        translation = transform.transform.translation
        tf_conversions.transformations.quaternion_from_matrix


    def __transformDataWithTransform(self, data, transformation_matrix):
        rotation_matrix = transformation_matrix[0:3, 0:3]
        translation_vector = transformation_matrix[0:3, 3]
        inverse_transformation_matrix = self.__euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation_vector))
        points_return = []
        for index, _ in enumerate(data.joints_position):
            vector_transformed = np.dot(inverse_transformation_matrix, np.append(self.__getPositionVectorForDataIndex(data, index), [1]))[0:3]
            points_return.append(vector_transformed)
        data_return = data
        data_return.joints_position = points_return
        data_return.header.frame_id = self.node_frame_name
        data_return.header.stamp = rospy.Time.now()
        return data_return

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
        self.last_data = self.__transformDataWithTransform(data, transformation_matrix)
        HPE_finger_tip_pose = self.last_data.joints_position[11]
        if self.last_callback_time != 0:
            self.human_hand_vel = (HPE_finger_tip_pose - self.last_human_hand_pose) / current_time
            self.last_callback_time = current_time
        else:
            self.last_callback_time = current_time
        self.last_human_hand_pose = HPE_finger_tip_pose
        self.__publishMarkers(self.node_frame_name)

    def execute(self):
        scheduler = sched.scheduler(time.time, time.sleep)
        scheduler.enter(self.sampling_time, 1, self.__setFingerTarget, (scheduler,))
        scheduler.run()
