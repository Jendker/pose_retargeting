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

    def __eulerAnglesToRotationMatrix(self, theta):

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

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
        rotation_matrix = np.identity(3) + skew_mat + np.linalg.multi_dot([skew_mat, skew_mat]) * (1.0/1.0+c)
        return rotation_matrix

    def __euclideanTransformation(self, rotationMatrix, transformationVector):
        top = np.concatenate((rotationMatrix, transformationVector[:, np.newaxis]), axis=1)
        return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)

    def __transformFrame(self, data, to_frame):
        points_return = []
        try:
            transform = self.tf_buffer.lookup_transform(to_frame,
                                                        data.header.frame_id, #source frame
                                                        rospy.Time(0), #get the tf at first available time
                                                        rospy.Duration(1.0)) #wait for max 1 second
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print('Exception on transform lookup!')
            return data
        for point in data.joints_position:
            point_transformed = Point()
            point_transformed.x = point.x + transform.transform.translation.x
            point_transformed.y = point.y + transform.transform.translation.y
            point_transformed.z = point.z + transform.transform.translation.z
            points_return.append(point_transformed)
        data_return = data
        data_return.joints_position = points_return
        data_return.header.frame_id = to_frame
        return data_return

    def __transformHandToOrigin(self, data, to_frame):
        data = self.__transformFrame(data, to_frame)
        position_palm_base = self.__getPositionVectorForDataIndex(data, 0)
        position_knuckle_middle_finger = self.__getPositionVectorForDataIndex(data, 3)
        position_knuckle_index_finger = self.__getPositionVectorForDataIndex(data, 2)
        given_vector = position_knuckle_middle_finger - position_palm_base
        target_vector = np.array([0, 0, 1])
        rotation_matrix = self.__getRotationMatrixFromVectors(desired_vector=target_vector, given_vector=given_vector)
        _, sim_list_position_knuckle_index_finger = vrep.simxGetObjectPosition(self.clientID, self.finger_tip_handle, -1,
                                                    vrep.simx_opmode_blocking)
        transformation_vector = self.__transformationVector(position_knuckle_index_finger,
                                                            np.array(sim_list_position_knuckle_index_finger))
        euclidean_transformation_matrix = self.__euclideanTransformation(rotation_matrix, transformation_vector)
        new_data = []
        for point in self.__dataToPointsList(data):
            extended_point = np.concatenate((point, [1]))
            new_data.append(np.dot(euclidean_transformation_matrix, extended_point)[0:3])
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
        message.header.stamp = rospy.get_rostime()
        message.color.r = 1.0
        # message.color.g = 1.0
        message.color.a = 1.0
        message.scale.x = message.scale.y = message.scale.z = 0.01
        message.type = message.LINE_STRIP

        index_finger_points = [self.last_data[2]]
        index_finger_points.extend(self.last_data[9:12])
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
        t.child_frame_id = "vrep_hand"

        position_knuckle_index_finger = data.joints_position[2]
        vectorized_index_knuckle = self.__getPositionVectorForDataIndex(data, 2)
        vectorized_palm_base = self.__getPositionVectorForDataIndex(data, 0)
        vectorized_middle_knuckle = self.__getPositionVectorForDataIndex(data, 3)
        vectorized_ring_knuckle = self.__getPositionVectorForDataIndex(data, 4)
        vector_hand = vectorized_middle_knuckle - vectorized_palm_base
        vector_hand2 = vectorized_index_knuckle - vectorized_ring_knuckle

        t.transform.translation.x = position_knuckle_index_finger.x
        t.transform.translation.y = position_knuckle_index_finger.y
        t.transform.translation.z = position_knuckle_index_finger.z

        vector_hand_x = np.array([vector_hand[1], vector_hand[2]])
        vector_hand_x = vector_hand_x / np.linalg.norm(vector_hand_x)
        vector_hand_y = np.array([vector_hand[0], vector_hand[2]])
        vector_hand_y = vector_hand_y / np.linalg.norm(vector_hand_y)
        vector_hand_z = np.array([vector_hand2[0], vector_hand2[1]])
        vector_hand_z = vector_hand_z / np.linalg.norm(vector_hand_z)

        x_y_reference = np.array([0, 1])
        z_reference = np.array([1, 0])
        x_angle = math.acos(np.dot(x_y_reference, vector_hand_x))
        y_angle = math.acos(np.dot(x_y_reference, vector_hand_y))
        z_angle = math.acos(np.dot(z_reference, vector_hand_z))

        q = tf_conversions.transformations.quaternion_from_euler(x_angle, y_angle, z_angle)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)

    def callback(self, data):
        current_time = time.time()
        self.__publishTransformation(data)
        target_frame = 'camera_link'
        self.last_data = self.__transformHandToOrigin(data, target_frame)
        HPE_finger_tip_pose = self.last_data[11]
        if self.last_callback_time != 0:
            self.human_hand_vel = (HPE_finger_tip_pose - self.last_human_hand_pose) / current_time
            self.last_callback_time = current_time
        else:
            self.last_callback_time = current_time
        self.last_human_hand_pose = HPE_finger_tip_pose
        self.__publishMarkers(target_frame)

    def execute(self):
        scheduler = sched.scheduler(time.time, time.sleep)
        scheduler.enter(self.sampling_time, 1, self.__setFingerTarget, (scheduler,))
        scheduler.run()
