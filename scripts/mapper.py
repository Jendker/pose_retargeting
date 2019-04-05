#!/usr/bin/env python

import vrep
import numpy as np
import time
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
import geometry_msgs.msg
import tf_conversions
from hand import Hand


class Mapper:
    def __init__(self):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        while self.clientID == -1:
            if rospy.is_shutdown():
                return
            rospy.loginfo("No connection to remote API server, retrying...")
            vrep.simxFinish(-1)
            time.sleep(3)
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        rospy.loginfo('Connected to remote API server')

        self.last_callback_time = 0  # 0 means no callback yet
        self.last_data = []
        self.node_frame_name = "hand_vrep"
        self.last_update = time.time()
        self.using_left_hand = rospy.get_param('transformation/left_hand')

        self.simulationFingerLength = 0.096
        self.marker_pub = rospy.Publisher('pose_mapping_vrep/transformed_hand', MarkerArray, queue_size=10)
        self.errors_in_connection = 0
        self.hand = Hand(self.clientID)
        self.sampling_time = 0.001

    def __del__(self):
        del self.hand  # not deleted properly, so executing explicitly
        # Close the connection to V-REP:
        vrep.simxFinish(self.clientID)

    def skew(self, vector):
        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])

    def __transformationVector(self, from_point, to_point):
        return to_point - from_point

    def __getRotationMatrixFromVectors(self, desired_vector, given_vector):
        desired_vector = desired_vector / np.linalg.norm(desired_vector)
        given_vector = given_vector / np.linalg.norm(given_vector)
        rotation_axis = np.cross(given_vector, desired_vector)
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
        points_return = []
        for index, _ in enumerate(data.joints_position):
            vector_transformed = np.dot(transformation_matrix,
                                        np.append(self.__getPositionVectorForDataIndex(data, index), [1]))[0:3]
            points_return.append(vector_transformed)
        data_return = data
        data_return.joints_position = points_return
        data_return.header.frame_id = self.node_frame_name
        data_return.header.stamp = rospy.Time.now()
        return data_return

    def __transformPointWithTransform(self, point, transformation_matrix):
        rotation_matrix = transformation_matrix[0:3, 0:3]
        translation_vector = transformation_matrix[0:3, 3]
        inverse_transformation_matrix = self.__euclideanTransformation(rotation_matrix.T,
                                                                       np.dot(-rotation_matrix.T, translation_vector))
        vector_transformed = np.dot(inverse_transformation_matrix,
                                    np.append(point, [1]))[0:3]
        return vector_transformed

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

    def __publishMarkers(self):
        message = MarkerArray()
        lines = [[2, 9, 10, 11], [3, 12, 13, 14], [4, 15, 16, 17], [5, 18, 19, 20], [6, 7, 8], [0, 1, 6, 2, 3, 4, 5, 0]]
        time_now = rospy.Time.now()
        for index, line_points in enumerate(lines):
            line_marker = Marker()
            line_marker.header.frame_id = self.node_frame_name
            line_marker.header.stamp = time_now
            line_marker.id = index
            line_marker.color.r = 1.0
            # message.color.g = 1.0
            line_marker.color.a = 1.0
            line_marker.scale.x = line_marker.scale.y = line_marker.scale.z = 0.01
            line_marker.type = line_marker.LINE_STRIP

            finger_points = [self.last_data.joints_position[x] for x in line_points]

            for point in finger_points:
                message_point = Point()
                message_point.x = point[0]
                message_point.y = point[1]
                message_point.z = point[2]
                line_marker.points.append(message_point)
            message.markers.append(line_marker)
        self.marker_pub.publish(message)

    def __correspondancePointsTransformation(self, from_set, to_set):
        num = from_set.shape[0]
        dim = from_set.shape[1]
        mean_from_set = from_set.mean(axis=0)
        from_zero_mean_set = from_set - mean_from_set
        mean_to_set = to_set.mean(axis=0)
        to_zero_mean_set = to_set - mean_to_set

        D = np.dot(to_zero_mean_set.T, from_zero_mean_set) / num  # eq. 38

        from_set_variance = from_set.var(axis=0).sum()

        U, Sigma, Vt = np.linalg.svd(D)
        rank = np.linalg.matrix_rank(D)
        S = np.identity(dim, dtype=np.double)
        if np.linalg.det(D) < 0:
            S[dim-1, dim-1] = -1.
        if rank == 0:
            return np.full([dim+1, dim+1], np.nan)
        if rank >= dim-1:
            scaling = 1. / from_set_variance * np.trace(np.dot(np.diag(Sigma), S))
            if rank == 2:
                if np.isclose(np.linalg.det(U) * np.linalg.det(Vt), -1., 0.00001):
                    S[dim-1, dim-1] = -1
                else:
                    S[dim-1, dim-1] = 1
            rotation_matrix = np.linalg.multi_dot([U, S, Vt])
            translation = mean_to_set - scaling * np.dot(rotation_matrix, mean_from_set)
            correct_rotation_matrix = rotation_matrix.copy()
            rotation_matrix *= scaling
            return self.__euclideanTransformation(rotation_matrix, translation),  self.__euclideanTransformation(correct_rotation_matrix, translation)
        else:
            rospy.logfatal("Transformation between the points not defined!")
            exit(1)

    def __publishTransformation(self, data):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = data.header.frame_id
        t.child_frame_id = self.node_frame_name

        finger_indices = [2, 3, 4, 5, 0, 1]  # palm base, index, middle, ring, little, thumb - 6
        world_points = []
        for index in finger_indices:
            world_points.append(self.__getPositionVectorForDataIndex(data, index))

        vrep_points = [np.array([0.033, -.0099, 0.352]), np.array([0.011, -0.0099, .356]), np.array([-.011, -.0099, .352]),
                       np.array([-0.033, -.0099, .3436]), np.array([-0.01, -0.0145, 0.27]), np.array([0.03, -0.0145, 0.27])]

        transformation_matrix, correct_rotation_matrix = self.__correspondancePointsTransformation(np.array(world_points), np.array(vrep_points))

        # transform publishing
        t.transform.translation.x = transformation_matrix[0, 3]
        t.transform.translation.y = transformation_matrix[1, 3]
        t.transform.translation.z = transformation_matrix[2, 3]

        q = tf_conversions.transformations.quaternion_from_matrix(correct_rotation_matrix)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)
        return transformation_matrix

    def __mirrorData(self, data):
        new_data = data
        for index, joint_position in enumerate(new_data.joints_position):
            new_data.joints_position[index].x = -new_data.joints_position[index].x

        return data

    def callback(self, data):
        if self.using_left_hand:
            data = self.__mirrorData(data)
        transformation_matrix = self.__publishTransformation(data)
        data = self.__transformDataWithTransform(data, transformation_matrix)
        self.last_data = self.__scaleHandData(data)  # ready to save after scaling

        self.hand.newPositionFromHPE(self.last_data)
        self.__publishMarkers()

    def __executeInverseOnce(self):
        self.hand.controlOnce()

    def execute(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            time_now = time.time()
            self.__executeInverseOnce()
            print (time.time() - time_now)
            time.sleep(self.sampling_time - ((time.time() - start_time) % self.sampling_time))
            self.last_update = time.time()
