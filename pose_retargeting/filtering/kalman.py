from pykalman import KalmanFilter, UnscentedKalmanFilter
from geometry_msgs.msg import Point
import numpy as np
from pose_retargeting.simulator.data.sample_data import SampleData
import pickle
import os
import time


def getObservationMatrix():
    ret = []
    for i in range(0, 63):
        this_vec = np.zeros(63*2)
        this_vec[i*2:i*2+2] = [1., 1.]
        ret.append(this_vec.tolist())
    return ret


def getTransitionMatrix():
    ret = np.zeros((63*2, 63*2))
    for i in range(0, 63):
        ret[i*2, i*2: i*2+2] = 1
        ret[i*2+1, i*2+1] = 1
    return ret


def flattenHandPoints(data):
    flattened_data = []
    for joint_point in data.joints_position:
        flattened_data.extend([joint_point.x, joint_point.y, joint_point.z])
    return tuple(flattened_data)


def packHandPoints(data, kalman_filter_output):
    for i in range(0, 21):
        point = Point()
        point.x = kalman_filter_output[i * 6]
        point.y = kalman_filter_output[i * 6 + 1]
        point.z = kalman_filter_output[i * 6 + 2]
        data.joints_position[i] = point
    return data


def getVelocityMask():
    ret = np.zeros((63 * 2, 63 * 2))
    for i in range(0, 63):
        ret[i*2, i*2: i*2+2] = 1
        ret[i*2+1, i*2+1] = 1
    return ret


class Kalman:
    def __init__(self):
        observation_matrix = getObservationMatrix()
        transition_matrix = getTransitionMatrix()

        script_path = os.path.realpath(__file__)
        self.data_path = os.path.dirname(script_path) + '/simulator/data/kalman_filter.pkl'
        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as handle:
                self.kf = pickle.load(handle)
        else:
            self.kf = KalmanFilter(n_dim_obs=63, observation_matrices=observation_matrix, transition_matrices=transition_matrix,
                                   observation_covariance=np.identity(63) * 0.2,
                                   em_vars=['transition_covariance', 'initial_state_covariance',
                                            'initial_state_mean'])
            sample_data = SampleData()
            measurements = np.asarray(sample_data.getData())
            self.kf.em(measurements, n_iter=5)
            with open(self.data_path, 'wb') as handle:
                pickle.dump(self.kf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.last_filtered_state_means = self.kf.initial_state_mean
        self.last_filtered_state_covariances = self.kf.initial_state_covariance
        self.last_time = time.time()
        self.positions = []
        self.velocities = []

    def filter(self, data):
        flattened_data = flattenHandPoints(data)
        self.velocities = flattened_data - self.positions
        self.positions = flattened_data


        # filtered_means, filtered_covariance = self.kf.filter(flattened_data)
        self.last_filtered_state_means, self.last_filtered_state_covariances = \
            self.kf.filter_update(self.last_filtered_state_means, self.last_filtered_state_covariances, flattened_data)
        return packHandPoints(data, self.last_filtered_state_means)

