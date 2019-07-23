import numpy as np
from geometry_msgs.msg import Point


def flattenHandPoints(data):
    flattened_data = []
    for joint_point in data.joints_position:
        flattened_data.extend([joint_point.x, joint_point.y, joint_point.z])
    return tuple(flattened_data)


def packHandPoints(data, flattened_data):
    for i in range(0, 21):
        point = Point()
        point.x = flattened_data[i * 3]
        point.y = flattened_data[i * 3 + 1]
        point.z = flattened_data[i * 3 + 2]
        data.joints_position[i] = point
    return data


def slerp(q0, q1, h):
    theta = np.arccos(np.dot(q0/np.linalg.norm(q0), q1/np.linalg.norm(q1)))
    sin_theta = np.sin(theta)
    return np.sin((1-h) * theta) / sin_theta * q0 + np.sin(h * theta)/sin_theta * q1


def lerp(q0, q1, h):
    return q0 * h + q1 * (1 - h)


def lerp_data(q0, q1, h):
    ret = []
    new_q0 = np.copy(q0)
    new_q1 = np.copy(q1)
    for q0, q1 in zip(new_q0, new_q1):
        ret.append(q0 * h + q1 * (1 - h))
    return np.array(ret)


def lerp_hand_data(hand_data, old_flattened_hand_data, h):
    flattened_data = flattenHandPoints(hand_data)
    filtered_flat_data = lerp_data(flattened_data, old_flattened_hand_data, h)
    return packHandPoints(hand_data, filtered_flat_data), filtered_flat_data
