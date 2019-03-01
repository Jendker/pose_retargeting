#!/usr/bin/env python

import rospy
from mapper import Mapper
from dl_pose_estimation.msg import JointsPosition


def callback(data):
    # rospy.loginfo("%s is age: %d" % (data.header.frame_id, data.header.stamp))
    rospy.loginfo("")


def listener():
    rospy.init_node('pose_mapping_vrep')
    mapper = Mapper()
    rospy.Subscriber("/dl_pose_estimation/joints_position", JointsPosition, mapper.callback)

    mapper.execute()  # keeps program running in loop


if __name__ == '__main__':
    listener()
