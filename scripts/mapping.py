#!/usr/bin/env python

import rospy
from mapper import Mapper
from dl_pose_estimation.msg import JointsPosition


def run():
    rospy.init_node('pose_mapping_vrep')
    mapper = Mapper()
    rospy.Subscriber("/dl_pose_estimation/joints_position", JointsPosition, mapper.callback)

    mapper.execute()  # keeps program running in loop


if __name__ == '__main__':
    run()
