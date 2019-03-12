import vrep
import random
from time import sleep
from std_msgs.msg import String
import rospy


class BugTest:
    def __init__(self):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', 19998, True, True, 5000, 5)  # Connect to V-REP
        self.new_pose = [0, 0, 0]
        _, self.dummy_handle = vrep.simxCreateDummy(self.clientID, 0.02, [255, 255, 255, 255], vrep.simx_opmode_blocking)

    def callback(self, _):
        self.new_pose = [random.random(), random.random(), random.random()]

    def execute(self):
        while not rospy.is_shutdown():
            self.execute_once()
            sleep(0.033)
        vrep.simxRemoveObject(self.clientID, self.dummy_handle, vrep.simx_opmode_blocking)

    def execute_once(self):
        new_pose = self.new_pose[:]  # copy to make sure, that the values are not changed when passing with vrep function
        vrep.simxSetObjectPosition(self.clientID, self.dummy_handle, -1, new_pose, vrep.simx_opmode_oneshot)


def run():
    rospy.init_node('bug_vrep')
    bug_tester = BugTest()
    rospy.Subscriber("test_topic", String, bug_tester.callback)

    bug_tester.execute()  # keeps program running in loop


if __name__ == '__main__':
    run()
