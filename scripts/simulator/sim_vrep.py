from .simulator import Simulator
import vrep
import numpy as np
import rospy
import time
from joint_handles_dict import JointHandlesDict


class VRep(Simulator):
    def __init__(self):
        super().__init__()
        self.name = 'vrep'

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

        self.joint_handles_dict = JointHandlesDict(self)
        self.hand_base_handle = self.getHandle('ShadowRobot_base_tip')

        self.errors_in_connection = 0

    def __del__(self):
        vrep.simxFinish(self.clientID)

    def simulationObjectsPose(self, handles, mode=vrep.simx_opmode_buffer):
        current_pos = []
        for handle in handles:
            _, this_current_pos = vrep.simxGetObjectPosition(self.clientID, handle, self.hand_base_handle, mode)
            current_pos.extend(this_current_pos)
        return np.array(current_pos)

    def getJointPosition(self, joint_handle, mode=vrep.simx_opmode_buffer):
        result, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, mode)
        return [result == vrep.simx_return_ok, joint_position]

    def getObjectPosition(self, handle, parent_handle, mode):
        return vrep.simxGetObjectPosition(self.clientID, handle, parent_handle, mode)[1]

    def getObjectPositionWithReturn(self, handle, parent_handle, mode):
        result, object_position = vrep.simxGetObjectPosition(self.clientID, handle, parent_handle, mode)
        return [result == vrep.simx_return_ok, object_position]

    def setObjectPosition(self, handle, base_handle, position_to_set):
        vrep.simxSetObjectPosition(self.clientID, handle, base_handle, position_to_set, vrep.simx_opmode_oneshot)

    def getObjectQuaternion(self, handle, parent_handle, mode):
        return vrep.simxGetObjectQuaternion(self.clientID, handle, parent_handle, mode)[1]

    def setObjectQuaternion(self, handle, parent_handle, quaternion_to_set):
        vrep.simxSetObjectQuaternion(self.clientID, handle, parent_handle, quaternion_to_set,
                                       vrep.simx_opmode_oneshot)

    def removeObject(self, handle):
        vrep.simxRemoveObject(self.clientID, handle, vrep.simx_opmode_blocking)

    def createDummy(self, size, color):
        return vrep.simxCreateDummy(self.clientID, size, color, vrep.simx_opmode_blocking)[1]
    
    def setJointTargetVelocity(self, handle, velocity, disable_warning_on_no_connection):
        result = vrep.simxSetJointTargetVelocity(self.clientID, handle, velocity, vrep.simx_opmode_oneshot)
        if result != 0 and not disable_warning_on_no_connection:
            self.errors_in_connection += 1
            if self.errors_in_connection > 10:
                rospy.logwarn("vrep.simxSetJointTargetVelocity return code: %d", result)
                rospy.loginfo("Probably no connection with remote API server. Exiting.")
                exit(0)
            else:
                time.sleep(0.2)

    def getObjectHandle(self, handle_name):
        return vrep.simxGetObjectHandle(self.clientID, handle_name, vrep.simx_opmode_blocking)
