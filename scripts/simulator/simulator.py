class Simulator:
    def __init__(self):
        pass

    def simulationObjectsPose(self, handle, mode):
        raise NotImplementedError

    def getJointPosition(self, joint_handle, mode):
        raise NotImplementedError

    def getObjectPosition(self, handle, parent_handle, mode):
        raise NotImplementedError

    def getObjectPositionWithReturn(self, handle, parent_handle, mode):
        raise NotImplementedError

    def setObjectPosition(self, handle, base_handle, position_to_set):
        raise NotImplementedError

    def getObjectQuaternion(self, handle, parent_handle, mode):
        raise NotImplementedError

    def setObjectQuaternion(self, handle, parent_handle, quaternion_to_set):
        raise NotImplementedError

    def removeObject(self, handle):
        raise NotImplementedError

    def getHandle(self, handle):
        return self.joint_handles_dict.getHandle(handle)

    def createDummy(self, size, color):
        raise NotImplementedError

    def setJointTargetVelocity(self, handle, velocity, disable_warning_on_no_connection):
        raise NotImplementedError
