from pose_retargeting.simulator.simulator import Simulator, SimulatorType
import pose_retargeting.vrep as vrep
import numpy as np
import pose_retargeting.transformations as transformations
from pose_retargeting.joint_handles_dict import JointHandlesDict
from pose_retargeting.jacobians.jacobian_calculation_mujoco import JacobianCalculationMujoco


def euclideanTransformation(rotation_matrix, transformation_vector):
    if len(transformation_vector.shape) < 2:
        transformation_vector = transformation_vector[:, np.newaxis]
    top = np.concatenate((rotation_matrix, transformation_vector), axis=1)
    return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)


class Mujoco(Simulator):
    def __init__(self, env):
        super().__init__()
        self.type = SimulatorType.MUJOCO
        self.env = env.env.env
        self.last_observations = []
        self.model = self.env.model
        self.data = self.env.data
        self.joint_handles_dict = JointHandlesDict(self)
        hand_base_name = 'rh_wrist'
        self.hand_base_index = self.model.body_names.index(hand_base_name)
        self.hand_target_position = self.getObjectIndexPosition(self.hand_base_index, -1)
        self.hand_target_orientation = transformations.euler_from_quaternion(  # here euler
            self.getObjectIndexQuaternion(self.hand_base_index))
        # self.handle_index_pairs = handle_index_pairs

    # def __get_body_xmat(self, body_name):
    #     idx = self.model.body_names.index(six.b(body_name))
    #     return self.model.data.body_xmat[idx].reshape((3, 3))

    def __getTransformationMatrixToBase(self):
        rotation_matrix = self.data.body_xmat[self.hand_base_index].reshape((3, 3))
        translation = self.data.body_xpos[self.hand_base_index].reshape((3, 1))
        return euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation))

    def __getTransformationMatrix(self, handle):
        idx = self.model.body_names.index(handle)
        rotation_matrix = self.data.body_xmat[idx].reshape((3, 3))
        translation = self.data.body_xpos[idx].reshape((3, 1))
        return euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation))
    
    def jacobianCalculation(self, *argv, **kwargs):
        return JacobianCalculationMujoco(*argv, **kwargs)

    def simulationObjectsPose(self, body_names, mode=vrep.simx_opmode_buffer):
        if mode != vrep.simx_opmode_buffer and mode != vrep.simx_opmode_blocking:
            return
        current_pos = []
        transformation_matrix = self.__getTransformationMatrixToBase()
        for body_name in body_names:
            idx = self.model.body_names.index(body_name)
            this_current_pos = self.data.body_xpos[idx].reshape((3, 1))
            current_pos.extend(np.dot(transformation_matrix, np.append(this_current_pos, [1]))[0:3])
        return np.array(current_pos)

    def getJointPosition(self, body_name, mode=vrep.simx_opmode_buffer):
        if mode != vrep.simx_opmode_buffer and mode != vrep.simx_opmode_blocking:
            return
        idx = self.model.joint_names.index(self.getBodyJointName(body_name))
        return [True, self.data.qpos[idx]]

    def getJointIndexPosition(self, index):
        assert(len(self.data.qpos) == len(self.data.qvel))  # need to make sure, for some envs this is not the same
        return self.data.qpos[index]

    def getJointNamePosition(self, joint_name):
        idx = self.model.joint_names.index(joint_name)
        return self.data.qpos[idx]

    def getObjectPosition(self, body_name, parent_handle, **kwargs):
        idx = self.model.body_names.index(body_name)
        return self.getObjectIndexPosition(idx, parent_handle)

    def getObjectIndexPosition(self, index, parent_handle, mode=None):
        if mode == vrep.simx_opmode_streaming:
            return None
        if parent_handle != -1:
            transformation_matrix = self.__getTransformationMatrix(parent_handle)
        else:
            transformation_matrix = np.identity(4)
        current_pos = self.data.body_xpos[index].reshape((3, 1))
        return np.dot(transformation_matrix, np.append(current_pos, [1]))[0:3]

    def getObjectQuaternion(self, handle, **kwargs):
        try:
            if kwargs['parent_handle'] != -1:
                raise NotImplementedError
        except KeyError:
            pass
        idx = self.model.body_names.index(handle)
        return self.data.body_xquat[idx]

    def getObjectIndexQuaternion(self, index, **kwargs):
        try:
            if kwargs['parent_handle'] != -1:
                raise NotImplementedError
        except KeyError:
            pass
        return self.data.body_xquat[index]

    def getObjectPositionWithReturn(self, handle, parent_handle, mode=None):
        return [True, self.getObjectPosition(handle, parent_handle, mode=mode)]

    def setJointTargetVelocity(self, handle, velocity, disable_warning_on_no_connection):
        raise NotImplementedError

    def setObjectPosition(self, handle, base_handle, position_to_set):
        raise NotImplementedError  # not needed

    def setObjectQuaternion(self, handle, parent_handle, quaternion_to_set):
        raise NotImplementedError  # not needed

    def setHandTargetPositionAndQuaternion(self, target_position, target_quaternion):
        self.hand_target_position = target_position
        self.hand_target_orientation = transformations.euler_from_quaternion(target_quaternion)

    def removeObject(self, handle):
        pass

    def getHandTargetPositionAndQuaternion(self):
        return self.hand_target_position, transformations.quaternion_from_euler(*self.hand_target_orientation)

    def createDummy(self, size, color):
        return None

    def getJointIndex(self, body_name):
        return self.model.joint_names.index(self.getBodyJointName(body_name))

    def getBodyJointName(self, body_name):
        return self.joint_handles_dict.getBodyJointName(body_name)

    def getJacobianFromBodyName(self, body_name):
        return self.data.get_body_jacp(body_name).reshape(3, -1)

    def getHandBaseAction(self):
        return np.concatenate((self.hand_target_position, self.hand_target_orientation))

    def getNumberOfJoints(self):
        return self.data.qvel.size
