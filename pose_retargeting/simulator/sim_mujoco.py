from .simulator import Simulator
import mujoco_py
import pose_retargeting.vrep as vrep
import six
import mj_envs
import numpy as np
from pose_retargeting.transformations import quaternion_from_matrix
from pose_retargeting.joint_handles_dict import JointHandlesDict
from pose_retargeting.jacobians.jacobian_calculation_mujoco import JacobianCalculationMujoco


def euclideanTransformation(rotationMatrix, transformationVector):
    top = np.concatenate((rotationMatrix, transformationVector[:, np.newaxis]), axis=1)
    return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)


class Mujoco(Simulator):
    def __init__(self, env):
        super().__init__()
        self.name = 'mujoco'
        self.env = env
        self.last_observations = []
        self.model = env.model
        self.data = env.model.data
        self.joint_handles_dict = JointHandlesDict(self)
        hand_base_handle = 'rh_wrist'
        self.hand_base_index = self.model.body_names.index(six.b(hand_base_handle))
        # self.handle_index_pairs = handle_index_pairs

    # def __get_body_xmat(self, body_name):
    #     idx = self.model.body_names.index(six.b(body_name))
    #     return self.model.data.xmat[idx].reshape((3, 3))

    def __getTransformationMatrixToBase(self):
        rotation_matrix = self.model.data.xmat[self.hand_base_index].reshape((3, 3))
        translation = self.model.data.body_xpos[self.hand_base_index].reshape((3, 1))
        return euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation))

    def __getTransformationMatrix(self, handle):
        idx = self.model.body_names.index(six.b(handle))
        rotation_matrix = self.model.data.xmat[idx].reshape((3, 3))
        translation = self.model.data.body_xpos[idx].reshape((3, 1))
        return euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation))
    
    def jacobianCalculation(self, *argv, **kwargs):
        return JacobianCalculationMujoco(*argv, **kwargs)

    def simulationObjectsPose(self, handles, mode=vrep.simx_opmode_buffer):
        if mode != vrep.simx_opmode_buffer:
            return
        current_pos = []
        transformation_matrix = self.__getTransformationMatrixToBase()
        for handle in handles:
            idx = self.model.body_names.index(six.b(handle))
            this_current_pos = self.model.data.body_xpos[idx].reshape((3, 1))
            current_pos.extend(np.dot(transformation_matrix, np.append(this_current_pos, [1]))[0:3])
        return np.array(current_pos)

    def getJointPosition(self, joint_handle, mode=vrep.simx_opmode_buffer):
        if mode != vrep.simx_opmode_buffer:
            return
        idx = self.model.joint_names.index(six.b(joint_handle))
        return [True, self.model.data.qpos[idx]]

    def getObjectPosition(self, handle, parent_handle, mode=None):
        if parent_handle != -1:
            transformation_matrix = self.__getTransformationMatrix(parent_handle)
        else:
            transformation_matrix = np.identity(4)
        idx = self.model.body_names.index(six.b(handle))
        current_pos = self.model.data.body_xpos[idx].reshape((3, 1))
        return np.dot(transformation_matrix, np.append(current_pos, [1]))[0:3]

    def getObjectPositionWithReturn(self, handle, parent_handle, mode=None):
        return [True, self.getObjectPosition(handle, parent_handle, mode)]

    def setJointTargetVelocity(self, handle, velocity, disable_warning_on_no_connection):
        raise NotImplementedError

    def setObjectPosition(self, handle, base_handle, position_to_set):
        raise NotImplementedError  # not needed

    def getObjectQuaternion(self, handle, parent_handle, mode):
        raise NotImplementedError  # not needed

    def setObjectQuaternion(self, handle, parent_handle, quaternion_to_set):
        raise NotImplementedError  # not needed

    def removeObject(self, handle):
        pass

    def createDummy(self, size, color):
        return None
