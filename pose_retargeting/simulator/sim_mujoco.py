from .simulator import Simulator
import mujoco_py
import pose_retargeting.vrep as vrep
import six
import mj_envs
import numpy as np
import pose_retargeting.transformations as transformations
from pose_retargeting.joint_handles_dict import JointHandlesDict
from pose_retargeting.jacobians.jacobian_calculation_mujoco import JacobianCalculationMujoco


def euclideanTransformation(rotationMatrix, transformationVector):
    top = np.concatenate((rotationMatrix, transformationVector[:, np.newaxis]), axis=1)
    return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)


class Mujoco(Simulator):
    def __init__(self, env):
        super().__init__()
        self.name = 'mujoco'
        self.env = env.env.env
        self.last_observations = []
        self.model = self.env.model
        self.data = self.env.data
        self.joint_handles_dict = JointHandlesDict(self)
        hand_base_handle = 'rh_wrist'
        self.hand_base_index = self.model.body_names.index(hand_base_handle)
        self.hand_position = self.getObjectPosition(hand_base_handle, -1)
        self.hand_orientation = transformations.euler_from_quaternion(self.getObjectQuaternion(hand_base_handle))
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
        idx = self.model.body_names.index(handle)
        current_pos = self.data.body_xpos[idx].reshape((3, 1))
        return np.dot(transformation_matrix, np.append(current_pos, [1]))[0:3]

    def getObjectQuaternion(self, handle, **kwargs):
        if kwargs['parent_handle']:
            if kwargs['parent_handle'] != -1:
                raise NotImplementedError
        idx = self.model.body_names.index(handle)
        return self.model.data.body_xquat[idx].reshape((4, 1))

    def getObjectPositionWithReturn(self, handle, parent_handle, mode=None):
        return [True, self.getObjectPosition(handle, parent_handle, mode)]

    def setJointTargetVelocity(self, handle, velocity, disable_warning_on_no_connection):
        raise NotImplementedError

    def setObjectPosition(self, handle, base_handle, position_to_set):
        raise NotImplementedError  # not needed

    def setObjectQuaternion(self, handle, parent_handle, quaternion_to_set):
        raise NotImplementedError  # not needed

    def setHandPositionAndQuaternion(self, target_position, target_quaternion):
        self.hand_position = target_position
        self.hand_orientation = transformations.euler_from_quaternion(target_quaternion)

    def removeObject(self, handle):
        pass

    def createDummy(self, size, color):
        return None        
    
    def getJointHandleIndex(self, joint_handle):
        return self.model.joint_names.index(self.joint_handles_dict.getHandle(joint_handle))

    def getJointBodyName(self, joint_handle):
        return self.joint_handles_dict.getJointBodyHandle(joint_handle)

    def getJacobianFromBodyName(self, body_name):
        return self.data.get_body_jacp(body_name).reshape(3, -1)

    def getHandBaseAction(self):
        return dict(zip(range(0, 6), np.concatenate((self.hand_position, self.hand_orientation))))

    def getNumberOfJoints(self):
        return self.data.qval.size
