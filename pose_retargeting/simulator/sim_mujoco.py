from pose_retargeting.simulator.simulator import Simulator, SimulatorType
import pose_retargeting.vrep as vrep
import numpy as np
import pose_retargeting.rotations_mujoco as rotations
from pose_retargeting.joint_handles_dict import JointHandlesDict
from pose_retargeting.jacobians.jacobian_calculation_mujoco import JacobianCalculationMujoco


def euclideanTransformation(rotation_matrix, transformation_vector):
    if len(transformation_vector.shape) < 2:
        transformation_vector = transformation_vector[:, np.newaxis]
    top = np.concatenate((rotation_matrix, transformation_vector), axis=1)
    return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)


class Mujoco(Simulator):
    def __init__(self, env, env_name):
        super().__init__()
        self.type = SimulatorType.MUJOCO
        self.env = env.env.env
        self.last_observations = []
        self.model = self.env.model
        self.data = self.env.data
        self.env_name = env_name

        self.translate_hand_position = np.array([-1.2, 0, 0])
        self.limits_hand_orientation = ((-3.14, 3.14), (-4.71, 1.57), (-4.71, 1.57))

        self.joint_handles_dict = JointHandlesDict(self)
        self.hand_base_name = self.getHandle('ShadowRobot_base_tip')
        self.hand_base_index = self.model.body_names.index(self.hand_base_name)
        self.hand_target_position = self.getObjectIndexPosition(self.hand_base_index, -1) - self.translate_hand_position
        self.hand_target_orientation = self.quat2euler(  # here euler because we set action as euler
            self.getObjectIndexQuaternion(self.hand_base_index))
        self.scaling_points_knuckles = self.__getKnucklesPositions()
        self.transformation_hand_points = [self.scaling_points_knuckles[0], self.scaling_points_knuckles[1],
                                           self.scaling_points_knuckles[2], np.array([-0.011, -0.005, 0.271])]

    def __getKnucklesPositions(self):  # only to run at startup, because metacarpal angle may change
        ret = []
        knuckles_handles = self.getHandles(['IMCP_side_joint', 'MMCP_side_joint', 'RMCP_side_joint', 'PMCP_side_joint',
                                            'TMCP_rotation_joint'])
        for knuckle_handle in knuckles_handles:
            ret.append(self.getObjectPosition(knuckle_handle, self.hand_base_name))
        return ret

    def __getTransformationMatrixToBase(self):
        rotation_matrix = self.data.body_xmat[self.hand_base_index].reshape((3, 3))
        translation = self.data.body_xpos[self.hand_base_index].reshape((3, 1))
        return euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation))

    def __getTransformationMatrix(self, handle):
        idx = self.model.body_names.index(handle)
        rotation_matrix = self.data.body_xmat[idx].reshape((3, 3))
        translation = self.data.body_xpos[idx].reshape((3, 1))
        return euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation))

    @staticmethod
    def quat2euler(quat):
        return rotations.quat2euler(quat)

    @staticmethod
    def euler2quat(euler):
        return rotations.euler2quat(euler)

    @staticmethod
    def mat2quat(matrix):
        if matrix.shape == (4, 4):
            matrix = matrix[:3, :3]
        return rotations.mat2quat(matrix)
    
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

    def getObjectPositionWithReturn(self, handle, parent_handle, mode=None):
        return [True, self.getObjectPosition(handle, parent_handle, mode=mode)]

    def getObjectIndexPosition(self, index, parent_handle, mode=None):
        if mode == vrep.simx_opmode_streaming:
            return None
        if parent_handle != -1:
            transformation_matrix = self.__getTransformationMatrix(parent_handle)
        else:
            transformation_matrix = np.identity(4)
        current_pos = self.data.body_xpos[index].reshape((3, 1))
        return np.dot(transformation_matrix, np.append(current_pos, [1]))[0:3]

    def setObjectPosition(self, handle, base_handle, position_to_set):
        raise NotImplementedError  # not needed

    def getObjectQuaternion(self, handle, **kwargs):
        try:
            if kwargs['parent_handle'] != -1:
                raise NotImplementedError
        except KeyError:
            pass
        idx = self.model.body_names.index(handle)
        return self.data.body_xquat[idx]

    def setObjectQuaternion(self, handle, parent_handle, quaternion_to_set):
        raise NotImplementedError  # not needed

    def getObjectIndexQuaternion(self, index, **kwargs):
        try:
            if kwargs['parent_handle'] != -1:
                raise NotImplementedError
        except KeyError:
            pass
        return self.data.body_xquat[index]

    def setJointTargetVelocity(self, handle, velocity, disable_warning_on_no_connection):
        raise NotImplementedError

    def setHandTargetPositionAndQuaternion(self, target_position, target_quaternion):
        self.hand_target_position = target_position
        self.hand_target_orientation = self.quat2euler(target_quaternion)

    def getHandTargetPositionAndQuaternion(self):
        return self.hand_target_position, self.euler2quat(self.hand_target_orientation)

    def removeObject(self, handle):
        pass

    def createDummy(self, size, color):
        return None

    def getJointIndex(self, body_name):
        return self.model.joint_names.index(self.getBodyJointName(body_name))
    
    def getJointNameIndex(self, joint_name):
        return self.model.joint_names.index(joint_name)

    def getBodyJointName(self, body_name):
        return self.joint_handles_dict.getBodyJointName(body_name)

    def getJacobianFromBodyName(self, body_name):
        return self.data.get_body_jacp(body_name).reshape(3, -1)

    def applyLimitsOfOrientation(self, old_angles):
        new_angles = old_angles.copy()
        for i in range(0, 3):
            if new_angles[i] < self.limits_hand_orientation[i][0]:
                new_angles[i] += 3.1416 * 2
            elif new_angles[i] > self.limits_hand_orientation[i][1]:
                new_angles[i] -= 3.1416 * 2
        return new_angles

    def updateHandPosition(self, old_position):
        return old_position + self.translate_hand_position

    @staticmethod
    def inverseUpdateHandPosition(old_position):
        new_position = old_position.copy()
        new_position[0] += 1.2
        return new_position

    def getHandBaseAction(self):
        return np.concatenate((self.updateHandPosition(self.hand_target_position),
                               self.applyLimitsOfOrientation(self.hand_target_orientation)))

    def getNumberOfJoints(self):
        return self.data.ctrl.size

    def getJointNameVelocity(self, joint_name):
        idx = self.model.joint_names.index(joint_name)
        return self.data.qvel[idx]

    def getJointIndexVelocity(self, index):
        return self.data.qvel[index]

    def getShiftTransformation(self):
        return euclideanTransformation(np.identity(3), np.zeros(3))

    def getJointLimits(self, body_name):
        idx = self.getJointIndex(body_name)
        return self.env.action_space.high[idx], self.env.action_space.low[idx]
