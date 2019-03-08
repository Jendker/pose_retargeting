import vrep
from enum import Enum
import sympy as sp
import numpy as np


class ConfigurationType(Enum):
    finger = 1


class JacobianCalculation:
    def __init__(self, clientID, joint_handles, configuration_type):
        self.clientID = clientID
        self.joint_handles = joint_handles
        self.q = sp.symbols('q_0:{}'.format(len(joint_handles)))

        transformations = []

        if configuration_type == ConfigurationType.finger:
            joint_positions = []

            for index, joint_handle in enumerate(self.joint_handles):
                _, this_joint_position = vrep.simxGetObjectPosition(self.clientID, joint_handle, -1,
                                                                    vrep.simx_opmode_blocking)
                vector_this_pose = np.array(this_joint_position)
                joint_positions.append(vector_this_pose)
                angle = -self.q[index]  # - angle ! the joints turn clockwise
                if index == 0:
                    transformations.append(sp.Matrix([[1, 0, 0, vector_this_pose[0]],
                                                      [0, 1, 0, vector_this_pose[1]],
                                                      [0, 0, 1, vector_this_pose[2]],
                                                      [0, 0, 0, 1]]))
                    transformations.append(sp.Matrix([[-1, 0, 0, 0],
                                                      [0, -1, 0, 0],
                                                      [0, 0, 1, 0],
                                                      [0, 0, 0, 1]]))
                    transformations.append(sp.Matrix([[sp.cos(angle), 0, sp.sin(angle), 0],
                                                      [0, 1, 0, 0],
                                                      [-sp.sin(angle), 0, sp.cos(angle), 0],
                                                      [0, 0, 0, 1]]))
                    continue
                length = np.linalg.norm(joint_positions[index] - joint_positions[index - 1])
                if index == 1:
                    length = 0.
                transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                  [0, sp.cos(angle), -sp.sin(angle), sp.sin(angle) * length],
                                                  [0, sp.sin(angle), sp.cos(angle), sp.cos(angle) * length],
                                                  [0, 0, 0, 1]]))
            T = sp.eye(4)
            for index, transformation in enumerate(transformations):
                T = T * transformation
            T = sp.simplify(T)
            self.T = T
            self.jacobian = T[0:3, 3].T.jacobian(sp.Matrix(self.q)).T
        else:
            print("Configuration not defined")

    def getJacobian(self):
        self.printTransformation()
        joint_positions = []
        for joint_handle in self.joint_handles:
            _, this_joint_position = vrep.simxGetObjectPosition(self.clientID, joint_handle, -1,
                                                                vrep.simx_opmode_blocking)
            _, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_oneshot_wait)
            joint_positions.append(joint_position)
        return np.array(self.jacobian.subs(zip(self.q, joint_positions)), dtype='float')
        # return np.array(self.jacobian.subs(zip(self.q, joint_positions))).astype(dtype='float64')

    def printTransformation(self):
        joint_positions = []
        for joint_handle in self.joint_handles:
            _, this_joint_position = vrep.simxGetObjectPosition(self.clientID, joint_handle, -1,
                                                                vrep.simx_opmode_blocking)
            _, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_oneshot_wait)
            joint_positions.append(joint_position)
        print(self.T.subs(zip(self.q, joint_positions))[0:3, 3])
