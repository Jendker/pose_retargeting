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
                joint_positions.append(np.array(this_joint_position))
                angle = self.q[index]
                if index == 0:
                    transformations.append(sp.Matrix([[sp.cos(angle), 0, -sp.sin(angle), this_joint_position[0]],
                                                      [0, 1, 0, this_joint_position[1]],
                                                      [sp.sin(angle), 0, sp.cos(angle), this_joint_position[2]],
                                                      [0, 0, 0, 1]]))  # we get -angle
                    continue
                length = np.linalg.norm(joint_positions[index] - joint_positions[index - 1])
                if index == 1:
                    length = 0.
                transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                  [0, sp.cos(angle), sp.sin(angle), -sp.sin(angle) * length],
                                                  [0, -sp.sin(angle), sp.cos(angle), sp.cos(angle) * length],
                                                  [0, 0, 0, 1]]))  # we get -angle
            T = sp.eye(4)
            for index, transformation in enumerate(transformations):
                T = T * transformation
            self.jacobian = T[0:3, 3].T.jacobian(sp.Matrix(self.q)).T
            print(self.jacobian)
        else:
            print("Configuration not defined")

    def getJacobian(self):
        joint_positions = []
        for joint_handle in self.joint_handles:
            _, this_joint_position = vrep.simxGetObjectPosition(self.clientID, joint_handle, -1,
                                                                vrep.simx_opmode_blocking)
            _, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_oneshot_wait)
            joint_positions.append(joint_position)
        return self.jacobian.subs(zip(self.q, joint_positions))
