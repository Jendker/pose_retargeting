import vrep
from enum import Enum
import sympy as sp
import numpy as np


class ConfigurationType(Enum):
    finger = 1


class JacobianCalculation:
    def __init__(self, clientID, joint_handles, configuration_type):
        self.clientID = clientID
        self.joint_handles = joint_handles[:-1]
        self.all_handles = joint_handles[:]  # contains also finger tip
        self.q = sp.symbols('q_0:{}'.format(len(self.joint_handles)))
        self.Ts = []
        self.jacobian = sp.zeros(4, 9)

        if configuration_type == ConfigurationType.finger:
            objects_positions = []
            for index, joint_handle in enumerate(self.all_handles):
                _, this_object_position = vrep.simxGetObjectPosition(self.clientID, joint_handle, -1,
                                                                     vrep.simx_opmode_blocking)
                objects_positions.append(np.array(this_object_position))
            for task in range(0, 3):
                transformations = []

                for index, joint_handle in enumerate(self.all_handles[:-task or None]):
                    _, this_joint_position = vrep.simxGetObjectPosition(self.clientID, joint_handle, -1,
                                                                        vrep.simx_opmode_blocking)
                    last_element = bool(index == len(self.all_handles) - task - 1)

                    vector_this_object_position = objects_positions[index]

                    if last_element:
                        transformations.append(sp.Matrix([[1, 0, 0, vector_this_object_position[0]],
                                                          [0, 1, 0, vector_this_object_position[1]],
                                                          [0, 0, 1, vector_this_object_position[2]],
                                                          [0, 0, 0, 1]]))
                        continue
                    angle = -self.q[index]  # -angle, because the joints turn clockwise
                    if index == 0:
                        transformations.append(sp.Matrix([[1, 0, 0, vector_this_object_position[0]],
                                                          [0, 1, 0, vector_this_object_position[1]],
                                                          [0, 0, 1, vector_this_object_position[2]],
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

                    length = np.linalg.norm(objects_positions[index] - objects_positions[index - 1])
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
                self.Ts.append(T)
                this_jacobian = T[0:3, 3].T.jacobian(sp.Matrix(self.q)).T
                self.jacobian[:, task * 3: task * 3 + 3] = this_jacobian
        else:
            print("Configuration not defined")

    def getJacobian(self):
        joint_positions = []
        for joint_handle in self.joint_handles:
            _, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_oneshot_wait)
            joint_positions.append(joint_position)
        return np.array(self.jacobian.subs(zip(self.q, joint_positions)), dtype='float')
        # return np.array(self.jacobian.subs(zip(self.q, joint_positions))).astype(dtype='float64')

    def printTransformation(self):
        joint_positions = []
        for joint_handle in self.joint_handles:
            _, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_oneshot_wait)
            joint_positions.append(joint_position)
        print(self.Ts[0].subs(zip(self.q, joint_positions))[0:3, 3])
