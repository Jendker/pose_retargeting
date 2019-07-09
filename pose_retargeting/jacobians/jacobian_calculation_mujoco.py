import numpy as np
from pose_retargeting.jacobians.jacobian_calculation import JacobianCalculation


class JacobianCalculationMujoco(JacobianCalculation):
    def __init__(self, transformation_handles, task_objects_handles_and_bases, simulator, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.simulator = simulator
        joint_handles = transformation_handles[:-1]
        task_object_handles = [pair[0] for pair in list(task_objects_handles_and_bases)]
        self.jacobian_joint_indices, self.jacobian_task_target_body_names = self.calculateJacobianElements(joint_handles,
                                                                                                   task_object_handles)

    def calculateJacobianElements(self, joint_handles, task_object_handles):
        jacobian_joint_indices = []
        jacobian_task_target_body_names = []
        for joint_handle in joint_handles:
            jacobian_joint_indices.append(self.simulator.getJointHandleIndex(joint_handle))
        for target_handle in task_object_handles:
            jacobian_task_target_body_names.append(self.simulator.getJointBodyName(target_handle))
        return jacobian_joint_indices, jacobian_task_target_body_names

    def getJacobian(self):
        jacobian = np.empty(len(self.jacobian_joint_indices), 3 * len(self.jacobian_task_target_body_names))
        for task_index, jacobian_task_target_body_name in enumerate(self.jacobian_task_target_body_names):
            this_jacobian = self.simulator.getJacobianFromBodyName(jacobian_task_target_body_name)
            for joint_index, jacobian_joint_index in enumerate(self.jacobian_joint_indices):
                jacobian[joint_index, task_index*3:task_index*3+3] = this_jacobian[:, jacobian_joint_index].T
        return jacobian
