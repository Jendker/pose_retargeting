from pose_retargeting.simulator.sim_mujoco import euclideanTransformation
import numpy as np


class Particle:
    def __init__(self, mujoco_env, forward_kinematics, position, velocity):
        self.forward_kinematics = forward_kinematics
        self.position = position  # vector of joint positions with dimension of the joint count
        self.velocity = velocity
        self.best_position = position
        self.bestLoss = float("inf")

    def evaluatePose(self, bodies):
        self.forward_kinematics.updateForwardKinematics(self.position)
        pose = []
        for body in bodies:
            pose.append(self.forward_kinematics.getWorldBodyPosition(body))
        return pose

class PSO:
    def __init__(self, mujoco_env):
        self.mujoco_env = mujoco_env
        self.parameters = {'c1': 2.8, 'c2': 1.3}
        psi = self.parameters['c1'] + self.parameters['c2']
        self.parameters['w'] = 2/abs(2-psi-np.sqrt(psi*psi-4 * psi))
        self.n_particles = 20
        self.iterations = 50
        self.dimension = mujoco_env.getNumberOfJoints()
        self.target_joints_pose = None
        self.hand_target_position = None
        self.hand_target_orientation = None

        bodies_for_hand_pose_energy = ['IMCP_front_joint', 'MMCP_front_joint', 'RMCP_front_joint', 'PMCP_front_joint',
            'TPIP_front_joint', 'TDIP_joint', 'TTIP_tip', 'IPIP_joint', 'IDIP_joint', 'ITIP_tip',
            'MPIP_joint', 'MDIP_joint', 'MTIP_tip', 'RPIP_joint', 'RDIP_joint', 'RTIP_tip',
            'PPIP_joint', 'PDIP_joint', 'PTIP_tip']
        self.bodies_for_hand_pose_energy = [self.mujoco_env.getHandle(x) for x in bodies_for_hand_pose_energy]
        # self.joint_indices_for_pose_energy = [mujoco_env.getJointNameIndex(mujoco_env.getBodyJointName(x))
        #                                       for x in self.bodies_for_hand_pose_energy]

        # weights hand pose energy
        self.weight_hand_pose_energy = 0.5
        self.pose_weights = np.ones(len(self.bodies_for_hand_pose_energy)).tolist()
        self.sum_of_hand_pose_weights = np.sum(self.pose_weights)
        # weights task energy
        self.weight_task_energy = 0.5
        self.palm_weight = self.finger_tip_weight = 1
        self.sum_of_task_energy_weights = self.palm_weight + self.finger_tip_weight * 5

    def getHandPoseEnergyPosition(self, particle):
        energy = 0
        current_hand_position = self.mujoco_env.simulationObjectsPoseList(self.bodies_for_hand_pose_energy)
        particle_evaluated_hand_position = particle.evaluatePose(self.bodies_for_hand_pose_energy)
        for index, (evaluated_pose, hand_pose) in enumerate(zip(particle_evaluated_hand_position, current_hand_position)):
            energy += self.pose_weights * np.linalg.norm(evaluated_pose - hand_pose)
        return energy / self.sum_of_hand_pose_weights

    # def getHandPoseEnergyAngles(self, particle):
    #     current_joint_positions = np.array([self.mujoco_env.getJointIndexPosition(x) for x in
    #                                         self.joint_indices_for_pose_energy])
    #     particle_joint_positions = particle.position[self.joint_indices_for_pose_energy]
    #     return np.linalg.norm((current_joint_positions - particle_joint_positions)/np.pi)/len(current_joint_positions)

    def getHandPoseEnergy(self, particle):
        return self.getHandPoseEnergyPosition(particle)  # + self.getHandPoseEnergyAngles(particle)

    def getTaskEnergy(self, particle):
        pass

    def fitness(self, particle):
        return self.weight_hand_pose_energy * self.getHandPoseEnergy(particle) \
               + self.weight_hand_pose_energy * self.getTaskEnergy(particle)

    def optimize(self, actions, frequency):
        pass

    def _setHandTargetPositionAndQuaternion(self, target_position, target_quaternion):
        self.hand_target_position = target_position
        self.hand_target_orientation = self.mujoco_env.quat2euler(target_quaternion)
    
    def new_taget_pose(self, new_target_fingers_pose, target_position, target_quaternion):
        self.target_joints_pose = new_target_fingers_pose
        shift_from_falling = np.array([0, 0, -0.075])
        self._setHandTargetPositionAndQuaternion(target_position + shift_from_falling, target_quaternion)
