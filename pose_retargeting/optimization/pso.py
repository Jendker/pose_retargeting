import numpy as np
from pose_retargeting.optimization.particle import Particle
from pose_retargeting.simulator.sim_mujoco import Mujoco
from mjrl.utils.gym_env import GymEnv
from multiprocessing import Pool
from pose_retargeting.rotations_mujoco import quat2euler


class Weights:
    def __init__(self, bodies_for_hand_pose_energy_position):
        self.weight_hand_pose_energy = 0.5
        self.pose_weights = np.ones(len(bodies_for_hand_pose_energy_position))
        self.sum_of_hand_pose_weights = np.sum(self.pose_weights)
        # weights task energy
        self.weight_task_energy = 0.5
        self.palm_weight = self.finger_tip_weight = 1
        self.sum_of_task_energy_weights = self.palm_weight + self.finger_tip_weight * 5


class PSO:
    def __init__(self, mujoco_env, parameters=None, no_cpu=7):
        self.num_cpu = no_cpu

        if parameters is None:
            self.parameters = {'c1': 2.8, 'c2': 1.3}
        else:
            self.parameters = parameters
        psi = self.parameters['c1'] + self.parameters['c2']
        self.parameters['omega'] = 2/abs(2-psi-np.sqrt(psi*psi-4 * psi))
        self.n_particles = 10
        self.iteration_count = 5
        self.dimension = mujoco_env.getNumberOfJoints()
        self.target_joints_pose = None
        self.hand_target_position = None
        self.hand_target_orientation = None
        self.particles = None
        self.best_global_fitness = float("inf")
        self.best_particle_position = None

        self.HPE_indices_for_hand_pose_energy_position = range(2, 21)
        self.bodies_for_hand_pose_energy_position = [mujoco_env.getHPEIndexEquivalentBody(x)
                                                     for x in self.HPE_indices_for_hand_pose_energy_position]
        self.HPE_indices_for_hand_pose_energy_angle = range(0, 21)
        self.bodies_for_hand_pose_energy_angles = [mujoco_env.getHPEIndexEquivalentBody(x)
                                                   for x in self.HPE_indices_for_hand_pose_energy_angle]
        # joint_names_for_hand_pose_energy_angles = self.mujoco_env.model.joint_names[
        #                                           self.mujoco_env.model.joint_name2id('rh_FFJ4'):
        #                                           self.mujoco_env.model.joint_name2id('rh_THJ1')+1]

        # weights hand pose energy
        self.weights = Weights(self.bodies_for_hand_pose_energy_position)

        # environments for parallel workers
        envs = [GymEnv(mujoco_env.env_name) for i in range(self.num_cpu)]
        for env in envs:
            env.reset()
        self.sims_mujoco_workers = [Mujoco(envs[i], mujoco_env.env_name) for i in range(self.num_cpu)]

        # create particles
        self.particle_batches = [[] for i in range(self.num_cpu)]
        self.createParticles(mujoco_env)

    def createParticles(self, mujoco_env):
        particles_count = 0
        split_index = 0
        while particles_count < self.n_particles:
            self.particle_batches[split_index].append(Particle(mujoco_env,
                                                               self.parameters, self.sims_mujoco_workers[split_index]))
            split_index += 1
            if split_index == self.num_cpu:
                split_index = 0
            particles_count += 1
        self.particle_batches = tuple(self.particle_batches)

    def initializeParticles(self, actions, simulator_state):
        for particles_batch in self.particle_batches:
            for particle in particles_batch:
                particle.initializePosition(actions, simulator_state)
        fitness_results = np.array(self._try_multiprocess(self.particle_batches, 1000, 4,
                                                          self.batchParticlesInitialization))
        lowest_batch_index = fitness_results.argmin()
        lowest_fitness = min(fitness_results[lowest_batch_index])
        self.best_global_fitness = lowest_fitness
        lowest_particle_index = fitness_results[lowest_batch_index].index(lowest_fitness)
        best_particle_position = self.particle_batches[lowest_batch_index][lowest_particle_index].position
        for particles_batch in self.particle_batches:
            for particle in particles_batch:
                particle.updateGlobalBest(best_particle_position)
                particle.initializeVelocity()

    def optimize(self, actions, mujoco_env):
        simulator_state = mujoco_env.env.gs()

        self.initializeParticles(actions, simulator_state)

        for i in range(0, self.iteration_count):
            fitness_results = np.array(self._try_multiprocess(self.particle_batches, 1000, 4,
                                                              self.batchParticlesGeneration))
            # update personal bests
            for j in range(self.num_cpu):
                for k in range(len(fitness_results[j])):
                    self.particle_batches[j][k].updatePersonalBest(fitness_results[j][k])

            lowest_batch_index = fitness_results.argmin()
            lowest_fitness = min(fitness_results[lowest_batch_index])
            if lowest_fitness < self.best_global_fitness:
                lowest_particle_index = fitness_results[lowest_batch_index].index(lowest_fitness)
                best_particle_position = self.particle_batches[lowest_batch_index][lowest_particle_index].position
                self.best_global_fitness = lowest_fitness
                self.best_particle_position = best_particle_position
                for particle_batch in self.particle_batches:
                    for particle in particle_batch:
                        particle.updateGlobalBest(best_particle_position)
        return self.best_particle_position

    def _try_multiprocess(self, args_list, max_process_time, max_timeouts, function):
        # Base case
        if max_timeouts == 0:
            return None

        # pool = Pool(processes=self.num_cpu, maxtasksperchild=1)
        # parallel_runs = [pool.apply_async(function,
        #                                   args=(args_list[i],)) for i in range(self.num_cpu)]
        # try:
        #     results = [p.get(timeout=max_process_time) for p in parallel_runs]
        # except Exception as e:
        #     print(str(e))
        #     print("Timeout Error raised... Trying again")
        #     pool.close()
        #     pool.terminate()
        #     pool.join()
        #     return self._try_multiprocess(args_list, max_process_time, max_timeouts - 1, function)
        # pool.close()
        # pool.terminate()
        # pool.join()
        results = []
        for i in range(self.num_cpu):
            results.append(function(args_list[i]))
        return results

    def _setHandTargetPositionAndQuaternion(self, target_position, target_quaternion):
        self.hand_target_position = target_position
        self.hand_target_orientation = quat2euler(target_quaternion)
    
    def new_taget_pose(self, new_target_fingers_pose, target_position, target_quaternion):
        self.target_joints_pose = new_target_fingers_pose
        self._setHandTargetPositionAndQuaternion(target_position, target_quaternion)

    def getHandPoseEnergyPosition(self, particle):
        energy = 0
        evaluated_hand_position = particle.sim_mujoco_worker.simulationObjectsPoseList(self.bodies_for_hand_pose_energy_position)
        for index, (target_pose, eval_hand_pose) in enumerate(
                zip(self.target_joints_pose, evaluated_hand_position)):
            energy += self.weights.pose_weights[index] * np.linalg.norm(target_pose - eval_hand_pose)**2
        # print("Position", energy / self.weights.sum_of_hand_pose_weights)
        return energy / self.weights.sum_of_hand_pose_weights

    def getHandPoseEnergyAngles(self, particle):
        fingers = [[0, 2, 9, 10, 11], [0, 3, 12, 13, 14], [0, 4, 15, 16, 17], [0, 5, 18, 19, 20], [0, 1, 6, 7, 8]]
        target_angles = []
        evaluated_joint_positions = []
        for finger_index, finger in enumerate(fingers):
            # TODO: handle thumb differently (proximal is different than [0, 0, 1]), later we have
            # again two joints in one point
            for index, this_point_index in enumerate(finger):
                if index - 1 < 0 or index + 1 >= len(finger):
                    continue
                previous_index = finger[index - 1]
                next_index = finger[index + 1]
                previous_vector_target = self.target_joints_pose[this_point_index] - self.target_joints_pose[
                    previous_index]
                next_vector_target = self.target_joints_pose[next_index] - self.target_joints_pose[this_point_index]
                target_angles.append(np.arccos(np.dot(next_vector_target, previous_vector_target)))
                evaluated_joint_positions.append(particle.sim_mujoco_worker.getJointPosition(self.bodies_for_hand_pose_energy_angles[
                    this_point_index])[1])
        target_angles = np.array(target_angles)
        evaluated_joint_positions = np.array(evaluated_joint_positions)
        mse = (((evaluated_joint_positions - target_angles) ** 2).mean(axis=None) / np.pi**2)
        # print("Angles", mse)
        return mse

    def getHandPoseEnergy(self, particle):
        return self.getHandPoseEnergyPosition(particle) + self.getHandPoseEnergyAngles(particle)

    @staticmethod
    def getTaskEnergy(particle):
        missing_weight = 2
        margin = 0.04
        constant = 0.004
        contact_dist = particle.getActiveContactsDist()
        # add palm
        palm_w = 3  # palm weight to make it more important than the rest of the tips (must be integer)
        if contact_dist is not None and 12 in contact_dist:  # palm id
            for i in range(palm_w - 1):
                contact_dist[12 + (i + 1) * 100] = contact_dist[12]  # add identical palm entries for the mean
        total = 5 + palm_w
        if contact_dist is not None:
            s = (total - len(contact_dist)) * missing_weight * (
                        (margin + constant) ** 2)  # punish for those that are not even in range
            for key in contact_dist:
                # ideally the distance is less than zero, so add constant to make it positive
                s += (max(max(contact_dist[key]) + constant,
                          0)) ** 2  # we want it to be less than 0 so there applied force
            # normalise
            # coeff = (len(contact_dist) + (5 - len(contact_dist)) * missing_weight) * ((margin + constant) ** 2)
            s /= (len(contact_dist) + (total - len(contact_dist)) * missing_weight) * ((margin + constant) ** 2)
            return s
        else:
            return 1

    def batchParticlesInitialization(self, particle_batch):
        energies = []
        for particle in particle_batch:
            particle.simulationStep()
            energies.append(self.fitness(particle))
        return energies

    def batchParticlesGeneration(self, particle_batch):
        energies = []
        for particle in particle_batch:
            particle.updatePositionAndVelocity()
            particle.simulationStep()
            energies.append(self.fitness(particle))
        return energies

    def fitness(self, particle):
        return self.weights.weight_hand_pose_energy * self.getHandPoseEnergy(particle) \
               + self.weights.weight_hand_pose_energy * self.getTaskEnergy(particle)
