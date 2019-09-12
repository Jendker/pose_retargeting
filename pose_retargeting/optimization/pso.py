import numpy as np
from pose_retargeting.optimization.particle import Particle
from pose_retargeting.simulator.sim_mujoco import Mujoco
from mjrl.utils.gym_env import GymEnv
from multiprocessing import Pool
from pose_retargeting.rotations_mujoco import quat2euler
from pose_retargeting.optimization.miscellaneous import Targets, ConstantData, Weights


glob_constant_data = None
glob_sim_mujoco_worker = None


class PSO:
    def __init__(self, mujoco_env, parameters=None, no_cpu=8):
        self.num_cpu = no_cpu

        if parameters is None:
            self.parameters = {'c1': 2.8, 'c2': 1.3}
        else:
            self.parameters = parameters
        psi = self.parameters['c1'] + self.parameters['c2']
        self.parameters['omega'] = 2/abs(2-psi-np.sqrt(psi*psi-4 * psi))

        self.n_particles = 16
        self.iteration_count = 5
        self.dimension = mujoco_env.getNumberOfJoints()

        self.particles = None
        self.best_global_fitness = float("inf")
        self.best_particle_position = None

        # joint_names_for_hand_pose_energy_angles = self.mujoco_env.model.joint_names[
        #                                           self.mujoco_env.model.joint_name2id('rh_FFJ4'):
        #                                           self.mujoco_env.model.joint_name2id('rh_THJ1')+1]

        self.constant_data = ConstantData(mujoco_env)

        # parallel workers with environments
        if self.num_cpu > 1:
            self.worker_pool = Pool(self.num_cpu, initializer=PSO.initialize_pool, initargs=[self.constant_data,
                                                                                         mujoco_env.env_name])

        self.weights = Weights(self.constant_data.bodies_for_hand_pose_energy_position, mujoco_env)
        self.targets = Targets()

        # create particles
        self.particle_batches = [[] for i in range(self.num_cpu)]
        self.createParticles(mujoco_env)
        self.obj_body_index = mujoco_env.env.model.body_name2id('Object')
        self.grasp_site_index = mujoco_env.env.model.site_name2id('S_grasp')

        self.mujoco_env = mujoco_env

    def __del__(self):
        try:
            self.worker_pool.close()
            self.worker_pool.terminate()
            self.worker_pool.join()
        except AttributeError:
            pass

    @staticmethod
    def getContactPairs(mujoco_env):
        geom1 = mujoco_env.env.model.pair_geom1
        geom2 = mujoco_env.env.model.pair_geom2
        # TODO: group the geoms into bodies
        pairs = {}
        if geom1 is not None and geom2 is not None:
            assert (len(geom1) == len(geom2))
            # group geom2 by geom1
            for elem in set(geom1):
                tmp = [geom2[i] for i in np.where(np.asarray(geom1) == elem)[0]]
                pairs[elem] = tmp
        return pairs

    def getDistanceBetweenObjectAndHand(self, mujoco_env):
        obj_pos = mujoco_env.env.data.body_xpos[self.obj_body_index].ravel()
        palm_pos = mujoco_env.env.data.site_xpos[self.grasp_site_index].ravel()
        return np.linalg.norm(obj_pos - palm_pos)

    @staticmethod
    def initialize_pool(constant_data, env_name):
        global glob_sim_mujoco_worker, glob_constant_data
        glob_constant_data = constant_data
        env = GymEnv(env_name)
        env.reset()
        glob_sim_mujoco_worker = Mujoco(env, env_name)

    def createParticles(self, mujoco_env):
        particles_count = 0
        split_index = 0

        contact_pairs = self.getContactPairs(mujoco_env)
        while particles_count < self.n_particles:
            self.particle_batches[split_index].append(Particle(mujoco_env, self.parameters, contact_pairs))
            split_index += 1
            if split_index == self.num_cpu:
                split_index = 0
            particles_count += 1
        self.particle_batches = tuple(self.particle_batches)

    @staticmethod
    def fillShorterLists(lists_input):
        max_len = max([len(lists_input) for lists_input in lists_input])
        for list_input in lists_input:
            list_input.extend([float("inf")] * (max_len - len(list_input)))


    def initializeParticles(self, actions, simulator_state):
        for particles_batch in self.particle_batches:
            for particle in particles_batch:
                particle.initializePosition(actions, simulator_state)

        inputs = [[particles_batch, self.weights, self.targets] for particles_batch in self.particle_batches]
        fitness_results = self._run_multiprocess(inputs, PSO.batchParticlesInitialization)
        PSO.fillShorterLists(fitness_results)
        fitness_results = np.array(fitness_results)
        lowest_batch_index, lowest_particle_index = np.unravel_index(np.argmin(fitness_results, axis=None),
                                                                     fitness_results.shape)
        lowest_fitness = fitness_results[lowest_batch_index][lowest_particle_index]
        self.best_particle_position = self.particle_batches[lowest_batch_index][lowest_particle_index].position
        self.best_global_fitness = lowest_fitness
        for particles_batch in self.particle_batches:
            for particle in particles_batch:
                particle.updateGlobalBest(self.best_particle_position)
                particle.initializeVelocity()

    def optimize(self, actions, mujoco_env):
        distance_between_object_and_hand = self.getDistanceBetweenObjectAndHand(mujoco_env)
        if distance_between_object_and_hand > 0.1:
            return actions

        self.weights.update_weights(distance_between_object_and_hand)
        simulator_state = mujoco_env.env.gs()

        self.initializeParticles(actions, simulator_state)

        for i in range(0, self.iteration_count):
            inputs = [[particles_batch, self.weights, self.targets] for particles_batch in self.particle_batches]
            fitness_results = self._run_multiprocess(inputs, PSO.batchParticlesInitialization)
            # update personal bests
            for j in range(self.num_cpu):
                for k in range(len(fitness_results[j])):
                    self.particle_batches[j][k].updatePersonalBest(fitness_results[j][k])

            PSO.fillShorterLists(fitness_results)
            fitness_results = np.array(fitness_results)
            lowest_batch_index, lowest_particle_index = np.unravel_index(np.argmin(fitness_results, axis=None),
                                                                         fitness_results.shape)
            lowest_fitness = fitness_results[lowest_batch_index][lowest_particle_index]
            if lowest_fitness < self.best_global_fitness:
                self.best_particle_position = self.particle_batches[lowest_batch_index][lowest_particle_index].position
                self.best_global_fitness = lowest_fitness
                for particle_batch in self.particle_batches:
                    for particle in particle_batch:
                        particle.updateGlobalBest(self.best_particle_position)
        return self.best_particle_position

    def _run_multiprocess(self, args_list, function):
        if self.num_cpu > 1:
            results = self.worker_pool.map(function, args_list)
        else:
            results = []
            if glob_sim_mujoco_worker is None:
                PSO.initialize_pool(self.constant_data, self.mujoco_env.env_name)
            for i in range(self.num_cpu):
                results.append(function(args_list[i]))
        return results

    def _setHandTargetPositionAndQuaternion(self, target_position, target_quaternion):
        self.targets.hand_target_position = target_position
        self.targets.hand_target_orientation = quat2euler(target_quaternion)
    
    def new_taget_pose(self, new_target_fingers_pose, target_position, target_quaternion):
        self.targets.target_joints_pose = new_target_fingers_pose
        self._setHandTargetPositionAndQuaternion(target_position, target_quaternion)

    @staticmethod
    def getHandPoseEnergyPosition(particle, weights, targets):
        global glob_constant_data
        energy = 0
        evaluated_hand_position = particle.sim_mujoco_worker.simulationObjectsPoseList(
            glob_constant_data.bodies_for_hand_pose_energy_position)
        for index, (target_pose, eval_hand_pose) in enumerate(
                zip(targets.target_joints_pose, evaluated_hand_position)):
            energy += weights.pose_weights[index] * np.linalg.norm(target_pose - eval_hand_pose)**2
        return energy / weights.sum_of_hand_pose_weights

    @staticmethod
    def getHandPoseEnergyAngles(particle, targets):
        global glob_constant_data
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
                previous_vector_target = targets.target_joints_pose[this_point_index] - targets.target_joints_pose[
                    previous_index]
                next_vector_target = targets.target_joints_pose[next_index] -\
                                     targets.target_joints_pose[this_point_index]
                target_angles.append(np.arccos(np.dot(next_vector_target, previous_vector_target)))
                evaluated_joint_positions.append(particle.sim_mujoco_worker.getJointPosition(glob_constant_data.bodies_for_hand_pose_energy_angles[
                    this_point_index])[1])
        target_angles = np.array(target_angles)
        evaluated_joint_positions = np.array(evaluated_joint_positions)
        mse = (((evaluated_joint_positions - target_angles) ** 2).mean(axis=None) / np.pi**2)
        # print("Angles", mse)
        return mse

    @staticmethod
    def getTaskEnergy(particle, weights):
        global glob_constant_data
        missing_weight = 2
        margin = 0.04
        constant = 0.004
        contact_dist = particle.getActiveContactsDist()
        # add palm
        assert(isinstance(weights.palm_weight, int))  # palm weight to make it more important than the rest of the tips (must be integer)
        real_contact_distances = []
        if contact_dist:
            # find smallest distance for palm (we have many, many geoms currently for palm)
            palm_distances = []
            for key in contact_dist:
                # check if key (index) belongs to palm
                if glob_constant_data.palm_max_index >= key >= glob_constant_data.palm_min_index:
                    # if so, append
                    palm_distances.append(contact_dist[key])
                else:
                    # it is finger, just add to to real_contact_dist
                    real_contact_distances.append(contact_dist[key])
            if palm_distances:
                # only add the smallest palm distance to real_contact_dist for energy calculation
                smallest_distance = min(palm_distances)
                for i in range(weights.palm_weight - 1):
                    real_contact_distances.append(smallest_distance)  # add identical palm entries for the mean
        total = weights.sum_of_task_energy_weights
        if real_contact_distances:
            s = (total - len(real_contact_distances)) * missing_weight * (
                        (margin + constant) ** 2)  # punish for those that are not even in range
            for distance in real_contact_distances:
                # ideally the distance is less than zero, so add constant to make it positive
                s += (max(max(distance) + constant,
                          0)) ** 2  # we want it to be less than 0 so there applied force
            # normalise
            # coeff = (len(contact_dist) + (5 - len(contact_dist)) * missing_weight) * ((margin + constant) ** 2)
            s /= (len(real_contact_distances) + (total - len(real_contact_distances)) * missing_weight) * ((margin + constant) ** 2)
            return s
        else:
            return 1

    @staticmethod
    def batchParticlesInitialization(inputs):
        particle_batch = inputs[0]
        weights = inputs[1]
        targets = inputs[2]
        global glob_sim_mujoco_worker

        energies = []
        for particle in particle_batch:
            particle.simulationStep(glob_sim_mujoco_worker)
            energies.append(PSO.fitness(particle, weights, targets))
        return energies

    @staticmethod
    def batchParticlesGeneration(inputs):
        particle_batch = inputs[0]
        weights = inputs[1]
        targets = inputs[2]
        global glob_sim_mujoco_worker

        energies = []
        for particle in particle_batch:
            particle.updatePositionAndVelocity()
            particle.simulationStep(glob_sim_mujoco_worker)
            energies.append(PSO.fitness(particle, weights, targets))
        return energies

    @staticmethod
    def fitness(particle, weights, targets):
        global glob_constant_data
        return weights.weight_hand_pose_energy_position * PSO.getHandPoseEnergyPosition(particle, weights, targets) + \
               weights.weight_hand_pose_energy_angle * PSO.getHandPoseEnergyAngles(particle, targets) + \
               + weights.weight_task_energy * PSO.getTaskEnergy(particle, weights)

