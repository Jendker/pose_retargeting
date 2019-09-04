import numpy as np

class Particle:
    def __init__(self, mujoco_env, position, parameters):
        self.best_position = position
        self.parameters = parameters
        self.personal_best = float("inf")
        self.sim = None

        global_orient_std = 5/180 * np.pi
        finger_std = 15/180 * np.pi
        global_trans_std = 0.02
        init_std_dev = [global_orient_std, global_orient_std, global_orient_std, global_trans_std, global_trans_std, global_trans_std]
        init_std_dev = np.pad(init_std_dev, (0, mujoco_env.getNumberOfJoints() - len(init_std_dev)), 'constant', constant_values=finger_std)
        self.position = position + np.random.normal(position, init_std_dev)
        self.velocity = np.zeros_like(self.position)
        # self.global_best_position = position
        self.global_best_position = np.zeros_like(self.best_position)

    def updatePositionAndVelocity(self):
        self.velocity = self.velocity + self.parameters['c1'] * (self.best_position - self.position) + \
                        self.parameters['c2'] * (self.global_best_position - self.position)
        self.position = self.position * self.velocity

    def updateGlobalBest(self, global_best_position):
        self.global_best_position = global_best_position

    def simulationStep(self, sim, initial_state):
        self.sim = sim
        self.sim.env.ss(initial_state)
        self.sim.env.step(self.position)
