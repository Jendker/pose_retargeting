import os
import re
import pickle
import numpy as np


def extract_int(x):
    ints = re.findall(r'\d+', x)
    if ints:
        return ints[0]
    else:
        return ''


def biggest_int(x):
    return max([int(extract_int(s)) for s in x if extract_int(s)])


class PositionController:
    def __init__(self, path=None, policy=None, job_name=None):
        if job_name is None:
            job_name = 'DAPG_position_controller-v0_env'

        if policy is None:
            if path is None:
                script_path = os.path.realpath(__file__)
                path = os.path.dirname(script_path) + "/../../../mt_src/mt_src/training/Runs"
            path = path + "/" + job_name + "/run_1/iterations"
            if not os.path.exists(path):
                print("NN optimize. Path with policy:\n", path, "\nDoes not exist! Exiting.")
                exit(1)
            try:
                file_list = os.listdir(path)
            except NotADirectoryError:
                print("No policy files found for NN optimize in path:\n" + path + "\nExiting.")
                exit(1)
            file_list = sorted(file_list, key=extract_int)
            if file_list:
                max_iteration_number = biggest_int(file_list)
                if policy is None:
                    policy = pickle.load(
                        open(path + '/checkpoint_' + str(max_iteration_number) + '.pickle', 'rb'))[0]
                print("NN optimize starting from iteration no. " + str(max_iteration_number))
            else:
                print("No policy files found for NN optimize in path:\n" + path + "\nExiting.")
                exit(1)
        self.policy = policy
        self.obj_body_index = None
        self.grasp_site_index = None

    @staticmethod
    def get_observation(simulator, targets):
        data = simulator.env.data
        error = targets - data.qpos[:30]
        return np.concatenate([error, data.qpos[:30]])

    def get_control(self, clamped_targets, simulator):
        unclamped_targets = simulator.unclampActions(clamped_targets)
        observation = self.get_observation(simulator, unclamped_targets)
        controller_actions = self.policy.get_action(observation)[1]['evaluation']
        controller_actions[:6] = clamped_targets[:6]  # don't control the position and orientation
        return controller_actions
