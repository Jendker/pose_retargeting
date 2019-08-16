from pose_retargeting.optimization.forward_kinematics import ForwardKinematics
import numpy as np
import ctypes
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

class PSO:
    def __init__(self):
        self.forward_kinematics = ForwardKinematics()
        # Set-up hyperparameters
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # Call instance of PSO
        self.optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

    @staticmethod
    def test_numpy():
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_get_max_threads = mkl_rt.mkl_get_max_threads

        def mkl_set_num_threads(cores):
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

        mkl_set_num_threads(4)
        print(mkl_get_max_threads())
        while True:
            np.arange(1000000).reshape((50,-1)).T @ np.arange(1000000).reshape((50,-1))

    def optimize(self, actions, frequency):
        best_cost, best_pos = self.optimizer.optimize(fx.sphere, iters=100, n_processes=8)
