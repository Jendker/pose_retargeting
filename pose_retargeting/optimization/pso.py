from pose_retargeting.optimization.forward_kinematics import ForwardKinematics
import numpy as np
import ctypes

class PSO:
    def __init__(self):
        self.forward_kinematics = ForwardKinematics()

    @staticmethod
    def test_numpy():
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_get_max_threads = mkl_rt.mkl_get_max_threads

        def mkl_set_num_threads(cores):
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

        mkl_set_num_threads(4)
        print(mkl_get_max_threads())
        while(True):
            np.arange(1000000).reshape((50,-1)).T @ np.arange(1000000).reshape((50,-1))
