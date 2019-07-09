import pose_retargeting.vrep as vrep
from enum import Enum
import sympy as sp
import numpy as np
import time
import rospy
import pickle
import os
import rospkg


class JacobianCalculation:
    def __init__(self, *argv, **kwargs):
        pass

    def getJacobian(self):
        raise NotImplementedError

