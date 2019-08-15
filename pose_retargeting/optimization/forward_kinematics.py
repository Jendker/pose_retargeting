from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
import numpy as np


class ForwardKinematics:
    def __init__(self):
        builder = DiagramBuilder()
        self.plant, _ = AddMultibodyPlantSceneGraph(builder)
        self.instance = Parser(self.plant).AddModelFromFile("../resources/ShadowRobot/URDF/shadowrobot.urdf")
        self.plant.Finalize()
        self.context = self.plant.CreateDefaultContext()

    def updateForwardKinematics(self, joint_positions):
        self.plant.SetPositions(self.context, self.instance, joint_positions)

    def getLocalBodyPosition(self, body_name):
        body = self.plant.GetBodyByName(body_name)
        return np.array(self.plant.EvalBodyPoseInWorld(self.context, body))

    def getWorkdBodyPosition(self, transformation_matrix, body_name):
        return (transformation_matrix @ np.append(self.getLocalBodyPosition(body_name), 1))[0:3]
