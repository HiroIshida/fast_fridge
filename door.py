import numpy as np

import skrobot
from skrobot.model import Axis
from skrobot.sdf import UnionSDF
from skrobot.planner import PoseConstraint

class Fridge(object):
    def __init__(self):
        file_name = "./models/fridge.urdf"
        self.model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=file_name)
        self.sdf = UnionSDF.from_robot_model(self.model)
        self.offset = -0.03

        axis = Axis(axis_length=0.2)
        self.model.handle_link.assoc(axis, relative_coords=axis)
        axis.translate([self.offset, 0, 0])
        self.model.link_list.append(axis)
        self.axis = axis

    def set_angle(self, angle):
        self.model.door_joint.joint_angle(angle)

    def reset_angle(self):
        self.set_angle(0.0)

    def gen_sdf(self, angle):
        def inner(X):
            self.set_angle(angle)
            sd_vals = self.sdf(X)
            self.reset_angle()
            return sd_vals
        return inner

    def grasping_gripper_pose(self, angle):
        self.set_angle(angle)
        coords = self.axis.copy_worldcoords()
        self.reset_angle()
        return coords

if __name__=='__main__':
    import time 
    fridge = Fridge()
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
    viewer.add(fridge.model)
    viewer.show()

    for i in range(20):
        time.sleep(0.5)
        fridge.set_angle(i * 0.1)
        viewer.redraw()
