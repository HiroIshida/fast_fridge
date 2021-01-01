import numpy as np

import skrobot
from skrobot.model import Axis
from skrobot.planner import PoseConstraint

class Fridge(object):
    def __init__(self):
        file_name = "./models/fridge.urdf"
        self.model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=file_name)
        self.offset = -0.03

        axis = Axis()
        self.model.handle_link.assoc(axis, relative_coords=axis)
        axis.translate([self.offset, 0, 0])
        self.model.link_list.append(axis)

    def set_angle(self, angle):
        self.model.door_joint.joint_angle(angle)

    def reset_pose(self):
        self.set_angle(0.0)

    def grasping_gripper_pose(self, angle):
        self.set_angle(angle)
        coords = self.model.handle_link.copy_worldcoords()
        return coords

    def gen_open_constraint(self, n_wp, k_start, k_end, open_angle):
        door_angle_seq = np.linspace(0, open_angle, k_end - k_start + 1)
        coords_seq = []
        for angle in door_angle_seq:
            coords = self.grasping_gripper_pose(angle)
            coords_seq.append(coords)
        self.reset_pose()
        return coords_seq

fridge = Fridge()
coords_seq = fridge.gen_open_constraint(20, 10, 15, 0.5)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
viewer.add(fridge.model)
viewer.show()
