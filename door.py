import numpy as np

import skrobot
from skrobot.model import Axis
from skrobot.sdf import UnionSDF
from skrobot.coordinates import rpy_angle

def door_open_angle_seq(n_wp, k_start, k_end, angle_open):
    n_step = k_end - k_start + 1
    angles_whole = []
    angles_opening = np.linspace(0, angle_open, n_step)
    for _ in range(k_start):
        angles_whole.append(0.0)
    for angle in angles_opening:
        angles_whole.append(angle)
    while(len(angles_whole)!=n_wp):
        angles_whole.append(angle_open)
    return angles_whole

class Fridge(skrobot.model.RobotModel):
    def __init__(self, full_demo):
        super(Fridge, self).__init__()
        if full_demo:
            file_name = "./models/fridge.urdf"
        else:
            file_name = "./models/simple_fridge.urdf"
        self.load_urdf_file(file_name)
        self.sdf = UnionSDF.from_robot_model(self)
        axis_offset = -0.03

        axis = Axis(axis_length=0.2)
        self.handle_link.assoc(axis, relative_coords=axis)
        axis.translate([axis_offset, 0, 0])
        self.link_list.append(axis)
        self.axis = axis

    def set_angle(self, angle):
        self.door_joint.joint_angle(angle)

    def reset_angle(self):
        self.set_angle(0.0)

    def gen_door_open_sdf_list(self, n_wp, k_start, k_end, angle_open):
        angle_seq = door_open_angle_seq(n_wp, k_start, k_end, angle_open)
        return [self.gen_sdf(a) for a in angle_seq]

    def gen_door_open_coords(self, k_start, k_end, angle_open):
        pair_list = []

        k_prepare = k_start - 1
        prepare_pose = self.prepare_gripper_pose()
        pair_list.append((k_prepare, prepare_pose))

        angle_seq = np.linspace(0, angle_open, k_end - k_start + 1)
        idx_seq = range(k_start, k_end+1)
        for idx, a in zip(idx_seq, angle_seq):
            pose = self.grasping_gripper_pose(a)
            pair_list.append((idx, pose))
        return pair_list

    def gen_sdf(self, angle):
        def inner(X):
            self.set_angle(angle)
            sd_vals = self.sdf(X)
            self.reset_angle()
            return sd_vals
        return inner

    def prepare_gripper_pose(self):
        coords = self.axis.copy_worldcoords()
        coords.translate([-0.1, 0, 0])
        pos = coords.worldpos()
        rot = coords.worldrot()
        ypr = rpy_angle(rot)[0] # skrobot's rpy is ypr
        rpy = [ypr[2], ypr[1], ypr[0]]
        return np.hstack([pos, rpy])

    def grasping_gripper_pose(self, angle):
        self.set_angle(angle)
        coords = self.axis.copy_worldcoords()
        pos = coords.worldpos()
        rot = coords.worldrot()
        ypr = rpy_angle(rot)[0] # skrobot's rpy is ypr
        rpy = [ypr[2], ypr[1], ypr[0]]
        self.reset_angle()
        return np.hstack([pos, rpy])

if __name__=='__main__':
    import time 
    fridge = Fridge()
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
    viewer.add(fridge)
    viewer.show()

    for i in range(20):
        time.sleep(0.5)
        fridge.set_angle(i * 0.1)
        viewer.redraw()
