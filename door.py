import numpy as np

import skrobot
from skrobot.model import Box
from skrobot.model import Axis
from skrobot.sdf import UnionSDF
from skrobot.coordinates import rpy_angle

from regexp import GridGraph

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
            file_name = "./models/complecated_fridge.urdf"
        else:
            file_name = "./models/simple_fridge.urdf"
        self.load_urdf_file(file_name)
        self.sdf = UnionSDF.from_robot_model(self)
        axis_offset = -0.0

        axis = Axis(axis_length=0.2)
        self.handle_link.assoc(axis, relative_coords=axis)
        axis.translate([axis_offset, 0, 0])
        self.link_list.append(axis)
        self.axis = axis

        tf_base2handle = self.handle_link.copy_worldcoords()
        tf_base2fridge = self.copy_worldcoords()
        self.tf_fridge2handle = tf_base2fridge.inverse_transformation().transform(tf_base2handle)

        self.inside_region_box = Box(extents=[0.5, 0.5, 0.5], face_colors=[255, 0, 0, 100], with_sdf=True)
        self.inside_region_box.translate([0.0, 0.0, 1.25])
        self.assoc(self.inside_region_box, relative_coords=self.inside_region_box)

    def is_inside(self, pts):
        sd_boxes = self.inside_region_box.sdf(pts)
        sd_fridge = self.sdf(pts)
        return np.logical_and(sd_boxes < 0.0, sd_fridge > 0.0)

    def typical_object_position(self):
        co_fridge_inside = self.copy_worldcoords()
        co_fridge_inside.translate([0.1, 0.0, 1.2])
        position = co_fridge_inside.worldpos()
        return position

    def sample_from_inside(self, N):
        extents = np.array(self.inside_region_box._extents)
        center = self.inside_region_box.worldpos()
        pts = np.random.rand(10 * N, 3) * extents[np.newaxis, :] + (center - extents * 0.5)[np.newaxis, :]

        logicals = self.is_inside(pts)
        pts_filtered = pts[logicals, :]
        return pts_filtered[1:N, :]

    def grid_sample_from_inside(self, N=20):
        extents = np.array(self.inside_region_box._extents)
        center = self.inside_region_box.worldpos()
        b_min = center - 0.5 * extents
        b_max = center + 0.5 * extents

        xlin, ylin, zlin = [np.linspace(b_min[i], b_max[i], N) for i in range(3)]
        Xmesh, Ymesh, Zmesh = np.meshgrid(xlin, ylin, zlin)
        pts = np.array([[x, y, z] for (x, y, z) in zip(Xmesh.flatten(), Ymesh.flatten(), Zmesh.flatten())])
        pts_filtered = pts[self.is_inside(pts), :]
        return pts_filtered

    def get_grid(self, N_grid=20):
        extents = np.array(self.inside_region_box._extents)
        center = self.inside_region_box.worldpos()
        b_min = center - 0.5 * extents
        b_max = center + 0.5 * extents
        predicate = lambda x: self.sdf(np.atleast_2d(x))[0] > 3e-2
        grid = GridGraph(b_min, b_max, N_grid, predicate)
        return grid

    def get_angle(self):
        return self.door_joint.joint_angle()

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

    def get_pose3d(self):
        x, y, z = self.worldpos()
        ypr = rpy_angle(self.worldrot())[0]
        pose3d = np.array([x, y, ypr[0]])
        return pose3d

if __name__=='__main__':
    import time 
    fridge = Fridge(True)
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
    viewer.add(fridge)
    viewer.add(fridge.inside_region_box)
    viewer.show()
    fridge.set_angle(1.2)
    grid = fridge.get_grid()

    pts = fridge.grid_sample_from_inside()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.show()
