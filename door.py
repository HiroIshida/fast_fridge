import numpy as np

import skrobot
from skrobot.model import Axis
from skrobot.sdf import UnionSDF
from skrobot.coordinates import rpy_angle

class Fridge(skrobot.model.RobotModel):
    def __init__(self):
        super(Fridge, self).__init__()
        file_name = "./models/fridge.urdf"
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
