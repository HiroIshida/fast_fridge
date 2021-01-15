from pyinstrument import Profiler
import time
import dill
from tqdm import tqdm

import numpy as np
import rospy
from math import *

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import quaternion2rpy
from skrobot.coordinates import make_coords, rpy_matrix
from skrobot.coordinates import rpy_angle
from skrobot.planner import tinyfk_sqp_plan_trajectory
from skrobot.planner import TinyfkSweptSphereSdfCollisionChecker
from skrobot.planner import ConstraintManager
from skrobot.planner import ConstraintViewer
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from skrobot.planner.utils import update_fksolver
from pr2opt_common import *
from door import Fridge, door_open_angle_seq
import copy


class PoseDependentTask(object):
    def __init__(self, robot_model, fridge, n_wp, full_demo=True):
        self.robot_model = robot_model
        self.joint_list = rarm_joint_list(robot_model)
        self.n_wp = n_wp
        self.fridge = fridge
        self.sscc = TinyfkSweptSphereSdfCollisionChecker(
                self.fridge.sdf, robot_model)

        for link in rarm_coll_link_list(robot_model):
            self.sscc.add_collision_link(link)

        self.cm = ConstraintManager(
                self.n_wp, self.joint_list,
                self.robot_model.fksolver,
                with_base=True)

        self.ftol = 1e-4

    def solve(self):
        av_seq_init = self._create_init_trajectory()

        slsqp_option = {'ftol': self.ftol, 'disp': True, 'maxiter': 100}
        res = tinyfk_sqp_plan_trajectory(
            self.sscc, self.cm, av_seq_init, self.joint_list, self.n_wp,
            safety_margin=3e-2, with_base=True, slsqp_option=slsqp_option)

        SUCCESS = 0
        print("status: {0}".format(res.status))
        if res.status in [SUCCESS]:
            av_seq = res.x
            self.av_seq_cache = av_seq
            self.fridge_pose_cache = self.fridge.copy_worldcoords()
            print("trajectory optimization completed")
            return av_seq
        else:
            return None

    def setup(self):
        raise NotImplementedError

    def _create_init_trajectory(self):
        raise NotImplementedError

    def fridge_door_angle(self, idx):
        raise NotImplementedError

    def reset_firdge_pose(self, trans, rpy=None):
        if rpy is not None:
            ypr = [rpy[2], rpy[1], rpy[0]]
            rot = rpy_matrix(*ypr)
        else:
            rot = None
        co = Coordinates(pos = trans, rot=rot)
        self.fridge.newcoords(co)
        self.fridge.reset_angle()

    def visualize_solution(self, viewer):
        for av in self.av_seq_cache:
            set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
            viewer.redraw()
            time.sleep(0.6)

class ApproachingTask(PoseDependentTask):
    def __init__(self, robot_model, fridge, n_wp):
        super(ApproachingTask, self).__init__(robot_model, fridge, n_wp, True)

    def _create_init_trajectory(self):
        av_current = get_robot_config(self.robot_model, self.joint_list, with_base=True)
        av_seq_init = self.cm.gen_initial_trajectory(av_init=av_current)
        return av_seq_init

    def fridge_door_angle(self, idx):
        return 0.0

    def setup(self, av_start, av_final):
        self.cm.add_eq_configuration(0, av_start, force=True)
        self.cm.add_eq_configuration(self.n_wp-1, av_final, force=True)
        self.sscc.set_sdf(self.fridge.sdf)

def initialize():
    robot_model = pr2_init()
    joint_list = rarm_joint_list(robot_model)
    av_start = get_robot_config(robot_model, joint_list, with_base=True)
    fridge = Fridge(True)

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
    viewer.add(robot_model)
    viewer.add(fridge)
    return robot_model, fridge, av_start, viewer

if __name__=='__main__':
    robot_model, fridge, av_start, viewer = initialize()
    av_end = copy.copy(av_start)
    av_end[-3] = 1.0

    n_wp = 10
    task1 = ApproachingTask(robot_model, fridge, n_wp)
    task1.reset_firdge_pose([2.0, 1.5, 0.0])
    task1.setup(av_start, av_end)
    task1.solve()
    viewer.show()
    task1.visualize_solution(viewer)
