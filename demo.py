#!/usr/bin/env python
import time

import numpy as np

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.coordinates import Coordinates
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

# initialization stuff
np.random.seed(0)

class PoseDependentProblem(object):
    def __init__(self, robot_model, n_wp, k_start, k_end, angle_open=0.8):
        joint_list = rarm_joint_list(robot_model)
        cm = ConstraintManager(n_wp, joint_list, robot_model.fksolver, with_base=True)
        update_fksolver(robot_model)

        angles = np.linspace(0, angle_open, k_end - k_start + 1)

        fridge = Fridge(full_demo=False)
        sscc = TinyfkSweptSphereSdfCollisionChecker(fridge.sdf, robot_model)
        for link in rarm_coll_link_list(robot_model):
            sscc.add_collision_link(link)

        self.cm = cm
        self.sscc = sscc
        self.fridge = fridge
        self.robot_model = robot_model
        self.joint_list = joint_list

        # problem parameters
        self.n_wp = n_wp
        self.k_start = k_start
        self.k_end = k_end
        self.angle_open = angle_open
        self.ftol = 1e-3

        # visualization stuff
        self.viewer = None
        self.constraint_viewer = None

        # cache for MPC
        self.av_seq_cache = None


    def solve(self, fridge_pose=None, use_sol_cache=False):
        if fridge_pose is not None:
            trans, rpy = fridge_pose
            ypr = [rpy[2], rpy[1], rpy[0]]
            co = Coordinates(pos=trans, rotation=ypr)
            self.fridge.newcoords(co)

        av_start = get_robot_config(robot_model, self.joint_list, with_base=True)
        self.cm.add_eq_configuration(0, av_start, force=True)
        sdf_list = self.fridge.gen_door_open_sdf_list(
                self.n_wp, self.k_start, self.k_end, self.angle_open)
        for idx, pose in self.fridge.gen_door_open_coords(k_start, k_end, self.angle_open):
            self.cm.add_pose_constraint(idx, "r_gripper_tool_frame", pose, force=True)

        if use_sol_cache:
            assert (self.av_seq_cache is not None)
            # TODO make better initial solution using robot's current configuration
            av_seq_init = copy.copy(self.av_seq_cache)
        else:
            av_current = get_robot_config(self.robot_model, self.joint_list, with_base=True)
            av_seq_init = self.cm.gen_initial_trajectory(av_init=av_current)

        slsqp_option = {'ftol': self.ftol, 'disp': True, 'maxiter': 100}
        av_seq = tinyfk_sqp_plan_trajectory(
            self.sscc, self.cm, av_seq_init, self.joint_list, self.n_wp,
            safety_margin=3e-2, with_base=True, slsqp_option=slsqp_option)
        self.av_seq_cache = av_seq
        return av_seq

    def vis_sol(self):
        if self.viewer is None:
            self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
            self.viewer.add(self.robot_model)
            self.viewer.add(self.fridge)
            self.cv = ConstraintViewer(self.viewer, self.cm)
            self.cv.show()
            self.viewer.show()

        door_angle_seq = door_open_angle_seq(self.n_wp, self.k_start, self.k_end, self.angle_open)
        for idx in range(self.n_wp):
            av = self.av_seq_cache[idx]
            set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
            self.fridge.set_angle(door_angle_seq[idx])
            self.viewer.redraw()
            time.sleep(0.6)


    def reset_firdge_pose(self, trans, rpy=None):
        if rpy is not None:
            ypr = [rpy[2], rpy[1], rpy[0]]
            rot = rpy_matrix(*ypr)
        else:
            rot = None
        co = Coordinates(pos = trans, rot=rot)
        self.fridge.newcoords(co)

n_wp = 12
k_start = 8
k_end = 11
robot_model = pr2_init()
problem = PoseDependentProblem(robot_model, n_wp, k_start, k_end)
problem.reset_firdge_pose([2.2, 2.0, 0.0])
av_seq = problem.solve()

from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
problem.reset_firdge_pose([2.2, 2.2, 0.0])
av_seq = problem.solve(use_sol_cache=True)
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))

problem.vis_sol()
