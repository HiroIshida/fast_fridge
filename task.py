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
from skrobot.planner import tinyfk_sqp_inverse_kinematics
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
    def __init__(self, robot_model, n_wp, full_demo=True):
        update_fksolver(robot_model)
        self.robot_model = robot_model
        self.joint_list = rarm_joint_list(robot_model)
        self.n_wp = n_wp
        self.fridge = Fridge(full_demo)
        self.sscc = TinyfkSweptSphereSdfCollisionChecker(
                self.fridge.sdf, robot_model)

        self.sscc_for_initial_trajectory = TinyfkSweptSphereSdfCollisionChecker(
                self.fridge.sdf, robot_model)

        for link in rarm_coll_link_list(robot_model):
            self.sscc.add_collision_link(link)
            self.sscc_for_initial_trajectory.add_collision_link(link)

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

class ApproachingTask(PoseDependentTask):
    def __init__(self, robot_model, n_wp):
        super(ApproachingTask, self).__init__(robot_model, n_wp, True)

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

class OpeningTask(PoseDependentTask):
    def __init__(self, robot_model, n_wp):
        super(OpeningTask, self).__init__(robot_model, n_wp, True)
        self.angle_open = 1.0

    def _create_init_trajectory(self):
        av_current = get_robot_config(self.robot_model, self.joint_list, with_base=True)

        constraint_start = self.cm.constraint_table[0]
        constraint_end = self.cm.constraint_table[self.n_wp-1]
        sdf_start = self.sscc.sdf[0]
        sdf_end = self.sscc.sdf[self.n_wp-1]
        sscc_here = self.sscc_for_initial_trajectory

        sscc_here.set_sdf(sdf_start)
        av_start = constraint_start.satisfying_angle_vector(av_init=av_current, collision_checker=sscc_here)

        sscc_here.set_sdf(sdf_end)
        av_end = constraint_end.satisfying_angle_vector(av_init=av_current, collision_checker=sscc_here)
        w = (av_end - av_start)/(self.n_wp - 1)
        av_seq_list = np.array([av_start + w * i for i in range(self.n_wp)])
        return av_seq_list

    def fridge_door_angle(self, idx):
        angle_seq = np.linspace(0, self.angle_open, self.n_wp-1)
        if idx == 0:
            return 0
        else:
            return angle_seq[idx-1]

    def setup(self):
        prepare_pose = self.fridge.prepare_gripper_pose()
        sdf_list = []
        for idx in range(self.n_wp):
            angle = self.fridge_door_angle(idx)
            if idx == 0:
                self.cm.add_pose_constraint(idx, "r_gripper_tool_frame", prepare_pose, force=True)
            else:
                pose = self.fridge.grasping_gripper_pose(angle)
                self.cm.add_pose_constraint(idx, "r_gripper_tool_frame", pose, force=True)
            sdf_list.append(self.fridge.gen_sdf(angle))
        self.sscc.set_sdf(sdf_list)

class Visualizer(object):
    def __init__(self):
        robot_model = pr2_init()
        joint_list = rarm_joint_list(robot_model)
        av_start = get_robot_config(robot_model, joint_list, with_base=True)
        fridge = Fridge(True)

        self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
        self.viewer.add(robot_model)
        self.viewer.add(fridge)
        self.fridge = fridge
        self.robot_model = robot_model
        self.joint_list = joint_list

        self.is_shown = False

    def show(self):
        self.viewer.show()
        self.is_shown = True

    def update(self, av, door_angle):
        set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
        self.fridge.set_angle(door_angle)
        self.viewer.redraw()

    def show_task(self, problem, idx=None):
        if not self.is_shown:
            self.show()

        self.fridge.newcoords(problem.fridge.copy_worldcoords())
        av_seq_cache = problem.av_seq_cache
        if idx is None:
            for idx in range(problem.n_wp):
                av = av_seq_cache[idx]
                door_angle = problem.fridge_door_angle(idx)
                self.update(av, door_angle)
                time.sleep(0.6)
        else:
            av = av_seq_cache[idx]
            door_angle = problem.fridge_door_angle(idx)
            self.update(av, door_angle)

if __name__=='__main__':
    np.random.seed(0)

    robot_model = pr2_init()
    joint_list = rarm_joint_list(robot_model)
    av_start = get_robot_config(robot_model, joint_list, with_base=True)
    av_end = copy.copy(av_start)
    av_end[-3] = 1.0

    fridge_pose = [[2.0, 1.5, 0.0], [0, 0, 0]]

    n_wp = 10
    task1 = ApproachingTask(robot_model, n_wp)
    task1.reset_firdge_pose(*fridge_pose)
    task1.setup(av_start, av_end)
    task1.solve()

    task2 = OpeningTask(robot_model, 5)
    task2.reset_firdge_pose(*fridge_pose)
    task2.setup()
    task2.solve()
    vis = Visualizer()
    vis.show_task(task2)
