#!/usr/bin/env python
import time
from pyinstrument import Profiler
import pickle

import numpy as np
import rospy

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

from geometry_msgs.msg import Pose

def transform_av_seq(av_seq, _transform3d):
    # 2dim trans
    # 1dim rot
    transform3d = np.array(_transform3d)
    for av in av_seq:
        av[-3:-1] += transform3d[-3:-1]
        av[-1] += transform3d[-1]

def bench(func):
    def wrapper(*args, **kwargs):
        ts = time.time()
        ret = func(*args, **kwargs)
        print("elapsed time : {0}".format(time.time() - ts))
        return ret
    return wrapper

def detailbench(func):
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        ret = func(*args, **kwargs)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True, show_all=True))
        return ret
    return wrapper

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
        self.fridge_pose_cache = None # 3dim pose

        self.debug_av_seq_init_cache = None

    def send_cmd_to_ri(self, ri):
        self.robot_model.fksolver = None
        base_pose_seq = self.av_seq_cache[:, -3:]

        full_av_seq = []
        for av in self.av_seq_cache:
            set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
            full_av_seq.append(self.robot_model.angle_vector())

        time_seq = [1.0]*self.n_wp
        ri.angle_vector_sequence(full_av_seq, time_seq)
        #ri.move_trajectory_sequence(base_pose_seq, time_seq, send_action=True)

    @bench
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
            ts = time.time()
            tf_base2fridge_now = self.fridge.copy_worldcoords()
            tf_base2fridge_pre = self.fridge_pose_cache

            # NOTE somehow, in applying to vector, we must take inverse
            traj_planer = np.zeros((self.n_wp, 3))
            traj_planer[:, 0] = self.av_seq_cache[:, -3]
            traj_planer[:, 1] = self.av_seq_cache[:, -2] # world
            traj_planer_wrt_fridge = tf_base2fridge_pre.inverse_transformation().transform_vector(traj_planer)
            traj_planer_wrt_base_now = tf_base2fridge_now.transform_vector(traj_planer_wrt_fridge)

            yaw_now = rpy_angle(tf_base2fridge_now.worldrot())[0][0]
            yaw_pre = rpy_angle(tf_base2fridge_pre.worldrot())[0][0]

            av_seq_init = copy.copy(self.av_seq_cache)
            av_seq_init[:, -3:-1] = traj_planer_wrt_base_now[:, -3:-1]
            av_seq_init[:, -1] += (yaw_now - yaw_pre)

            print("debug==========")
            print(self.av_seq_cache)
            print(av_seq_init)

            self.debug_av_seq_init_cache = av_seq_init
        else:
            av_current = get_robot_config(self.robot_model, self.joint_list, with_base=True)
            av_seq_init = self.cm.gen_initial_trajectory(av_init=av_current)

        slsqp_option = {'ftol': self.ftol, 'disp': True, 'maxiter': 100}
        av_seq = tinyfk_sqp_plan_trajectory(
            self.sscc, self.cm, av_seq_init, self.joint_list, self.n_wp,
            safety_margin=3e-2, with_base=True, slsqp_option=slsqp_option)
        self.av_seq_cache = av_seq
        self.fridge_pose_cache = self.fridge.copy_worldcoords()
        return av_seq

    def debug_view(self):
        if self.viewer is None:
            self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
            self.viewer.add(self.robot_model)
            self.viewer.add(self.fridge)
            self.sscc.add_coll_spheres_to_viewer(self.viewer)
            self.viewer.show()

    def vis_sol(self, av_seq=None):
        if self.viewer is None:
            self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
            self.viewer.add(self.robot_model)
            self.viewer.add(self.fridge)
            self.cv = ConstraintViewer(self.viewer, self.cm)
            self.sscc.add_coll_spheres_to_viewer(self.viewer)
            self.cv.show()
            self.viewer.show()

        if av_seq is None:
            assert self.av_seq_cache is not None
            av_seq = self.av_seq_cache

        door_angle_seq = door_open_angle_seq(self.n_wp, self.k_start, self.k_end, self.angle_open)
        for idx in range(self.n_wp):
            av = av_seq[idx]
            set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
            self.fridge.set_angle(door_angle_seq[idx])
            self.sscc.update_color()
            self.viewer.redraw()
            time.sleep(0.6)

    def reset_firdge_pose_from_handle_pose(self, trans, rpy=None):
        if rpy is None:
            rotmat = None
        else:
            rotmat = rpy_matrix(rpy[2], rpy[1], rpy[0]) # actually ypr
        tf_base2handle = make_coords(pos=trans, rot=rotmat)
        tf_handle2fridge = self.fridge.tf_fridge2handle.inverse_transformation()
        tf_base2fridge = tf_base2handle.transform(tf_handle2fridge)
        self.fridge.newcoords(tf_base2fridge)
        self.fridge.reset_angle()

    def reset_firdge_pose(self, trans, rpy=None):
        if rpy is not None:
            ypr = [rpy[2], rpy[1], rpy[0]]
            rot = rpy_matrix(*ypr)
        else:
            rot = None
        co = Coordinates(pos = trans, rot=rot)
        self.fridge.newcoords(co)
        self.fridge.reset_angle()

def setup_rosnode():
    rospy.init_node('planner', anonymous=True)
    pose_current = {"pose": None}

    def cb_pose(msg):
        pos_msg = msg.position
        quat_msg = msg.orientation
        ypr = quaternion2rpy([quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z])[0]
        rpy = [ypr[2], ypr[1], ypr[0]]
        pos = [pos_msg.x, pos_msg.y, pos_msg.z]
        pose_current["pose"] = [pos, rpy]
    topic_name = "handle_pose"
    sub = rospy.Subscriber(topic_name, Pose, cb_pose)
    return (lambda : pose_current["pose"])

if __name__=='__main__':
    get_current_pose = setup_rosnode()
    n_wp = 12
    k_start = 8
    k_end = 11
    robot_model = pr2_init()
    problem = PoseDependentProblem(robot_model, n_wp, k_start, k_end)

    def solve(use_sol_cache=False):
        co = Coordinates()
        robot_model.newcoords(co)
        trans, rpy = get_current_pose()
        problem.reset_firdge_pose_from_handle_pose(trans, rpy)
        problem.solve(use_sol_cache=use_sol_cache)

    def solve_in_simulater(use_sol_cache=False):
        problem.reset_firdge_pose([2.2, 2.2, 0.0])
        av_seq = problem.solve(use_sol_cache=use_sol_cache)

    robot_model2 = pr2_init()
    robot_model2.fksolver = None
    ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model2)

    trans, rpy = get_current_pose()
    problem.reset_firdge_pose_from_handle_pose(trans, rpy)
    solve(False)

    problem.send_cmd_to_ri(ri)
    #problem.vis_sol()


    """
    av_seq_full = problem.dump_full_av_seq()

    import time
    for av_full in av_seq_full:
        ri.angle_vector(av_full, time=1.0, time_scale=1.0)
        time.sleep(0.5)
    """


    """
    problem.reset_firdge_pose([2.2, 2.2, 0.0])
    av_seq = problem.solve(use_sol_cache=False)
    problem.reset_firdge_pose([2.1, 2.1, 0.0], [0, 0, 0.1])
    av_seq = problem.solve(use_sol_cache=True)
    problem.vis_sol(problem.debug_av_seq_init_cache)
    problem.vis_sol(av_seq)
    """


