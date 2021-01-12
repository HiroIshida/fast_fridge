#!/usr/bin/env python
import time
from pyinstrument import Profiler
import dill
from tqdm import tqdm


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
from sample_from_manifold import ManifoldSampler

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
np.random.seed(1)

class PoseDependentProblem(object):
    def __init__(self, robot_model, n_wp, k_start, k_end, angle_open=0.8):
        joint_list = rarm_joint_list(robot_model)
        cm = ConstraintManager(n_wp, joint_list, robot_model.fksolver, with_base=True)
        update_fksolver(robot_model)

        angles = np.linspace(0, angle_open, k_end - k_start + 1)

        fridge = Fridge(full_demo=True)
        sscc = TinyfkSweptSphereSdfCollisionChecker(fridge.sdf, robot_model)
        sscc_for_initial_trajectory = TinyfkSweptSphereSdfCollisionChecker(fridge.sdf, robot_model)
        for link in rarm_coll_link_list(robot_model):
            sscc.add_collision_link(link)
            sscc_for_initial_trajectory.add_collision_link(link)

        self.cm = cm
        self.sscc = sscc
        self.sscc_for_initial_trajectory = sscc_for_initial_trajectory

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

        # cmd ri
        self.duration = 0.8

    def send_cmd_to_ri(self, ri):
        self.robot_model.fksolver = None
        base_pose_seq = self.av_seq_cache[:, -3:]

        full_av_seq = []
        for av in self.av_seq_cache:
            set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
            full_av_seq.append(self.robot_model.angle_vector())

        time_seq = [self.duration]*self.n_wp
        ri.angle_vector_sequence(full_av_seq, time_seq)
        ri.move_trajectory_sequence(base_pose_seq, time_seq, send_action=True)

    def load_sol_cache(self, name="sol_cache.dill"):
        with open(name, "rb") as f:
            data = dill.load(f)
            self.av_seq_cache = data["av_seq_cache"]
            self.fridge_pose_cache = data["fridge_pose_cache"]

    def dump_sol_cache(self, name="sol_cache.dill"):
        assert (self.av_seq_cache is not None)
        data = {"av_seq_cache": self.av_seq_cache,
                "fridge_pose_cache": self.fridge_pose_cache}
        with open(name, "wb") as f:
            dill.dump(data, f)

    def setup(self):
        av_start = get_robot_config(self.robot_model, self.joint_list, with_base=True)
        self.cm.add_eq_configuration(0, av_start, force=True)
        sdf_list = self.fridge.gen_door_open_sdf_list(
                self.n_wp, self.k_start, self.k_end, self.angle_open)
        self.sscc.set_sdf(sdf_list)
        for idx, pose in self.fridge.gen_door_open_coords(self.k_start, self.k_end, self.angle_open):
            self.cm.add_pose_constraint(idx, "r_gripper_tool_frame", pose, force=True)

        # add left arm constraint 
        co_fridge_inside = self.fridge.copy_worldcoords()
        co_fridge_inside.translate([0.0, 0.0, 1.2])
        trans = co_fridge_inside.worldpos()
        #rpy = rpy_angle(co_fridge_inside.worldrot())[0]
        self.cm.add_pose_constraint(
            self.n_wp-1, "l_gripper_tool_frame", trans, force=True)

    @bench
    def solve(self, use_sol_cache=False, maxiter=100, only_ik=False):
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

            self.debug_av_seq_init_cache = av_seq_init
        else:
            av_current = get_robot_config(self.robot_model, self.joint_list, with_base=True)
            sdf_last = self.sscc.sdf[-1]
            self.sscc_for_initial_trajectory.set_sdf(sdf_last)
            av_seq_init = self.cm.gen_initial_trajectory(
                av_init=av_current, collision_checker=self.sscc_for_initial_trajectory)

        if only_ik:
            set_robot_config(self.robot_model, self.joint_list, av_seq_init[-1], with_base=True)
            return


        slsqp_option = {'ftol': self.ftol, 'disp': False, 'maxiter': maxiter}
        res = tinyfk_sqp_plan_trajectory(
            self.sscc, self.cm, av_seq_init, self.joint_list, self.n_wp,
            safety_margin=3e-2, with_base=True, slsqp_option=slsqp_option)

        SUCCESS = 0
        ITER_LIMIT = 9
        print("status: {0}".format(res.status))
        if res.status in [SUCCESS, ITER_LIMIT]:
            av_seq = res.x
            self.av_seq_cache = av_seq
            self.fridge_pose_cache = self.fridge.copy_worldcoords()
            return av_seq
        else:
            return None

    def debug_view(self):
        if self.viewer is None:
            self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
            self.viewer.add(self.robot_model)
            self.viewer.add(self.fridge)
            self.cv = ConstraintViewer(self.viewer, self.cm)
            self.cv.show()
            self.sscc.add_coll_spheres_to_viewer(self.viewer)
            self.viewer.show()

    def vis_sol(self, av_seq=None):
        if self.viewer is None:
            self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
            self.viewer.add(self.robot_model)
            self.viewer.add(self.fridge)
            self.cv = ConstraintViewer(self.viewer, self.cm)
            #self.sscc.add_coll_spheres_to_viewer(self.viewer)
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
            #self.sscc.update_color()
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

    def sample_from_constraint_manifold(self, k_wp, n_sample=3000, eps=0.2):
        # call this after setup problem
        # TODO this method should be inside constraint manager

        # get_joint_limit 
        fix_negative_inf = lambda x: -6.28 if x == -np.inf else x
        fix_positive_inf = lambda x: 6.28 if x == np.inf else x
        j_mins, j_maxs = zip(*[(fix_negative_inf(j.min_angle), fix_positive_inf(j.max_angle))
            for j in self.joint_list])

        sdf = self.sscc.sdf[k_wp]
        sscc_here = TinyfkSweptSphereSdfCollisionChecker(sdf, self.robot_model)

        for link in rarm_coll_link_list(self.robot_model):
            sscc_here.add_collision_link(link)

        eq_const = self.cm.constraint_table[k_wp] 
        const_func = eq_const.gen_subfunc()

        joint_ids = self.robot_model.fksolver.get_joint_ids([j.name for j in self.joint_list])
        def predicate(av):
            sds, _ = sscc_here._compute_batch_sd_vals(joint_ids, np.array([av]), with_base=True)
            return np.all(sds > 0)

        av_init = eq_const.satisfying_angle_vector(collision_checker=sscc_here)
        # tweak base
        j_mins_with_base = np.hstack([j_mins, av_init[-3:]-0.1]) + 1e-3
        j_maxs_with_base = np.hstack([j_maxs, av_init[-3:]+0.1]) + 1e-3

        assert predicate(av_init)
        assert np.all(av_init < j_maxs_with_base), "{0} neq {1}".format(av_init, j_maxs_with_base)
        assert np.all(av_init > j_mins_with_base), "{0} neq {1}".format(av_init, j_mins_with_base)
        ms = ManifoldSampler(av_init, const_func, j_mins_with_base, j_maxs_with_base,
                feasible_predicate=predicate, eps=eps)
        print("Sampling from constraint manifold...")
        for i in tqdm(range(n_sample)):
            ms.extend()
        return ms.get_whole_sample()

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

def generate_door_opening_trajectories():
    n_wp = 9
    k_start = 1
    k_end = 4
    robot_model = pr2_init()
    problem = PoseDependentProblem(robot_model, n_wp, k_start, k_end)

    problem.reset_firdge_pose([2.0, 1.5, 0.0])
    problem.setup()

    X_start = problem.sample_from_constraint_manifold(0, n_sample=10000, eps=0.1)
    X_end = problem.sample_from_constraint_manifold(n_wp-1, n_sample=10000, eps=0.1)
    print("start solving")
    while True:
        x_start = X_start[np.random.randint(X_start.shape[0])]
        x_end = X_end[np.random.randint(X_end.shape[0])]
        w = (x_end - x_start)/(n_wp - 1)
        av_seq_init = np.array([x_start + w * i for i in range(n_wp)])
        problem.av_seq_cache = av_seq_init
        problem.fridge_pose_cache = problem.fridge.copy_worldcoords()
        av_seq_sol = problem.solve(use_sol_cache=True)
        if av_seq_sol is not None:
            return problem, av_seq_sol
        print("cannot be solved. retry...")

if __name__=='__main__':
    problem, av_seq_sol = generate_door_opening_trajectories()
    #get_current_pose = setup_rosnode()
    n_wp = 16
    k_start = 8
    k_end = 11
    robot_model = pr2_init()
    problem = PoseDependentProblem(robot_model, n_wp, k_start, k_end)

    def solve(use_sol_cache=False, maxiter=100):
        co = Coordinates()
        robot_model.newcoords(co)
        trans, rpy = get_current_pose()
        problem.reset_firdge_pose_from_handle_pose(trans, rpy)
        problem.solve(use_sol_cache=use_sol_cache, maxiter=maxiter)

    def solve_in_simulater(use_sol_cache=False, only_ik=False):
        problem.reset_firdge_pose([2.0, 1.5, 0.0])
        problem.setup()
        av_seq = problem.solve(use_sol_cache=use_sol_cache, maxiter=100, only_ik=only_ik)

    problem.reset_firdge_pose([2.0, 1.5, 0.0])
    problem.setup()
    av_seq = problem.solve()

    S = problem.sample_from_constraint_manifold(n_wp-1, n_sample=10000, eps=0.1)
    set_robot_config(robot_model, problem.joint_list, S[-1], with_base=True)

    #solve_in_simulater(use_sol_cache=False, only_ik=False)
    #problem.debug_view()
    #solve_in_simulater(use_sol_cache=True)

    """
    robot_model2 = pr2_init()
    robot_model2.fksolver = None
    ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model2)
    ri.move_gripper("rarm", pos=0.06)

    trans, rpy = get_current_pose()
    problem.reset_firdge_pose_from_handle_pose(trans, rpy)
    solve(False)

    problem.send_cmd_to_ri(ri)
    time.sleep(problem.duration * (problem.k_start-1.3))
    ri.move_gripper("rarm", pos=0)
    #problem.vis_sol()
    """
