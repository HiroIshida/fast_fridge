import os
from pyinstrument import Profiler
import time
import threading
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
from skrobot.planner import tinyfk_measure_nullspace
from skrobot.planner import TinyfkSweptSphereSdfCollisionChecker
from skrobot.planner.constraint_manager import InvalidPoseCstrException
from skrobot.planner.constraint_manager import InvalidConfigurationCstrException
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
from control_msgs.msg import FollowJointTrajectoryActionFeedback

class PreReplanFailException(Exception):
    # ad hoc; should be put with TrajectoryLibrary
    pass

def bench(func):
    def wrapper(*args, **kwargs):
        use_bench = ("bench_type" in kwargs)

        if use_bench:
            if kwargs["bench_type"]=="detail":
                profiler = Profiler()
                profiler.start()
            else:
                ts = time.time()

        ret = func(*args, **kwargs)

        if use_bench:
            if kwargs["bench_type"]=="detail":
                profiler.stop()
                print(profiler.output_text(unicode=True, color=True, show_all=True))
            else:
                print("elapsed time : {0}".format(time.time() - ts))
        return ret

    return wrapper

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

        self.ftol = 5e-3
        self.angle_open = 1.2
        self.is_setup = False

        # debug
        self.av_seq_init_cache = None
        self.av_seq_cache = None
        self.fridge_pose_cache = None

    @bench
    def solve(self, use_cache=True, callback=None, ignore_collision=False):
        assert self.is_setup, "plase setup the task before solving"
        self.is_setup = False

        if use_cache:
            av_seq_init = self.av_seq_cache
        else:
            av_seq_init = self._create_init_trajectory()
        self.av_seq_init_cache = av_seq_init

        self.cm.check_eqconst_validity(collision_checker=self.sscc)
        slsqp_option = {'ftol': self.ftol, 'disp': True, 'maxiter': 50}
        res = tinyfk_sqp_plan_trajectory(
            self.sscc, self.cm, av_seq_init, self.joint_list, self.n_wp,
            safety_margin=2e-2, with_base=True, slsqp_option=slsqp_option,
            callback=callback, ignore_collision=ignore_collision)

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

    def setup(self, use_cache=True, **kwargs):
        self.is_setup = True
        self._setup(**kwargs)
        if use_cache:
            self.create_av_init_from_cached_trajectory()

    def _setup(self, **kwargs):
        # task specific setup
        raise NotImplementedError

    def _create_init_trajectory(self):
        raise NotImplementedError

    def fridge_door_angle(self, idx):
        raise NotImplementedError

    def reset_fridge_pose_from_handle_pose(self, trans, rpy=None):
        if rpy is None:
            rotmat = None
        else:
            rotmat = rpy_matrix(rpy[2], rpy[1], rpy[0]) # actually ypr

        tf_base2handle = make_coords(pos=trans, rot=rotmat)
        tf_handle2fridge = self.fridge.tf_fridge2handle.inverse_transformation()
        tf_base2fridge = tf_base2handle.transform(tf_handle2fridge)

        # force reset z position to be 0.0
        pos = tf_base2fridge.worldpos()
        pos[2] = 0.0
        self.fridge.newcoords(tf_base2fridge)
        self.fridge.reset_angle()

    def reset_fridge_pose(self, trans, rpy=None):
        if rpy is not None:
            ypr = [rpy[2], rpy[1], rpy[0]]
            rot = rpy_matrix(*ypr)
        else:
            rot = None
        co = Coordinates(pos = trans, rot=rot)
        self.fridge.newcoords(co)
        self.fridge.reset_angle()

    def load_sol_cache(self, name="sol_cache.dill"):
        print("loading cache...")
        prefix = type(self).__name__
        name = prefix + "_" + name
        with open(name, "rb") as f:
            data = dill.load(f)
            self.av_seq_cache = data["av_seq_cache"]
            self.fridge_pose_cache = data["fridge_pose_cache"]

    def dump_sol_cache(self, name="sol_cache.dill"):
        prefix = type(self).__name__
        name = prefix + "_" + name
        assert (self.av_seq_cache is not None)
        data = {"av_seq_cache": self.av_seq_cache,
                "fridge_pose_cache": self.fridge_pose_cache}
        with open(name, "wb") as f:
            dill.dump(data, f)

    def create_av_init_from_cached_trajectory(self):
        if (self.fridge_pose_cache is None) or (self.av_seq_cache is None):
            raise Exception("cannot find cached trajectory.")
            print("not regeneraed from cached trajectory...")
            return 
        tf_base2fridge_now = self.fridge.copy_worldcoords()
        tf_base2fridge_pre = self.fridge_pose_cache

        # NOTE somehow, in applying to vector, we must take inverse
        traj_planer = np.zeros((self.n_wp, 2))
        traj_planer[:, 0] = self.av_seq_cache[:, -3]
        traj_planer[:, 1] = self.av_seq_cache[:, -2] # world

        # The transformatoin should be composite : COMPOSITE(x) = T_F'^I * T_I^F (x)
        # T_I^F (x) = R(-theta) * (x - o)
        # T_I^F'(x) = R(theta') * (x - o')
        # COMPOSITE(x) = T(theta' - theta) * (x - o) + o'

        o_pre = tf_base2fridge_pre.worldpos()[:2]
        o_now = tf_base2fridge_now.worldpos()[:2]
        yaw_now = rpy_angle(tf_base2fridge_now.worldrot())[0][0]
        yaw_pre = rpy_angle(tf_base2fridge_pre.worldrot())[0][0]
        R = lambda a: np.array([[cos(a), -sin(a)],[sin(a), cos(a)]])

        # transpose because X is array of raw vector
        def T_composite(X):
            Rmat = R(yaw_now - yaw_pre)
            return Rmat.dot((X - o_pre).transpose()).transpose() + o_now 

        av_seq_init = copy.copy(self.av_seq_cache)
        av_seq_init[:, -3:-1] = T_composite(traj_planer)
        av_seq_init[:, -1] += (yaw_now - yaw_pre)

        #av_seq_init[0, -3:] = np.zeros(3)

        self.av_seq_cache = av_seq_init
        self.fridge_pose_cache = tf_base2fridge_now
        return av_seq_init

    def sample_from_constraint_manifold(self, k_wp, n_sample=3000, eps=0.2):
        # call this after setup problem
        # TODO this method should be inside constraint manager

        # get_joint_limit 
        fix_negative_inf = lambda x: -6.28 if x == -np.inf else x
        fix_positive_inf = lambda x: 6.28 if x == np.inf else x
        j_mins, j_maxs = zip(*[(fix_negative_inf(j.min_angle), fix_positive_inf(j.max_angle))
            for j in self.joint_list])

        if isinstance(self.sscc.sdf, list):
            sdf = self.sscc.sdf[k_wp]
        else:
            sdf = self.sscc.sdf
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
        j_mins_with_base = np.hstack([j_mins, av_init[-3:]-0.1]) - 1e-3
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

    def check_trajectory(self, n_mid=2):
        return self.sscc.check_trajectory(self.joint_list, self.av_seq_cache, n_mid=n_mid, with_base=True)

    @property
    def default_send_duration(self):
        return [0.35] * (self.n_wp - 1)

class ApproachingTask(PoseDependentTask):
    def __init__(self, robot_model, n_wp):
        super(ApproachingTask, self).__init__(robot_model, n_wp, True)

    def _create_init_trajectory(self):
        av_current = get_robot_config(self.robot_model, self.joint_list, with_base=True)
        av_seq_init = self.cm.gen_initial_trajectory(av_init=av_current)
        return av_seq_init

    def fridge_door_angle(self, idx):
        return 0.0

    def _setup(self, **kwargs):
        assert "av_start" in kwargs
        assert "av_final" in kwargs
        av_start = kwargs["av_start"]
        av_final = kwargs["av_final"]

        self.cm.add_eq_configuration(0, av_start, force=True)
        self.cm.add_eq_configuration(self.n_wp-1, av_final, force=True)
        self.sscc.set_sdf(self.fridge.sdf)

class OpeningTask(PoseDependentTask):
    def __init__(self, robot_model, n_wp):
        super(OpeningTask, self).__init__(robot_model, n_wp, True)

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

    def _setup(self, **kwargs):
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

    @property
    def default_send_duration(self):
        duration_list = super(OpeningTask, self).default_send_duration
        duration_list[0] = 1.0
        return duration_list

class ReachingTask(PoseDependentTask):
    def __init__(self, robot_model, n_wp, n_wp_replan=None):
        super(ReachingTask, self).__init__(robot_model, n_wp, True)
        self.l_gripper_pose = None

        if n_wp_replan is None:
            n_wp_replan = int(n_wp / 3.0)
        self.n_wp_replan  = n_wp_replan
        n_wp_replan_dummy = n_wp_replan + 1
        self.cm_replan = ConstraintManager(
                n_wp_replan_dummy, self.joint_list,
                self.robot_model.fksolver,
                with_base=True)
        self.traj_lib = None

    def fridge_door_angle(self, idx):
        return self.angle_open

    def _create_init_trajectory(self):
        av_current = get_robot_config(self.robot_model, self.joint_list, with_base=True)

        def is_large_enough_ns(angle_vector):
            null_space_measure = tinyfk_measure_nullspace(angle_vector, self.joint_list,
                    "l_gripper_tool_frame", self.robot_model.fksolver, with_base=True)
            print("nullspace size {0}".format(null_space_measure))
            return (null_space_measure > 4.0)

        constraint_start = self.cm.constraint_table[0]
        constraint_end = self.cm.constraint_table[self.n_wp-1]

        av_start = constraint_start.satisfying_angle_vector(av_init=av_current, collision_checker=self.sscc)
        av_end = constraint_end.satisfying_angle_vector(av_init=av_current, collision_checker=self.sscc)
        w = (av_end - av_start)/(self.n_wp - 1)
        av_seq_list = np.array([av_start + w * i for i in range(self.n_wp)])
        return av_seq_list

    @bench
    def replanning(self, ignore_collision=False, callback=None, **kwargs):
        assert self.is_setup, "plase setup the task before solving"
        self.is_setup = False

        n_wp_replan_dummy = (self.n_wp_replan + 1)
        av_dummy = self.av_seq_cache[-n_wp_replan_dummy]
        av_start = self.av_seq_cache[-self.n_wp_replan]
        self.cm_replan.add_eq_configuration(0, av_dummy, force=True)
        self.cm_replan.add_eq_configuration(1, av_start, force=True)
        self.cm_replan.add_pose_constraint(n_wp_replan_dummy - 1, "l_gripper_tool_frame", self.l_gripper_pose, force=True)

        av_seq_init_partial = self.av_seq_cache[-n_wp_replan_dummy:, :]

        slsqp_option = {'ftol': self.ftol, 'disp': True, 'maxiter': 50}
        res = tinyfk_sqp_plan_trajectory(
            self.sscc, self.cm_replan, av_seq_init_partial, self.joint_list, n_wp_replan_dummy,
            safety_margin=2e-2, with_base=True, slsqp_option=slsqp_option,
            callback=callback, ignore_collision=ignore_collision)

        SUCCESS = 0
        print("status: {0}".format(res.status))
        if res.status in [SUCCESS]:
            av_seq_partial_solved = res.x
            self.av_seq_cache = np.vstack((self.av_seq_cache[:-n_wp_replan_dummy], av_seq_partial_solved))
            self.fridge_pose_cache = self.fridge.copy_worldcoords()
            print("trajectory optimization completed")
            return res
        else:
            return None

    def load_trajectory_library(self):
        filename = "traj_lib20.dill"
        """
        self.traj_lib = TrajectoryLibrary.load_dill(filename)
        with open(filename, 'rb') as f:
            self.traj_lib = dill.load(f)
        """
        with open(filename, 'rb') as f:
            self.traj_lib = dill.load(f)
        self.fridge_pose_cache = Coordinates()
        
    def attractor(self, av):
        co_fridge_inside = self.fridge.copy_worldcoords()
        rot = co_fridge_inside.worldrot()
        ypr = rpy_angle(rot)[0] # skrobot's rpy is ypr
        rpy = [ypr[2], ypr[1], ypr[0]]

        frame_name_list = ["l_gripper_tool_frame"]
        target_pose_list = [self.l_gripper_pose]
        av_new = tinyfk_sqp_inverse_kinematics(
                frame_name_list, 
                target_pose_list, 
                av, 
                self.joint_list,
                self.robot_model.fksolver,
                strategy="simple",
                maxiter=20,
                constraint_radius=1e-1,
                collision_checker=self.sscc,
                with_base=True)
        return av_new

    def _setup(self, **kwargs):
        sdf_open = self.fridge.gen_sdf(self.angle_open)
        self.sscc.set_sdf(sdf_open)

        assert "position" in kwargs
        position = kwargs["position"]

        r_gripper_pose = self.fridge.grasping_gripper_pose(self.angle_open)

        co_fridge_inside = self.fridge.copy_worldcoords()

        if position is None:
            position = self.fridge.typical_object_position()
            
        rot = co_fridge_inside.worldrot()
        ypr = rpy_angle(rot)[0] # skrobot's rpy is ypr
        rpy = [ypr[2], ypr[1], ypr[0]]

        if "av_start" in kwargs:
            self.cm.add_eq_configuration(0, kwargs["av_start"], force=True)
        else:
            print("[IMPORTANT] because av_start is not set, we solve using pose constraint")
            self.cm.add_pose_constraint(0, "r_gripper_tool_frame", r_gripper_pose, force=True)

        # final pose
        l_gripper_pose = np.hstack([position, rpy])
        self.cm.add_pose_constraint(self.n_wp-1, "l_gripper_tool_frame", l_gripper_pose, force=True)

        self.l_gripper_pose = l_gripper_pose

        if self.traj_lib is not None:
            print("load initial trajectory using precompted library")  
            print(position)
            x = position - self.fridge.worldpos()
            print(x)
            traj = self.traj_lib.find_trajectory(x)
            if traj is None:
                raise PreReplanFailException()
            print("proper trajectory is found")
            self.av_seq_cache = traj.av_seq


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

    def update(self, av, door_angle=None):
        if door_angle is None:
            door_angle = self.fridge.get_angle()
        set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
        self.fridge.set_angle(door_angle)
        self.viewer.redraw()

    def show_task(self, problem, idx=None):
        if not self.is_shown:
            self.show()

        cv = ConstraintViewer(self.viewer, problem.cm)
        cv.show()
        self.fridge.newcoords(problem.fridge.copy_worldcoords())
        av_seq_cache = problem.av_seq_cache
        if idx is None:
            for idx in range(problem.n_wp):
                av = av_seq_cache[idx]
                door_angle = problem.fridge_door_angle(idx)
                self.update(av, door_angle)
                time.sleep(0.3)
        else:
            av = av_seq_cache[idx]
            door_angle = problem.fridge_door_angle(idx)
            self.update(av, door_angle)
        cv.delete()

def generate_solution_cache():
    # np.random.seed(1010) # good
    # np.random.seed(1020) # not bad
    np.random.seed(100000)

    robot_model = pr2_init()
    joint_list = rarm_joint_list(robot_model)
    av_start = get_robot_config(robot_model, joint_list, with_base=True)

    fridge_pose = [[2.0, 1.5, 0.0], [0, 0, 0]]

    n_wp = 12
    task3 = ReachingTask(robot_model, n_wp)
    task3.reset_fridge_pose(*fridge_pose)
    task3.setup(use_cache=False, position=None)

    n_mid = 20
    N = 2000
    """
    X3_start = task3.sample_from_constraint_manifold(k_wp=0, n_sample=N, eps=0.1)
    X3_end = task3.sample_from_constraint_manifold(k_wp=task3.n_wp-1, n_sample=N, eps=0.1)
    """
    while True:
        """
        x3_start = X3_start[np.random.randint(X3_start.shape[0])]
        x3_end = X3_end[np.random.randint(X3_end.shape[0])]
        w = (x3_end - x3_start)/(n_wp - 1)
        av_seq_init = np.array([x3_start + w * i for i in range(n_wp)])
        task3.av_seq_cache = av_seq_init
        task3.fridge_pose_cache = task3.fridge.copy_worldcoords()
        """

        task3.setup(use_cache=False, position=None)
        av_seq_sol3 = task3.solve(use_cache=False)
        if av_seq_sol3 is not None:
            if task3.check_trajectory(n_mid=n_mid):
                print("task3 solved")
                task2 = OpeningTask(robot_model, 10)
                task2.reset_fridge_pose(*fridge_pose)
                task2.setup(use_cache=False)
                task2.cm.add_eq_configuration(task2.n_wp-1, av_seq_sol3[0], force=True)
                av_seq_sol2 = task2.solve(use_cache=False)

                if av_seq_sol2 is not None:
                    print("task2 solved")
                    task1 = ApproachingTask(robot_model, 10)
                    task1.reset_fridge_pose(*fridge_pose)
                    task1.setup(av_start=av_start, av_final=av_seq_sol2[0], use_cache=False)
                    av_seq_sol1 = task1.solve(use_cache=False)
                    if av_seq_sol1 is not None:
                        if task1.check_trajectory(n_mid=n_mid):
                            for task in [task1, task2, task3]:
                                task.dump_sol_cache()
                            return task1, task2, task3
        print("retry..")

if __name__=='__main__':
    do_prepare = True
    if do_prepare:
        task1, task2, task3 = generate_solution_cache()
        vis = Visualizer()
        for task in [task1, task2, task3]:
            vis.show_task(task)
    else:
        #np.random.seed(3)

        robot_model = pr2_init()
        joint_list = rarm_joint_list(robot_model)
        av_start = get_robot_config(robot_model, joint_list, with_base=True)

        trans = [-1.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0]

        task3 = ReachingTask(robot_model, 12)
        task3.load_sol_cache()
        task3.reset_fridge_pose_from_handle_pose(trans, rpy)
        #task3.reset_fridge_pose(*fridge_pose)
        pos = task3.fridge.typical_object_position()
        #task3.load_trajectory_library()
        task3.setup(position=pos + np.array([0.09, 0, 0.12]))

        opt_res = task3.replanning(ignore_collision=False, bench_type="normal")
        assert task3.check_trajectory()

        task2 = OpeningTask(robot_model, 10)
        task2.load_sol_cache()
        task2.reset_fridge_pose_from_handle_pose(trans, rpy)
        task2.setup()

        vis = Visualizer()
        vis.show_task(task2)
        vis.show_task(task3)
