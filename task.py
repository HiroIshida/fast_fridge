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

from sample_from_manifold import ManifoldSampler

from geometry_msgs.msg import Pose

def bench(func):
    def wrapper(*args, **kwargs):
        ts = time.time()
        ret = func(*args, **kwargs)
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

        self.ftol = 1e-4

        # debug
        self.av_seq_init_cache = None
        self.av_seq_cache = None
        self.fridge_pose_cache = None

    @bench
    def solve(self, use_cache=False):
        if use_cache:
            av_seq_init = self.create_av_init_from_cached_trajectory()
        else:
            av_seq_init = self._create_init_trajectory()
        self.av_seq_init_cache = av_seq_init

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
        self.create_av_init_from_cached_trajectory()

    def reset_firdge_pose(self, trans, rpy=None):
        if rpy is not None:
            ypr = [rpy[2], rpy[1], rpy[0]]
            rot = rpy_matrix(*ypr)
        else:
            rot = None
        co = Coordinates(pos = trans, rot=rot)
        self.fridge.newcoords(co)
        self.fridge.reset_angle()
        self.create_av_init_from_cached_trajectory()

    def load_sol_cache(self, name="sol_cache.dill"):
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
        if self.fridge_pose_cache is None:
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

class ReachingTask(PoseDependentTask):
    def __init__(self, robot_model, n_wp):
        super(ReachingTask, self).__init__(robot_model, n_wp, True)
        self.angle_open = 1.0

    def fridge_door_angle(self, idx):
        return self.angle_open

    def _create_init_trajectory(self):
        av_current = get_robot_config(self.robot_model, self.joint_list, with_base=True)

        constraint_start = self.cm.constraint_table[0]
        constraint_end = self.cm.constraint_table[self.n_wp-1]
        av_start = constraint_start.satisfying_angle_vector(av_init=av_current, collision_checker=self.sscc)
        av_end = constraint_end.satisfying_angle_vector(av_init=av_current, collision_checker=self.sscc)
        w = (av_end - av_start)/(self.n_wp - 1)
        av_seq_list = np.array([av_start + w * i for i in range(self.n_wp)])
        return av_seq_list

    def setup(self, position=None):
        r_gripper_pose = self.fridge.grasping_gripper_pose(self.angle_open)

        co_fridge_inside = self.fridge.copy_worldcoords()

        if position is None:
            co_fridge_inside.translate([0.1, 0.0, 1.2])
            position = co_fridge_inside.worldpos()
        rot = co_fridge_inside.worldrot()
        ypr = rpy_angle(rot)[0] # skrobot's rpy is ypr
        rpy = [ypr[2], ypr[1], ypr[0]]

        self.cm.add_pose_constraint(0, "r_gripper_tool_frame", r_gripper_pose, force=True)

        # final pose
        l_gripper_pose = np.hstack([position, rpy])
        self.cm.add_pose_constraint(self.n_wp-1, "l_gripper_tool_frame", l_gripper_pose, force=True)
        sdf_open = self.fridge.gen_sdf(self.angle_open)
        self.sscc.set_sdf(sdf_open)

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

def generate_solution_cache():
    np.random.seed(130)

    robot_model = pr2_init()
    joint_list = rarm_joint_list(robot_model)
    av_start = get_robot_config(robot_model, joint_list, with_base=True)

    fridge_pose = [[2.0, 1.5, 0.0], [0, 0, 0]]

    n_wp = 10
    task3 = ReachingTask(robot_model, n_wp)
    task3.reset_firdge_pose(*fridge_pose)
    task3.setup()

    N = 10000
    X3_start = task3.sample_from_constraint_manifold(k_wp=0, n_sample=N, eps=0.1)
    X3_end = task3.sample_from_constraint_manifold(k_wp=task3.n_wp-1, n_sample=N, eps=0.1)
    while True:
        x3_start = X3_start[np.random.randint(X3_start.shape[0])]
        x3_end = X3_end[np.random.randint(X3_end.shape[0])]
        w = (x3_end - x3_start)/(n_wp - 1)
        av_seq_init = np.array([x3_start + w * i for i in range(n_wp)])
        task3.av_seq_cache = av_seq_init
        task3.fridge_pose_cache = task3.fridge.copy_worldcoords()

        av_seq_sol3 = task3.solve()
        if av_seq_sol3 is not None:
            task2 = OpeningTask(robot_model, 5)
            task2.reset_firdge_pose(*fridge_pose)
            task2.setup()
            task2.cm.add_eq_configuration(task2.n_wp-1, av_seq_sol3[0], force=True)
            av_seq_sol2 = task2.solve()
            if av_seq_sol2 is not None:
                task1 = ApproachingTask(robot_model, 8)
                task1.reset_firdge_pose(*fridge_pose)
                task1.setup(av_start, av_seq_sol2[0])
                av_seq_sol1 = task1.solve()
                if av_seq_sol1 is not None:
                    for task in [task1, task2, task3]:
                        task.dump_sol_cache()
                    return task1, task2, task3
        print("retry..")

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

def send_cmd_to_ri(ri, robot_model, joint_list, duration, av_seq):
    base_pose_seq = av_seq[:, -3:]

    full_av_seq = []
    for av in av_seq:
        set_robot_config(robot_model, joint_list, av, with_base=True)
        full_av_seq.append(robot_model.angle_vector())
    n_wp = len(full_av_seq)

    time_seq = [duration]*n_wp
    ri.angle_vector_sequence(full_av_seq, time_seq)
    ri.move_trajectory_sequence(base_pose_seq, time_seq, send_action=True)


if __name__=='__main__':
    #task1, task2, task3 = generate_solution_cache()
    get_current_pose = setup_rosnode()

    vis = Visualizer()
    np.random.seed(3)

    robot_model = pr2_init()
    joint_list = rarm_joint_list(robot_model)
    av_start = get_robot_config(robot_model, joint_list, with_base=True)

    fridge_pose = [[2.0, 1.5, 0.0], [0, 0, 0]]

    task3 = ReachingTask(robot_model, 10)
    task3.reset_firdge_pose(*fridge_pose)
    task3.setup()
    task3.load_sol_cache()
    #task3.solve(use_cache=True)

    task2 = OpeningTask(robot_model, 5)
    task2.reset_firdge_pose(*fridge_pose)
    task2.load_sol_cache()
    #task2.solve(use_cache=True)

    task1 = ApproachingTask(robot_model, 8)
    task1.reset_firdge_pose(*fridge_pose)
    task1.setup(av_start, task2.av_seq_cache[0])
    task1.load_sol_cache()
    task1.solve(use_cache=True)

    robot_model2 = pr2_init()
    robot_model2.fksolver = None
    ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model2)
    ri.move_gripper("rarm", pos=0.08)
    ri.angle_vector(robot_model2.angle_vector()) # copy angle vector to real robot

    def update():
        co = Coordinates()
        robot_model.newcoords(co)
        trans, rpy = get_current_pose()
        task3.reset_firdge_pose_from_handle_pose(trans, rpy)
        task2.reset_firdge_pose_from_handle_pose(trans, rpy)
        task1.reset_firdge_pose_from_handle_pose(trans, rpy)
        task1.setup(av_start, task2.av_seq_cache[0])
        task1.solve(use_cache=False)
        av_seq = np.vstack([task1.av_seq_cache, task2.av_seq_cache, task3.av_seq_cache])
        return av_seq



    print("start solving")
    av_seq = update()
    send_cmd_to_ri(ri, robot_model, joint_list, 3.0, av_seq)
    #send_cmd_to_ri(ri, robot_model, joint_list, 3.0, task2.av_seq_cache)
