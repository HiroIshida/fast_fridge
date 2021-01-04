#!/usr/bin/env python
import time

import numpy as np

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.planner import tinyfk_sqp_plan_trajectory
from skrobot.planner import TinyfkSweptSphereSdfCollisionChecker
from skrobot.planner import ConstraintManager
from skrobot.planner import ConstraintViewer
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from skrobot.planner.utils import update_fksolver
from pr2opt_common import *
from door import Fridge
import copy

# initialization stuff
np.random.seed(0)
robot_model = pr2_init()



joint_list = rarm_joint_list(robot_model)

with_base = True
full_demo = False
fridge = Fridge(full_demo)
fridge.translate([2.2, 2.0, 0.0])

# constraint manager
if full_demo:
    n_wp = 20
    cm = ConstraintManager(n_wp, joint_list, robot_model.fksolver, with_base)
    update_fksolver(robot_model)

    av_start = get_robot_config(robot_model, joint_list, with_base=with_base)
    cm.add_eq_configuration(0, av_start)
    cm.add_pose_constraint(n_wp-1, "l_gripper_tool_frame", [1.85, 2.1, 1.0])

    angle_open = 0.8
    angles = np.linspace(0, angle_open, 5)
    k_start = 10
    k_end = 14
    sdf_list = fridge.gen_door_open_sdf_list(n_wp, k_start, k_end, angle_open)
    for idx, pose in fridge.gen_door_open_coords(k_start, k_end, angle_open):
        cm.add_pose_constraint(idx, "r_gripper_tool_frame", pose)

else:
    n_wp = 12
    cm = ConstraintManager(n_wp, joint_list, robot_model.fksolver, with_base)
    update_fksolver(robot_model)

    av_start = get_robot_config(robot_model, joint_list, with_base=with_base)
    cm.add_eq_configuration(0, av_start)

    angle_open = 0.8
    angles = np.linspace(0, angle_open, 5)
    k_start = 8
    k_end = 11
    sdf_list = fridge.gen_door_open_sdf_list(n_wp, k_start, k_end, angle_open)
    for idx, pose in fridge.gen_door_open_coords(k_start, k_end, angle_open):
        cm.add_pose_constraint(idx, "r_gripper_tool_frame", pose)

def simulate_fridge(idx):
    if idx>k_start-1 and idx<=k_end:
        fridge.set_angle(angles[idx-k_start])



sscc = TinyfkSweptSphereSdfCollisionChecker(sdf_list, robot_model)
sscc2 = TinyfkSweptSphereSdfCollisionChecker(sdf_list[-1], robot_model)
for link in rarm_coll_link_list(robot_model):
    sscc.add_collision_link(link)
    sscc2.add_collision_link(link)
av_current = get_robot_config(robot_model, joint_list, with_base=with_base)
av_seq_init = cm.gen_initial_trajectory(av_init=av_current, collision_checker=sscc2)

solve = True

if solve:
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    slsqp_option = {'ftol': 1e-3, 'disp': True, 'maxiter': 100}
    av_seq = tinyfk_sqp_plan_trajectory(
        sscc, cm, av_seq_init, joint_list, n_wp,
        safety_margin=3e-2, with_base=with_base, slsqp_option=slsqp_option)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=True))

    if not full_demo: # MPC
        ts = time.time()
        av_seq[0] = av_seq[1]
        av_seq += 0.05
        cm.add_eq_configuration(0, av_seq[1], force=True)
        av_seq = tinyfk_sqp_plan_trajectory(
            sscc, cm, av_seq, joint_list, n_wp,
            safety_margin=3e-2, with_base=with_base, slsqp_option=slsqp_option)
        print("elapsed : {}".format(time.time() - ts))
else:
    av_seq = av_seq_init

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
viewer.add(robot_model)
viewer.add(fridge)
cv = ConstraintViewer(viewer, cm)
cv.show()
viewer.show()

for av, idx in zip(av_seq, range(len(av_seq))):
    set_robot_config(robot_model, joint_list, av, with_base=with_base)
    simulate_fridge(idx)
    viewer.redraw()
    time.sleep(0.3)
