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

# initialization stuff
np.random.seed(0)
robot_model = pr2_init()

fridge = Fridge()
fridge.translate([2.2, 2.0, 0.0])

joint_list = rarm_joint_list(robot_model)

with_base = True

# constraint manager
n_wp = 20
cm = ConstraintManager(n_wp, [j.name for j in joint_list], robot_model.fksolver, with_base)
update_fksolver(robot_model)

av_start = get_robot_config(robot_model, joint_list, with_base=with_base)
cm.add_eq_configuration(0, av_start)
cm.add_pose_constraint(n_wp-1, "l_gripper_tool_frame", [1.5, 2.3, 1.3])

sdf_list = [fridge.gen_sdf(0.0) for i in range(n_wp)]
angle_open = 0.8
angles = np.linspace(0, angle_open, 5)
for i, angle in zip(range(5), angles):
    idx = 10 + i
    sdf = fridge.gen_sdf(angle)
    sdf_list[idx] = sdf
    pose = fridge.grasping_gripper_pose(angle)
    cm.add_pose_constraint(idx, "r_gripper_tool_frame", pose)
    sdf_list.append(sdf)

sscc = TinyfkSweptSphereSdfCollisionChecker(sdf_list, robot_model)
for link in rarm_coll_link_list(robot_model):
    sscc.add_collision_link(link)

av_current = get_robot_config(robot_model, joint_list, with_base=with_base)
av_seq_init = cm.gen_initial_trajectory(av_current)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
viewer.add(robot_model)
viewer.add(fridge)
cv = ConstraintViewer(viewer, cm)
cv.show()
viewer.show()

solve = True

if solve:
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    av_seq = tinyfk_sqp_plan_trajectory(
        sscc, cm, av_seq_init, joint_list, n_wp,
        safety_margin=1e-2, with_base=with_base)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=True))
else:
    av_seq = av_seq_init

for av, idx in zip(av_seq, range(len(av_seq))):
    set_robot_config(robot_model, joint_list, av, with_base=with_base)
    if idx>9 and idx<=14:
        fridge.set_angle(angles[idx-10])
    viewer.redraw()
    time.sleep(1.0)

print('==> Press [q] to close window')
while not viewer.has_exit:
    time.sleep(0.1)
    fridge
    viewer.redraw()
