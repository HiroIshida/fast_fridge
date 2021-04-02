#!/usr/bin/env python
import numpy as np
import rospy
from real_robot_demo import *
from std_srvs.srv import Trigger, TriggerResponse

try:
    demo
except:
    rospy.init_node('planner', anonymous=True)
    np.random.seed(3)
    demo = FridgeDemo()
demo.initialize_robot_pose()
demo.tf_can_to_world = None

def door_open_failure_recovery():
    print("FailureDoorOpeningException catched: enter recovery")
    demo.initialize_robot_pose()
    demo.ri.go_pos_unsafe_no_wait(*[0.1, 0.15, 0], sec=1.0)
    time.sleep(3.5)
    demo.update_fridge_pose()
    demo.solve_first_phase(send_action=True)
    demo.solve_while_second_phase(send_action=True)

def handle_start_plan(req):
    print("requested; and start ishida demo")
    demo.update_fridge_pose()
    try:
        demo.solve_first_phase(send_action=True)
        demo.solve_while_second_phase(send_action=True)
    except FailureDoorOpeningException:
        try:
            door_open_failure_recovery()
        except:
            TriggerResponse(string="abort1")
    time.sleep(1.0)

    try:
        demo.solve_third_phase(send_action=True)
    except PreReplanFailException:
        print("wait a bit for observing the can and solve again")
        time.sleep(2.0)
        demo.solve_third_phase(send_action=True)

    demo.ri.move_gripper("rarm", pos=0.2, wait=False)
    time.sleep(2.5)
    demo.ri.move_gripper("larm", pos=0.2, wait=False)
    time.sleep(2.5)
    demo.send_final_phase()

    return TriggerResponse()


s = rospy.Service('start_plan', Trigger, handle_start_plan)
rospy.spin()
