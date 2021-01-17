from task import *

class TransformManager(object):
    # TODO this should be define inside setup_rosnode
    def __init__(self):
        self.T_b_to_w = None # 'robot base when cmd is sent' to world transform
        self.T_bdash_to_b = None # current robot base to world transform
        self.T_obj_to_rdash = None # object to current robot base

    def set_b_to_w(xytheta):
        # convert it to transform
        pass

def send_cmd_to_ri(ri, robot_model, joint_list, duration, av_seq):

    def modify_base_pose(base_pose_seq):
        # TODO consider removing the first waypoint
        base_init = base_pose_seq[0]
        base_pose_seq = base_pose_seq - base_init
        return base_pose_seq

    base_pose_seq = modify_base_pose(av_seq[:, -3:])

    full_av_seq = []
    for av in av_seq:
        set_robot_config(robot_model, joint_list, av, with_base=True)
        full_av_seq.append(robot_model.angle_vector())
    n_wp = len(full_av_seq)

    time_seq = [duration]*n_wp
    ri.angle_vector_sequence(full_av_seq, time_seq)
    ri.move_trajectory_sequence(base_pose_seq, time_seq, send_action=True)

class FridgeDemo(object):
    def __init__(self):
        self.robot_model = pr2_init()

        joint_list = rarm_joint_list(self.robot_model)
        av_start = get_robot_config(self.robot_model, joint_list, with_base=True)
        self.av_start = av_start

        self.task_approach = ApproachingTask(self.robot_model, 10)
        self.task_open = OpeningTask(self.robot_model, 10)
        self.task_reach = ReachingTask(self.robot_model, 10)

        self.task_approach.load_sol_cache()
        self.task_open.load_sol_cache()
        self.task_reach.load_sol_cache()

        robot_model2 = pr2_init() # for robot interface
        robot_model2.fksolver = None
        self.ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model2)

    def initialize_robot_pose(self):
        self.ri.move_gripper("rarm", pos=0.08)
        self.ri.angle_vector(robot_model2.angle_vector()) # copy angle vector to real robot

    def update_fridge_pose(self, handle_pose):
        trans, rpy = handle_pose
        tasks = [self.task_reach, self.task_open, self.task_approach] 
        for task in tasks:
            task.reset_fridge_pose_from_handle_pose(trans, rpy)

    def solve_first_phase(self):
        co = Coordinates()
        self.robot_model.newcoords(co)
        self.task_open.setup(use_cache=True)
        self.task_approach.setup(
                av_start=self.av_start,
                av_final=self.task_open.av_seq_cache[0],
                use_cache=True)
        self.task_approach.solve(use_cache=True)

    def simulate(self, vis):
        vis.show_task(self.task_approach)
        vis.show_task(self.task_open)
        vis.show_task(self.task_reach)

def setup_rosnode():
    rospy.init_node('planner', anonymous=True)
    inner_shared = {"handle_pose": None, "feedback_status": None, "can_pose": None}

    def cb_handle_pose(msg):
        pos_msg = msg.position
        quat_msg = msg.orientation
        ypr = quaternion2rpy([quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z])[0]
        rpy = [ypr[2], ypr[1], ypr[0]]
        pos = [pos_msg.x, pos_msg.y, pos_msg.z]
        inner_shared["handle_pose"] = [pos, rpy]

    def cb_feedback(msg):
        state = msg.feedback.actual.positions
        inner_shared["feedback_status"] = state

    def cb_can_pose(msg):
        p = msg.position
        q = msg.orientation
        tf_can_to_base = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        inner_shared["can_pose"] = tf_can_to_base

    topic_name_handle_pose = "handle_pose"
    topic_name_can_pose = "can_pose"
    topic_name_feedback = "/base_controller/follow_joint_trajectory/feedback"
    sub1 = rospy.Subscriber(topic_name_handle_pose, Pose, cb_handle_pose)
    sub2 = rospy.Subscriber(topic_name_feedback, FollowJointTrajectoryActionFeedback, cb_feedback)
    sub3 = rospy.Subscriber(topic_name_can_pose, Pose, cb_can_pose)

    get_can_pose = (lambda : inner_shared["can_pose"])
    get_handle_pose = (lambda : inner_shared["handle_pose"])
    get_feedback_state = (lambda : inner_shared["feedback_status"])
    # TODO TransformManager should be returned from here instead of returning get_can_pose and get_feedback_state
    return get_handle_pose, get_can_pose, get_feedback_state

if __name__=='__main__':
    vis = Visualizer()
    get_handle_pose, get_can_pose, get_feedback_state = setup_rosnode()
    np.random.seed(3)
    
    demo = FridgeDemo()
    demo.update_fridge_pose(get_handle_pose())
    demo.solve_first_phase()
    demo.simulate(vis)

    """
    robot_model = pr2_init()

    fridge_pose = [[2.0, 1.5, 0.0], [0, 0, 0]]

    hpose = get_handle_pose()
    task3 = ReachingTask(robot_model, 10)
    task3.reset_fridge_pose(*fridge_pose)
    task3.load_sol_cache()
    task3.setup(position=None)

    task2 = OpeningTask(robot_model, 10)
    task2.reset_fridge_pose(*fridge_pose)
    task2.load_sol_cache()
    task2.setup()

    task1 = ApproachingTask(robot_model, 10)
    task1.reset_fridge_pose(*fridge_pose)
    task1.load_sol_cache()
    task1.setup(av_start=av_start, av_final=task2.av_seq_cache[0])
    task1.solve(use_cache=True)

    robot_model2 = pr2_init()
    robot_model2.fksolver = None
    ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model2)
    ri.move_gripper("rarm", pos=0.08)
    ri.angle_vector(robot_model2.angle_vector()) # copy angle vector to real robot
    time.sleep(3)

    def first_update():
        co = Coordinates()
        robot_model.newcoords(co)
        trans, rpy = get_handle_pose()
        task2.reset_fridge_pose_from_handle_pose(trans, rpy)
        task2.setup()
        task1.reset_fridge_pose_from_handle_pose(trans, rpy)
        task1.setup(av_start=av_start, av_final=task2.av_seq_cache[0])
        task1.solve(use_cache=True)
        av_seq = np.vstack([task1.av_seq_cache, task2.av_seq_cache, task3.av_seq_cache])
        return av_seq

    print("start solving")
    av_seq = first_update()
    send_cmd_to_ri(ri, robot_model, joint_list, 1.0, task2.av_seq_cache)
    """
