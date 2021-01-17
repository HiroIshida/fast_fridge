from task import *
import threading

class TransformManager(object):
    # TODO this should be define inside setup_rosnode
    def __init__(self):
        self.T_b_to_w = None # 'robot base when cmd is sent' to world transform
        self.T_bdash_to_b = None # current robot base to world transform
        self.T_obj_to_rdash = None # object to current robot base

    def set_b_to_w(xytheta):
        # convert it to transform
        pass

class FridgeDemo(object):
    def __init__(self):
        self.robot_model = pr2_init()

        self.joint_list = rarm_joint_list(self.robot_model)
        av_start = get_robot_config(self.robot_model, self.joint_list, with_base=True)
        self.av_start = av_start

        self.task_approach = ApproachingTask(self.robot_model, 10)
        self.task_open = OpeningTask(self.robot_model, 10)
        self.task_reach = ReachingTask(self.robot_model, 10)

        self.task_approach.load_sol_cache()
        self.task_open.load_sol_cache()
        self.task_reach.load_sol_cache()

        self.robot_model2 = pr2_init() # for robot interface
        self.robot_model2.fksolver = None
        self.ri = skrobot.interfaces.ros.PR2ROSRobotInterface(self.robot_model2)

        # real robot command stuff
        self.duration = 1.0

        # ros stuff
        self.handle_pose = None
        sub_handle_pose = rospy.Subscriber(
                "handle_pose", Pose, self._handle_pose_callback)

    def _handle_pose_callback(self, msg):
        pos_msg = msg.position
        quat_msg = msg.orientation
        ypr = quaternion2rpy([quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z])[0]
        rpy = [ypr[2], ypr[1], ypr[0]]
        pos = [pos_msg.x, pos_msg.y, pos_msg.z]
        self.handle_pose = [pos, rpy]

    def initialize_robot_pose(self):
        self.ri.move_gripper("rarm", pos=0.08)
        self.ri.angle_vector(self.robot_model2.angle_vector()) # copy angle vector to real robot

    def update_fridge_pose(self):
        assert self.handle_pose is not None, "handle pose should be observed"

        trans, rpy = self.handle_pose
        tasks = [self.task_reach, self.task_open, self.task_approach] 
        for task in tasks:
            task.reset_fridge_pose_from_handle_pose(trans, rpy)

    def solve_first_phase(self, send_action=False):
        co = Coordinates()
        self.robot_model.newcoords(co)
        self.task_open.setup(use_cache=True)
        self.task_approach.setup(
                av_start=self.av_start,
                av_final=self.task_open.av_seq_cache[0],
                use_cache=True)
        self.task_approach.solve(use_cache=True)

        if send_action:
            self._send_cmd(self.task_approach.av_seq_cache)

    def solve_while_second_phase(self, send_action=False):
        share_dict = {"pose": None, "result": None, "is_running": True}

        def keep_solvin():
            while share_dict["is_running"]:
                self.task_reach.setup(position=None)
                self.task_reach.solve()

        thread = threading.Thread(target=keep_solvin)
        thread.start()
        if send_action:
            self._send_cmd(self.task_open.av_seq_cache)
        time.sleep(5) # TODO auto set
        share_dict["is_running"] = False

    def _send_cmd(self, av_seq):
        def modify_base_pose(base_pose_seq):
            # TODO consider removing the first waypoint
            base_init = base_pose_seq[0]
            base_pose_seq = base_pose_seq - base_init
            return base_pose_seq

        base_pose_seq = modify_base_pose(av_seq[:, -3:])

        full_av_seq = []
        for av in av_seq:
            set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
            full_av_seq.append(self.robot_model.angle_vector())
        n_wp = len(full_av_seq)

        time_seq = [self.duration]*n_wp
        self.ri.angle_vector_sequence(full_av_seq, time_seq)
        self.ri.move_trajectory_sequence(base_pose_seq, time_seq, send_action=True)

    def simulate(self, vis):
        vis.show_task(self.task_approach)
        vis.show_task(self.task_open)
        vis.show_task(self.task_reach)

if __name__=='__main__':
    rospy.init_node('planner', anonymous=True)
    vis = Visualizer()
    np.random.seed(3)
    demo = FridgeDemo()
    demo.initialize_robot_pose()
    time.sleep(3)

    demo.update_fridge_pose()
    #demo.solve_first_phase(send_action=True)
    demo.solve_while_second_phase(send_action=True)
    #demo.simulate(vis)
