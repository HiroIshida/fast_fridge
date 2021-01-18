from task import *
import threading

def qv_mult(q1, v1_):
    length = np.linalg.norm(v1_)
    v1 = v1_/length
    v1 = tf.transformations.unit_vector(v1)
    q2 = list(v1)
    q2.append(0.0)
    v_converted = tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2), 
        tf.transformations.quaternion_conjugate(q1)
    )[:3]
    return v_converted * length

def convert(tf_12, tf_23):
    tran_12, rot_12 = [np.array(e) for e in tf_12]
    tran_23, rot_23 = [np.array(e) for e in tf_23]

    rot_13 = tf.transformations.quaternion_multiply(rot_12, rot_23)
    tran_13 = tran_23 + qv_mult(rot_23, tran_12)
    return list(tran_13), list(rot_13)

def tf_from_xytheta(xytheta):
    x, y, theta = xytheta
    pos = [x, y, theta]
    rot = rpy_matrix(*[theta, 0, 0])
    quat = wxyz2xyzw(matrix2quaternion(rot))
    tf = [pos, quat.tolist()]
    return tf

class TransformManager(object):
    # TODO this should be define inside setup_rosnode
    def __init__(self):
        self.T_b_to_w = None # 'robot base when cmd is sent' to world transform
        self.T_bdash_to_b = None # current robot base to world transform
        self.T_obj_to_bdash = None # object to current robot base

        sub_object_pose = rospy.Subscriber(
                "object_pose", Pose, self._object_pose_callback)
        self._reset_bdash_to_b()

    def _reset_bdash_to_b(self):
        self.T_bdash_to_b = [[0, 0, 0.], [0, 0, 0, 1.0]]

    def _object_pose_callback(self, msg):
        # msg must be posestamped
        pos = msg.pose.position
        quat = msg.pose.orientation
        self.T_obj_to_bdash = [[pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]]

    def set_b_to_w(xytheta):
        # from simulater
        self.T_b_to_w = tf_from_xytheta(xytheta)
        self._reset_bdash_to_b()

    def get_object_world_pose(self):
        assert (self.T_obj_to_bdash is not None)
        assert (self.T_bdash_to_b is not None)
        assert (self.T_b_to_w is not None)
        T_obj_to_w = convert(convert(self.T_obj_to_bdash, self.T_bdash_to_b), self.T_b_to_w)
        return T_obj_to_w

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

        self.tf_manager = TransformManager()

    def _handle_pose_callback(self, msg):
        pos_msg = msg.position
        quat_msg = msg.orientation
        ypr = quaternion2rpy([quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z])[0]
        rpy = [ypr[2], ypr[1], ypr[0]]
        pos = [pos_msg.x, pos_msg.y, pos_msg.z]
        self.handle_pose = [pos, rpy]

    def initialize_robot_pose(self):
        self.ri.move_gripper("rarm", pos=0.08)
        self.ri.angle_vector(self.robot_model2.angle_vector(), time=2.5, time_scale=1.0) # copy angle vector to real robot

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
            time.sleep(self.duration * len(self.task_approach.av_seq_cache))

    def solve_while_second_phase(self, send_action=False):
        share_dict = {"pose": None, "is_running": True}

        def keep_solvin():
            while share_dict["is_running"]:
                self.task_open.setup()
                self.task_reach.setup(position=None)
                self.task_reach.solve()

        thread = threading.Thread(target=keep_solvin)
        thread.start()
        if send_action:
            self._send_cmd(self.task_open.av_seq_cache)
        time.sleep(self.duration * len(self.task_open.av_seq_cache))
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
    #is.show_task(demo.task_approach)
    demo.solve_first_phase(send_action=False)
    """
    demo.solve_while_second_phase(send_action=False)
    """
    #demo._send_cmd(demo.task_reach.av_seq_cache)
    #demo.simulate(vis)
