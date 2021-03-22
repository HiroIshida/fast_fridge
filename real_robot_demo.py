from task import *
import threading
from nav_msgs.msg import Odometry
from skrobot.coordinates.math import wxyz2xyzw, matrix2quaternion
import tf

class TakingTooLongException(Exception):
    pass

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

def invert_tf(tf_inp):
    trans, rot = tf_inp
    rot_ = copy.copy(rot)
    for i in range(3):
        rot_[i] *= -1
    rot_ = rot_/np.linalg.norm(rot_)
    trans_ = qv_mult(rot_, [-e for e in trans])

    if np.isnan(trans_[0]):
        trans_ = [0, 0, 0]
    return [trans_, rot_]

def convert(tf_12, tf_23):
    tran_12, rot_12 = [np.array(e) for e in tf_12]
    tran_23, rot_23 = [np.array(e) for e in tf_23]

    rot_13 = tf.transformations.quaternion_multiply(rot_12, rot_23)
    tran_13 = tran_23 + qv_mult(rot_23, tran_12)
    return list(tran_13), list(rot_13)

def tf_from_xytheta(xytheta):
    x, y, theta = xytheta
    pos = [x, y, 0.0]
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
        self.duration = 0.35

        self.task_approach = ApproachingTask(self.robot_model, 10)
        self.task_open = OpeningTask(self.robot_model, 10)
        self.task_reach = ReachingTask(self.robot_model, 12)

        self.task_approach.load_sol_cache()
        self.task_open.load_sol_cache()
        self.task_reach.load_sol_cache()
        self.task_reach.load_trajectory_library()

        self.robot_model2 = pr2_init() # for robot interface
        self.robot_model2.fksolver = None
        self.ri = skrobot.interfaces.ros.PR2ROSRobotInterface(self.robot_model2)

        # ros stuff
        self.handle_pose = None
        sub_handle_pose = rospy.Subscriber(
                "handle_pose", Pose, self._handle_pose_callback)
        self.tf_base_to_odom = None
        self.tf_base_nominal_to_odom = None
        self.tf_base_nominal_to_world = None
        self.tf_can_to_world = None
        sub_odom = rospy.Subscriber("/base_odometry/odom", Odometry, self._odom_callback)

        sub_can_pose = rospy.Subscriber('pose_can_to_odom', Pose, self._can_pose_callback)

    def _can_pose_callback(self, msg):
        if self.tf_base_nominal_to_odom is None:
            return
        pos = msg.position
        rot = msg.orientation
        tf_can_to_odom = ([pos.x, pos.y, pos.z], [rot.x, rot.y, rot.z, rot.w])
        tf_can_to_base_nominal = convert(tf_can_to_odom, invert_tf(self.tf_base_nominal_to_odom))
        tf_can_to_world = convert(tf_can_to_base_nominal, self.tf_base_nominal_to_world)
        self.tf_can_to_world = tf_can_to_world

    def _odom_callback(self, msg):
        pos = msg.pose.pose.position
        rot = msg.pose.pose.orientation
        self.tf_base_to_odom = ([pos.x, pos.y, pos.z], [rot.x, rot.y, rot.z, rot.w])

    def _handle_pose_callback(self, msg):
        pos_msg = msg.position
        quat_msg = msg.orientation
        ypr = quaternion2rpy([quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z])[0]
        #rpy = [ypr[2], ypr[1], ypr[0]]
        # TODO adhoc workaround
        rpy = [0.0, 0.0, ypr[0]]
        pos = [pos_msg.x, pos_msg.y, pos_msg.z]
        self.handle_pose = [pos, rpy]

    def initialize_robot_pose(self):
        self.ri.move_gripper("rarm", pos=0.060)
        self.ri.move_gripper("larm", pos=0.0)
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
                use_cache=False)
        av_seq_sol = self.task_approach.solve(use_cache=False)
        if av_seq_sol is None:
            raise Exception # for safety

        if send_action:
            time_seq = self.task_approach.default_send_duration
            self._send_cmd(self.task_approach.av_seq_cache, time_seq=time_seq)
            time.sleep(sum(time_seq))

    def send_cmd_first_and_second_batch(self):
        av_seq_batch = np.vstack([self.task_approach.av_seq_cache, self.task_open.av_seq_cache])
        self._send_cmd(av_seq_batch)

    def solve_while_second_phase(self, send_action=False):
        """
        share_dict = {"pose": None, "is_running": True}


        log_prefix = "[SECOND] " 
        def keep_solvin():
            share_in_solving = {"ts_optimization": None}

            def callback(xk):
                ts_now = time.time()
                time_elapsed = ts_now - share_in_solving["ts_optimization"]
                if time_elapsed > 1.0:
                    rospy.loginfo(log_prefix + "aborted due to taking too much in solving. elapsed {0}".format(time_elapsed))
                    raise TakingTooLongException

            while share_dict["is_running"]:
                if self.tf_can_to_world is not None:
                    self.task_open.setup()
                    share_in_solving["ts_optimization"] = time.time()
                    trans = self.tf_can_to_world[0]
                    rospy.loginfo(log_prefix + "try to solve for can position {0}".format(trans))
                    self.task_reach.setup(position=trans)
                    try:
                        self.task_reach.solve(callback=callback)
                        rospy.loginfo(log_prefix + "successfully solved")
                    except TakingTooLongException:
                        pass
                rospy.loginfo(log_prefix + "aborted because tf_can_to_wrold is None")
        """

        if send_action:
            time_seq = self.task_open.default_send_duration
            self._send_cmd(self.task_open.av_seq_cache, time_seq=time_seq)
            self.ri.move_gripper("rarm", pos=0.0, effort=10000)
            time.sleep(sum(time_seq)-2.0)

    def solve_third_phase(self, send_action=False):
        counter = 0
        while (self.tf_can_to_world is None):
            if counter == 6:
                raise Exception
            time.sleep(0.5)
            counter += 1

        trans = self.tf_can_to_world[0]
        fridge_pos = self.task_open.fridge.worldpos()
        relative = trans - fridge_pos

        tmp_coords = self.task_open.fridge.copy_worldcoords()
        addhoc_translate = 0.02
        tmp_coords.translate([-0.05, addhoc_translate, -0.02])

        trans_modified = tmp_coords.worldpos() + relative

        av_start = self.task_open.av_seq_cache[-1]
        self.task_reach.setup(position=trans_modified, av_start=av_start, use_cache=True)

        ts = time.time()
        ret = self.task_reach.replanning()
        assert ret is not None
        rospy.loginfo("[replanning] elapsed time: {0}".format(time.time()-ts))

        if send_action:
            time_seq = self.task_reach.default_send_duration
            self._send_cmd(self.task_reach.av_seq_cache, time_seq=time_seq)

    def send_final_phase(self):
        vec_go_pos = np.array([0.12, 0.08])*1.0
        self.ri.go_pos_unsafe_no_wait(*vec_go_pos.tolist(), sec=2.0)
        time.sleep(1.5)
        self.ri.move_gripper("larm", pos=0.0, wait=False)
        self.ri.move_gripper("rarm", pos=0.0, wait=False)
        time.sleep(1.0)
        """
        # robomech version
        set_robot_config(self.task_reach.robot_model, self.joint_list, self.task_reach.av_seq_cache[-1], with_base=True)
        demo.task_reach.robot_model.larm.move_end_pos([-0.2, 0.0, 0.05])
        self.ri.angle_vector(demo.task_reach.robot_model.angle_vector(), time=1.0)
        """
        self.ri.go_pos_unsafe_no_wait(*((-vec_go_pos).tolist()), sec=2.0)
        time.sleep(1.0)
        av_seq_reverse = np.flip(self.task_reach.av_seq_cache, axis=0)
        self._send_cmd(av_seq_reverse[:-1])
        time.sleep(self.duration * len(av_seq_reverse)-0.3)

        #set_robot_config(self.task_reach.robot_model, self.joint_list, self.task_reach.av_seq_cache[-1], with_base=True)

        # TODO fix demo -> self
        demo.task_reach.robot_model.angle_vector(demo.ri.angle_vector())

        ret = demo.task_reach.robot_model.larm.move_end_pos(np.array([-0.3, 0, -0.4]), rotation_axis=None)
        demo.ri.angle_vector(demo.task_reach.robot_model.angle_vector(), time=1.0)
        time.sleep(0.6)

        diff = np.array([0.4, -0.2, 0.15])
        ret = demo.task_reach.robot_model.rarm.move_end_pos(diff, rotation_axis=None)
        ik_success = (ret is not None)
        assert ik_success
        print(ret)

        demo.ri.angle_vector(demo.task_reach.robot_model.angle_vector(), time=1.3)
        time.sleep(0.9)
        demo.ri.angle_vector(demo.robot_model2.angle_vector(), time=1.5, time_scale=1.0)

        """
        av_seq_reverse = np.flip(self.task_open.av_seq_cache, axis=0)
        self._send_cmd(av_seq_reverse)
        time.sleep(0.4 * len(av_seq_reverse))
        """


    def _send_cmd(self, av_seq, time_seq=None):
        def modify_base_pose(base_pose_seq):
            # TODO consider removing the first waypoint
            base_init = base_pose_seq[0]
            base_pose_seq = base_pose_seq - base_init

            angle_init = base_init[2]
            mat = np.array([[cos(angle_init), sin(angle_init)], [-sin(angle_init), cos(angle_init)]])
            base_pose_seq[:, :2] = base_pose_seq[:, :2].dot(mat.T)
            return base_pose_seq

        base_pose_seq = modify_base_pose(av_seq[:, -3:])

        full_av_seq = []
        for av in av_seq:
            set_robot_config(self.robot_model, self.joint_list, av, with_base=True)
            full_av_seq.append(self.robot_model.angle_vector())
        n_wp = len(full_av_seq)

        if time_seq is None:
            time_seq = [self.duration]*(n_wp - 1)
        assert len(time_seq) == (n_wp - 1)
        self.ri.angle_vector_sequence(full_av_seq[1:], time_seq)
        self.ri.move_trajectory_sequence(base_pose_seq[1:], time_seq, send_action=True)

        # set transforms for later tf computation
        self.tf_base_nominal_to_odom = self.tf_base_to_odom
        self.tf_base_nominal_to_world = tf_from_xytheta(av_seq[0, -3:])
        rospy.loginfo("[_send_cmd] start configuration of av_seq is : {0}".format(av_seq[0]))
        rospy.loginfo("[_send_cmd] start base config of av_seq is : {0}".format(av_seq[0][-3:]))

    def simulate(self, vis):
        vis.show_task(self.task_approach)
        vis.show_task(self.task_open)
        vis.show_task(self.task_reach)

if __name__=='__main__':
    try:
        vis
    except:
        rospy.init_node('planner', anonymous=True)
        vis = Visualizer()
        viewer = vis.viewer
        np.random.seed(3)
        demo = FridgeDemo()
    demo.initialize_robot_pose()
    demo.tf_can_to_world = None
    time.sleep(3)

    demo.update_fridge_pose()

    demo.solve_first_phase(send_action=True)
    demo.solve_while_second_phase(send_action=True)
    time.sleep(1.0)
    demo.solve_third_phase(send_action=True)
    time.sleep(2.5)
    demo.ri.move_gripper("rarm", pos=0.2, wait=False)
    demo.ri.move_gripper("larm", pos=0.2, wait=False)
    time.sleep(2.5)
    demo.send_final_phase()

