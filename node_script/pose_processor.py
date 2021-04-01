#!/usr/bin/env python 
import rospy
import numpy as np
from std_msgs.msg import String
from posedetection_msgs.msg import ObjectDetection
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf
import utils

from skrobot.coordinates.math import quaternion2rpy, quaternion2matrix
from skrobot.coordinates import make_coords, rpy_matrix, matrix2quaternion
from skrobot.coordinates import rpy_angle

def tf2skcoords(transform):
    trans, rotation = transform
    position = [trans[0], trans[1], trans[2]]
    quaternion = [rotation[3], rotation[0], rotation[1], rotation[2]]
    mat = quaternion2matrix(quaternion)
    co = make_coords(pos = position, rot=mat)
    return co

class AverageQueue(object):
    def __init__(self, N_average, dim):
        self.N_average = N_average
        self.dim = dim
        self.data = np.zeros((N_average, dim))

    def push(self, elem):
        assert len(elem) == self.dim
        data_remain = self.data[:self.N_average - 1, :]
        self.data = np.vstack([elem, data_remain])
        assert self.data.shape == (self.N_average, self.dim)

    def get_average(self):
        return np.mean(self.data, axis=0)


class PoseProcessor(object):
    def __init__(self):
        topic_name_detection = "/kinect_head/rgb/ObjectDetection"
        #topic_name_amcl = "/amcl_pose"
        self.sub1 = rospy.Subscriber(topic_name_detection, ObjectDetection, self.cb_object_detection)
        #self.sub2 = rospy.Subscriber(topic_name_amcl, PoseWithCovarianceStamped, self.cb_amcl)

        self.broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher('handle_pose', Pose, queue_size=1)

        self.handle_pose = None
        self.rough_handle_pose = None
        N_average = 10
        self.aveque = AverageQueue(N_average, 7)

    def cb_object_detection(self, msg):
        assert len(msg.objects)==1
        tf_handle_to_camera = utils.convert_pose2tf(msg.objects[0].pose)
        target_frame  = "base_link"
        source_frame = msg.header.frame_id
        try:
            tf_camera_to_base = self.listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0))
        except:
            print("cannot obtain transform")
            return
        sktf_camera_to_handle = tf2skcoords(tf_handle_to_camera) # gyaku?
        sktf_base_to_camera = tf2skcoords(tf_camera_to_base)
        sktf_base_to_handle = sktf_base_to_camera.transform(sktf_camera_to_handle)
        pos = sktf_base_to_handle.worldpos()
        rot = sktf_base_to_handle.worldrot()
        skquaternion = matrix2quaternion(rot)
        w, x, y, z = skquaternion
        quaternion = [x, y, z, w]
        self.broadcaster.sendTransform(
                pos, quaternion,
                rospy.Time.now(),
                "fridge_handle",
                "base_link")

        self.aveque.push(np.hstack((pos, quaternion)))
        pose_average = self.aveque.get_average()
        pos = pose_average[:3].tolist()
        quat_tmp = pose_average[3:]
        quat = (quat_tmp / np.linalg.norm(quat_tmp)).tolist()
        self.handle_pose = [pos, quat]

        self.broadcaster.sendTransform(
                pos, quat,
                rospy.Time.now(),
                "fridge_handle_filtered",
                "base_link")

    def publish_handle_pose_msg(self):
        if self.handle_pose is not None:
            msg_handle_to_map = utils.convert_tf2posemsg(self.handle_pose)
            self.pub.publish(msg_handle_to_map)
            print("publish sift")
            print(self.handle_pose)

        # NOTE Do not publish rough handle pose! it's really dangerous
        """ 
        elif self.rough_handle_pose is not None:
            msg_handle_to_map = utils.convert_tf2posemsg(self.rough_handle_pose)
            self.pub.publish(msg_handle_to_map)
            print("publish rough")
            print(self.rough_handle_pose)
        """

    def relative_fridge_pose(self):
        try:
            tf_base_to_map = self.listener.lookupTransform(
                "map", "base_link", rospy.Time(0))
        except:
            return
        current_position, current_quat = tf_base_to_map
        #fridge_pos = np.array([5.7, 7.6, 0.0])

        mat = quaternion2matrix([current_quat[3], current_quat[0], current_quat[1], current_quat[2]])
        robot_pose = make_coords(current_position, mat)

        fridge_pos = np.array([5.7, 7.6, 0.0])
        handle_pos = fridge_pos + np.array([-0.33, 0.23, 1.1])
        handle_pose = make_coords(handle_pos)
        #pose = handle_pose.inverse_transformation().transform(robot_pose)


        pose_diff = robot_pose.inverse_transformation().transform(handle_pose)
        quat_diff_ = matrix2quaternion(pose_diff.rotation)
        quat_diff = [quat_diff_[1], quat_diff_[2], quat_diff_[3], quat_diff_[0]]
        position_diff = pose_diff.worldpos()

        self.rough_handle_pose = [position_diff, quat_diff]

if __name__ == '__main__':
    rospy.init_node('dummy_listener', anonymous=True)
    pp = PoseProcessor()
    rate = rospy.Rate(30) # 10hz
    print("node start")
    import time
    while not rospy.is_shutdown():
        pp.publish_handle_pose_msg()
        pp.relative_fridge_pose()
        time.sleep(0.05)
        #rate.sleep()
