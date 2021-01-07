#!/usr/bin/env python 
import rospy
import numpy as np
from std_msgs.msg import String
from posedetection_msgs.msg import ObjectDetection
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf
import utils

class PoseProcessor(object):
    def __init__(self):
        topic_name_detection = "/kinect_head/rgb/ObjectDetection"
        #topic_name_amcl = "/amcl_pose"
        self.sub1 = rospy.Subscriber(topic_name_detection, ObjectDetection, self.cb_object_detection)
        #self.sub2 = rospy.Subscriber(topic_name_amcl, PoseWithCovarianceStamped, self.cb_amcl)

        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher('handle_pose', Pose, queue_size=1)

        self.handle_pose = None
        self.rough_handle_pose = None

    def cb_object_detection(self, msg):
        assert len(msg.objects)==1
        tf_handle_to_camera = utils.convert_pose2tf(msg.objects[0].pose)
        target_frame  = "base_link"
        source_frame = msg.header.frame_id
        try:
            tf_camera_to_map = self.listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0))
        except:
            print("cannot obtain transform")
            return
        tf_handle_to_map = utils.convert(tf_handle_to_camera, tf_camera_to_map)
        self.handle_pose = tf_handle_to_map

    def publish_handle_pose_msg(self):
        if self.handle_pose is not None:
            msg_handle_to_map = utils.convert_tf2posemsg(self.handle_pose)
            self.pub.publish(msg_handle_to_map)
        elif self.rough_handle_pose is not None:
            msg_handle_to_map = utils.convert_tf2posemsg(self.rough_handle_pose)
            self.pub.publish(msg_handle_to_map)

    def relative_fridge_pose(self):
        try:
            tf_base_to_map = self.listener.lookupTransform(
                "map", "base_link", rospy.Time(0))
        except:
            return
        current_position, current_quat = tf_base_to_map
        fridge_pos = np.array([5.7, 7.6, 0.0])
        handle_pos = fridge_pos + np.array([-0.25, 0.2, 1.07])

        diff = handle_pos - np.array(current_position)
        current_quat[3] *= -1
        self.rough_handle_pose = [diff, current_quat]

if __name__ == '__main__':
    rospy.init_node('dummy_listener', anonymous=True)
    pp = PoseProcessor()
    rate = rospy.Rate(30) # 10hz
    print("node start")
    while not rospy.is_shutdown():
        pp.publish_handle_pose_msg()
        pp.relative_fridge_pose()
        rate.sleep()
