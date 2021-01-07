#!/usr/bin/env python 
import rospy
from std_msgs.msg import String
from posedetection_msgs.msg import ObjectDetection
from geometry_msgs.msg import Pose
import tf
import utils

class PoseProcessor(object):
    def __init__(self):
        topic_name = "/kinect_head/rgb/ObjectDetection"
        self.sub = rospy.Subscriber(topic_name, ObjectDetection, self.cb_object_detection)
        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher('handle_pose', Pose, queue_size=1)

        self.handle_pose_msg = None

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
        msg_handle_to_map = utils.convert_tf2posemsg(tf_handle_to_map)
        self.handle_pose_msg = msg_handle_to_map

    def publish_handle_pose_msg(self):
        pose_msg = self.handle_pose_msg
        if pose_msg is not None:
            self.pub.publish(pose_msg)

if __name__ == '__main__':
    rospy.init_node('dummy_listener', anonymous=True)
    pp = PoseProcessor()
    rate = rospy.Rate(30) # 10hz
    while not rospy.is_shutdown():
        pp.publish_handle_pose_msg()
        rate.sleep()
