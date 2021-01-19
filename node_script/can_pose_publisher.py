#!/usr/bin/env python 
import time
import rospy
import numpy as np
from std_msgs.msg import String
from posedetection_msgs.msg import ObjectDetection
from geometry_msgs.msg import PoseArray, Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import tf
import utils
import message_filters

from skrobot.coordinates.math import quaternion2rpy, quaternion2matrix, rpy2quaternion
from skrobot.coordinates import make_coords, rpy_matrix, matrix2quaternion
from skrobot.coordinates import rpy_angle

if __name__ == '__main__':
    rospy.init_node('can_pose_publisher', anonymous=True)
    listener = tf.TransformListener() 

    odom_topic_name = "/base_odometry/odom"
    pose_array_topic_name = "/ishida_demo/cluster_decomposer/centroid_pose_array"
    sub1 = message_filters.Subscriber(odom_topic_name, Odometry)
    sub2 = message_filters.Subscriber(pose_array_topic_name, PoseArray)
    pub = rospy.Publisher('pose_can_to_odom', Pose, queue_size=1)

    def callbcak(msg_odom, msg_pose_array):
        pos = msg_odom.pose.pose.position
        rot = msg_odom.pose.pose.orientation
        tf_base_to_odom = ([pos.x, pos.y, pos.z], [rot.x, rot.y, rot.z, rot.w])

        if not len(msg_pose_array.poses)==1:
            print("multiple/no objects are detected. goint to abort.")
            return

        try:
            source_frame = msg_pose_array.header.frame_id
            target_frame = "base_footprint"
            tf_camera_to_foot = listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0))
        except:
            print("cannot obtain proper transform")
            return 
        #tf_can_to_camera = 
        pos = msg_pose_array.poses[0].position
        rot = msg_pose_array.poses[0].orientation
        tf_can_to_camera = ([pos.x, pos.y, pos.z], [rot.x, rot.y, rot.z, rot.w])
        tf_can_to_base = utils.convert(tf_can_to_camera, tf_camera_to_foot)
        tf_can_to_odom = utils.convert(tf_can_to_base, tf_base_to_odom)
        pose_can_to_odom = utils.convert_tf2posemsg(tf_can_to_odom)
        pub.publish(pose_can_to_odom)

    ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2], 200, 0.2)
    ts.registerCallback(callbcak)
    print("node start")
    rospy.spin()
