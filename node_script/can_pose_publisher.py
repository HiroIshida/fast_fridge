#!/usr/bin/env python 
import time
import rospy
import numpy as np
from std_msgs.msg import String
from posedetection_msgs.msg import ObjectDetection
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import tf
import utils

from skrobot.coordinates.math import quaternion2rpy, quaternion2matrix, rpy2quaternion
from skrobot.coordinates import make_coords, rpy_matrix, matrix2quaternion
from skrobot.coordinates import rpy_angle

if __name__ == '__main__':
    rospy.init_node('can_pose_publisher', anonymous=True)
    listener = tf.TransformListener() 

    def callback_odom(msg):
        pos = msg.pose.pose.position
        rot = msg.pose.pose.orientation
        tf_base_to_odom = ([pos.x, pos.y, pos.z], [rot.x, rot.y, rot.z, rot.w])
        print(tf_base_to_odom)

    odom_topic_name = "/base_odometry/odom"
    sub_feedback = rospy.Subscriber(
        odom_topic_name, Odometry, callback_odom)


    def callback_centroid_pose_array(msg):
        if not len(msg.poses)==1:
            print("multiple/no objects are detected. goint to abort.")
            return

        try:
            source_frame = msg.header.frame_id
            target_frame = "base_footprint"
            tf_camera_to_foot = listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0))
        except:
            print("cannot obtain proper transform")
            return 
        #tf_can_to_camera = 
        pos = msg.poses[0].position
        rot = msg.poses[0].orientation
        tf_can_to_camera = ([pos.x, pos.y, pos.z], [rot.x, rot.y, rot.z, rot.w])
        ts = time.time()
        tf_can_to_foot = utils.convert(tf_can_to_camera, tf_camera_to_foot)

    pose_array_topic_name = "/ishida_demo/cluster_decomposer/centroid_pose_array"
    sub1 = rospy.Subscriber(pose_array_topic_name, PoseArray, callback_centroid_pose_array)
    rospy.spin()

    """
    import time
    while not rospy.is_shutdown():
        time.sleep(0.05)
    """


""" topic example for feedback
---
header: 
  seq: 43463
  stamp: 
    secs: 1611019069
    nsecs: 417159258
  frame_id: ''
status: 
  goal_id: 
    stamp: 
      secs: 1611019062
      nsecs: 401598930
    id: "/planner_29606_1611019038121-10-1611019062.402"
  status: 1
  text: ''
feedback: 
  header: 
    seq: 0
    stamp: 
      secs: 1611019069
      nsecs: 417128433
    frame_id: ''
  joint_names: 
    - base_link_x
    - base_link_y
    - base_link_pan
  desired: 
    positions: [-0.9482321981037277, -0.16833751222783505, -6.874734376531748]
    velocities: [-0.0006569418651102762, 0.0024462702019135696, -0.007821040390345257]
    accelerations: []
    effort: []
    time_from_start: 
      secs: 0
      nsecs:         0
  actual: 
    positions: [-0.8960264298472721, -0.36732265076155773, 0.4159330182925426]
    velocities: [0.05963064357638359, 0.2533145844936371, -0.45493537187576294]
    accelerations: []
    effort: []
    time_from_start: 
      secs: 0
      nsecs:         0
  error: 
    positions: [-0.05220576825645562, 0.19898513853372268, -1.0074820876447044]
    velocities: [1.7976931348623157e+308, 1.7976931348623157e+308, 1.7976931348623157e+308]
    accelerations: []
    effort: []
    time_from_start: 
      secs: 0
      nsecs:         0
---
"""
