#!/usr/bin/env python 
import rospy
from std_msgs.msg import String
from posedetection_msgs.msg import ObjectDetection

topic_name = "/kinect_head/rgb/ObjectDetection"

global is_init
is_init = False
def callback(data):
    global is_init
    if not is_init:
        print("dummy node working well")
        is_init = True
    pass
    
def listener():
    rospy.init_node('dummy_listener', anonymous=True)
    rospy.Subscriber(topic_name, ObjectDetection, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
