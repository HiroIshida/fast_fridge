
Node [/kinect_head/rgb/fridge_demo_sift]
Publications: 
 * /diagnostics [diagnostic_msgs/DiagnosticArray]
 * /kinect_head/rgb/Feature0D [posedetection_msgs/Feature0D]
 * /kinect_head/rgb/ImageFeature0D [posedetection_msgs/ImageFeature0D]
 * /rosout [rosgraph_msgs/Log]

Subscriptions: None

Services: 
 * /kinect_head/rgb/Feature0DDetect
 * /kinect_head/rgb/fridge_demo_sift/get_loggers
 * /kinect_head/rgb/fridge_demo_sift/list
 * /kinect_head/rgb/fridge_demo_sift/load_nodelet
 * /kinect_head/rgb/fridge_demo_sift/set_logger_level
 * /kinect_head/rgb/fridge_demo_sift/unload_nodelet

Feature0D
rostopic echo /kinect_head/rgb/Feature0DDetect

# これをよまないと何も表示されない. 
rostopic echo /kinect_head/rgb/ObjectDetection
