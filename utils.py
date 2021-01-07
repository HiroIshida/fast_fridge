import numpy as np
import tf
from geometry_msgs.msg import Pose

def convert_tf2posemsg(transform):
    trans, rot = transform
    pose_msg = Pose()
    pose_msg.position.x = trans[0]
    pose_msg.position.y = trans[1]
    pose_msg.position.z = trans[2]

    pose_msg.orientation.x = rot[0]
    pose_msg.orientation.y = rot[1]
    pose_msg.orientation.z = rot[2]
    pose_msg.orientation.w = rot[3]
    return pose_msg

def convert_pose2tf(pose):
    pos = pose.position
    rot = pose.orientation
    trans = [pos.x, pos.y, pos.z]
    rot = [rot.x, rot.y, rot.z, rot.w]
    return (trans, rot)

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
