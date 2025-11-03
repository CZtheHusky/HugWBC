import numpy as np

#q shape([4,], v shape([3,])
def quat_rotate_inverse(q, v):
  # q: w, x, y, z
  q_w = q[0]
  q_vec = q[1:]
  a = v * (2.0 * q_w ** 2 - 1.0)
  # b = np.cross(q_vec, v) * q_w * 2.0
  # directly compute the cross is faster than np.cross
  b = np.array([q_vec[1]*v[2] - v[1] * q_vec[2], q_vec[2]*v[0] - v[2] * q_vec[0], q_vec[0]*v[1] - v[0] * q_vec[1]]) \
       * q_w * 2.0
  c = q_vec * np.dot(q_vec, v) * 2.0
  return a - b + c

from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    # w = np.dot(RzWaist, imu_omega) - np.array([0, 0, waist_yaw_omega])
    # return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w
    vec = np.array([0, 0, -1])
    w = np.dot(RzWaist, imu_omega) - np.array([0, 0, waist_yaw_omega])
    vec = np.dot(R_pelvis.T, vec)
    return vec, w
