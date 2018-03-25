import numpy as np
np.set_printoptions(precision=4)


def euler_to_quaternion(angles):
    roll = angles[0] # \phi
    pitch = angles[1] # \theta
    yaw = angles[2] # \psi

    # TODO: complete the conversion
    # and return a numpy array of
    # 4 elements representing a quaternion [a, b, c, d]
    a = np.cos(roll/2.0)*np.cos(pitch/2.0)*np.cos(yaw/2.0) + np.sin(roll/2.0)*np.sin(pitch/2.0)*np.sin(yaw/2.0)
    b = np.sin(roll/2.0)*np.cos(pitch/2.0)*np.cos(yaw/2.0) - np.cos(roll/2.0)*np.sin(pitch/2.0)*np.sin(yaw/2.0)
    c = np.cos(roll/2.0)*np.sin(pitch/2.0)*np.cos(yaw/2.0) + np.sin(roll/2.0)*np.cos(pitch/2.0)*np.sin(yaw/2.0)
    d = np.cos(roll/2.0)*np.cos(pitch/2.0)*np.sin(yaw/2.0) - np.sin(roll/2.0)*np.sin(pitch/2.0)*np.cos(yaw/2.0)
    return np.array([a,b,c,d])


def quaternion_to_euler(quaternion):
    a = quaternion[0]
    b = quaternion[1]
    c = quaternion[2]
    d = quaternion[3]

    # TODO: complete the conversion
    # and return a numpy array of
    # 3 element representing the euler angles [roll, pitch, yaw]
    roll = np.arctan2(2.0*(a*b+c*d), (1.0 - 2.0*(b*b + c*c)))
    pitch = np.arcsin(2.0*(a*c - b*d))
    yaw = np.arctan2(2.0*(a*d+b*c), (1.0-2.0*(c*c-d*d)))
    return np.array([roll, pitch, yaw])


if __name__ == "__main__":
    euler = np.array([np.deg2rad(90), np.deg2rad(30), np.deg2rad(0)])

    q = euler_to_quaternion(euler)  # should be [ 0.683  0.683  0.183 -0.183]
    print(q)

    assert np.array_equal(euler, quaternion_to_euler(q))