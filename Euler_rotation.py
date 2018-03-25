# In this notebook you'll explore Euler rotations and get a feel for why the order of rotations matters.

# Euler rotations as we define them in this program are counterclockwise about the axes of the vehicle body frame, where:

# Roll -  ϕ  is about the x-axis
# Pitch - θ  is about the y-axis
# Yaw -   ψ  is about the z-axis
# As you'll see the same set of rotation transformations, applied in a different order can produce a very different
# final result!

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from enum import Enum

# %matplotlib inline
# plt.rcParams["figure.figsize"] = [12, 12]
# np.set_printoptions(precision=3, suppress=True)


class Rotation(Enum):
    ROLL = 0
    PITCH = 1
    YAW = 2


class EulerRotation:

    def __init__(self, rotations):
        """
        `rotations` is a list of 2-element tuples where the
        first element is the rotation kind and the second element
        is angle in degrees.

        Ex:
            [(Rotation.ROLL, 45), (Rotation.YAW, 32), (Rotation.PITCH, 55)]

        """
        self._rotations = rotations
        self._rotation_map = {Rotation.ROLL: self.roll, Rotation.PITCH: self.pitch, Rotation.YAW: self.yaw}

    def roll(self, phi):
        """Returns a rotation matrix along the roll axis"""
        rollMatrix = np.array([[1, 0,           0],
                                [0, np.cos(phi),  np.sin(phi)],
                                [0, -np.sin(phi),  np.cos(phi)]])
        return rollMatrix

    def pitch(self, theta):
        """Returns the rotation matrix along the pitch axis"""
        pitchMatrix = np.array([[np.cos(theta),  0, np.sin(theta)],
                                 [0,              1,  0],
                                 [-np.sin(theta), 0, np.cos(theta)]])
        return pitchMatrix

    def yaw(self, psi):
        """Returns the rotation matrix along the yaw axis"""
        yawMatrix = np.array([[np.cos(psi), np.sin(psi), 0],
                               [-np.sin(psi), np.cos(psi), 0],
                               [0, 0, 1]])
        return yawMatrix

    def rotate(self):
        """Applies the rotations in sequential order"""
        t = np.eye(3)
        for kind in self._rotations:
            ract = kind[0]
            degree = kind[1] * np.pi/180
            rmatrix = self._rotation_map[ract](degree)
            t = np.dot(t, rmatrix)
        return t


def test_rotation():
    # Test your code by passing in some rotation values
    rotations = [
        (Rotation.ROLL, 25),
        (Rotation.PITCH, 75),
        (Rotation.YAW, 90),
    ]

    R = EulerRotation(rotations).rotate()
    print('Rotation matrix ...')
    print(R)
    # Should print
    # Rotation matrix ...
    # [[ 0.    -0.259  0.966]
    # [ 0.906 -0.408 -0.109]
    # [ 0.423  0.875  0.235]]


def vis_rotation(roll=45, pitch=60, yaw=30):
    v = np.array([1, 0, 0])
    # TODO: calculate the new rotated versions of `v`.
    rotations = [
        (Rotation.ROLL, roll),
        (Rotation.PITCH, 0),
        (Rotation.YAW, 0),
    ]
    rv1 = EulerRotation(rotations).rotate()
    rotations = [
        (Rotation.ROLL, 0),
        (Rotation.PITCH, pitch),
        (Rotation.YAW, 0),
    ]
    rv2 = EulerRotation(rotations).rotate()
    rotations = [
        (Rotation.ROLL, 0),
        (Rotation.PITCH, 0),
        (Rotation.YAW, yaw),
    ]
    rv3 = EulerRotation(rotations).rotate()
    # rv = np.dot(R, v)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # axes (shown in black)
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='black', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='black', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='black', arrow_length_ratio=0.15)

    # Original Vector (shown in blue)
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color='blue', arrow_length_ratio=0.15)

    # Rotated Vectors (shown in red)
    ax.quiver(0, 0, 0, rv1[0], rv1[1], rv1[2], color='red', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, rv2[0], rv2[1], rv2[2], color='purple', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, rv3[0], rv3[1], rv3[2], color='green', arrow_length_ratio=0.15)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(1, -1)
    ax.set_zlim3d(1, -1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    test_rotation()
    vis_rotation()