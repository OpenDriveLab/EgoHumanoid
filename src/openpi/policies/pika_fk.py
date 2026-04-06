import math
import numpy as np
from typing_extensions import (
    Literal,
)

class C_PiperForwardKinematics():
    def __init__(self, dh_is_offset: Literal[0x00, 0x01] = 0x01):
        self.RADIAN = 180 / math.pi
        self.PI = math.pi
        # Denavit-Hartenberg parameters for each link
        # _a: link lengths
        # _alpha: link twists
        # _theta: joint angles
        # _d: link offsets
        self._a     = [0     , 0                      , 285.03                   , -21.98        , 0             , 0          ]
        self._alpha = [0     , -self.PI / 2           , 0                        , self.PI / 2   , -self.PI / 2  , self.PI / 2]
        self._theta = [0     , -self.PI * 174.22 / 180, -100.78 / 180 * self.PI  , 0             , 0             , 0          ]
        self._d     = [123   , 0                      , 0                        , 250.75        , 0             , 91         ]
        self.init_pos   = [55.0  , 0.0                    , 205.0                    , 0.0           , 85.0          , 0.0] # unit xyz-mm, rpy-degree
        # if j2, j3 offset 2°
        if(dh_is_offset == 0x01):
            self._a     = [0     , 0                      , 285.03                   , -21.98        , 0             , 0          ]
            self._alpha = [0     , -self.PI / 2           , 0                        , self.PI / 2   , -self.PI / 2  , self.PI / 2]
            self._theta = [0     , -self.PI * 172.22 / 180, -102.78 / 180 * self.PI  , 0             , 0             , 0          ]
            self._d     = [123   , 0                      , 0                        , 250.75        , 0             , 91         ]
            self.init_pos   = [56.128, 0.0                    , 213.266                  , 0.0           , 85.0          , 0.0] # unit xyz-mm, rpy-degree
    def __MatrixToeula(self, T):
        '''
        Convert a transformation matrix to Euler angles (roll, pitch, yaw).
        T: 4x4 transformation matrix
        '''
        Pos = [0.0] * 6
        # Extract position (x, y, z)
        Pos[0] = T[3]  # x position
        Pos[1] = T[7]  # y position
        Pos[2] = T[11] # z position
        # Calculate Euler angles (roll, pitch, yaw) based on rotation matrix
        if T[8] < -1 + 0.0001:
            Pos[4] = self.PI / 2 * self.RADIAN  # pitch (beta)
            Pos[5] = 0
            Pos[3] = math.atan2(T[1], T[5]) * self.RADIAN # roll (alpha)
        elif T[8] > 1 - 0.0001:
            Pos[4] = -self.PI / 2 * self.RADIAN # pitch (beta)
            Pos[5] = 0
            Pos[3] = -math.atan2(T[1], T[5]) * self.RADIAN # roll (alpha)
        else:
            # General case for Euler angles computation
            _bt = math.atan2(-T[8], math.sqrt(T[0] * T[0] + T[4] * T[4])) # pitch (beta)
            Pos[4] = _bt * self.RADIAN
            Pos[5] = math.atan2(T[4] / math.cos(_bt), T[0] / math.cos(_bt)) * self.RADIAN # yaw (gamma)
            Pos[3] = math.atan2(T[9] / math.cos(_bt), T[10] / math.cos(_bt)) * self.RADIAN # roll (alpha)

        return Pos

    def __MatMultiply(self, matrix1, matrix2, m, l, n):
        '''
        Multiply two matrices
        matrix1: first matrix
        matrix2: second matrix
        m: number of rows in matrix1
        l: number of columns in matrix1 (rows in matrix2)
        n: number of columns in matrix2
        '''
        matrixOut = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                tmp = 0.0
                for k in range(l):
                    tmp += matrix1[l * i + k] * matrix2[n * k + j]
                matrixOut[n * i + j] = tmp
        return matrixOut
    
    def __LinkTransformtion(self, alpha, a, theta, d):
        '''
        Compute the transformation matrix for a single link using the Denavit-Hartenberg parameters
        alpha: link twist
        a: link length
        theta: joint angle
        d: link offset
        '''
        # Precompute trigonometric functions for efficiency
        calpha = math.cos(alpha)
        salpha = math.sin(alpha)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        T = [0.0] * 16 # 4x4 transformation matrix
        T[0] = ctheta
        T[1] = -stheta
        T[2] = 0
        T[3] = a

        T[4] = stheta * calpha
        T[5] = ctheta * calpha
        T[6] = -salpha
        T[7] = -salpha * d

        T[8] = stheta * salpha
        T[9] = ctheta * salpha
        T[10] = calpha
        T[11] = calpha * d

        T[12] = 0
        T[13] = 0
        T[14] = 0
        T[15] = 1

        return T
    
    def CalFK(self, cur_j):
        '''
        Calculate Forward Kinematics for a given joint configuration
        cur_j: list of joint angles
        Returns the positions and Euler angles for each link
        '''
        # Initialize transformation matrices
        _Rt = [[0.0] * 16 for _ in range(6)]

        # Compute the individual transformation matrices
        for i in range(6):
            c_theta = cur_j[i] + self._theta[i]
            _Rt[i] = self.__LinkTransformtion(self._alpha[i], self._a[i], c_theta, self._d[i])

        # Multiply transformation matrices
        R02 = self.__MatMultiply(_Rt[0], _Rt[1], 4, 4, 4)
        R03 = self.__MatMultiply(R02, _Rt[2], 4, 4, 4)
        R04 = self.__MatMultiply(R03, _Rt[3], 4, 4, 4)
        R05 = self.__MatMultiply(R04, _Rt[4], 4, 4, 4)
        R06 = self.__MatMultiply(R05, _Rt[5], 4, 4, 4)

        # Extract Euler angles for each transformation
        j_pos = []
        j_pos.append(self.__MatrixToeula(_Rt[0]))   # Euler angles for link1
        j_pos.append(self.__MatrixToeula(R02))      # Euler angles for link2
        j_pos.append(self.__MatrixToeula(R03))      # Euler angles for link3
        j_pos.append(self.__MatrixToeula(R04))      # Euler angles for link4
        j_pos.append(self.__MatrixToeula(R05))      # Euler angles for link5
        j_pos.append(self.__MatrixToeula(R06))      # Euler angles for link6

        return j_pos


def qpos_to_eef_pos(qpos: np.ndarray) -> np.ndarray:
    """
    Convert 14-dimensional qpos to end-effector positions (6 DOF + 1 gripper for each arm).
    
    Args:
        qpos: numpy array of shape (14,) containing:
            - qpos[0:6]: left arm joint angles (6 DOF)
            - qpos[6]: left gripper position
            - qpos[7:13]: right arm joint angles (6 DOF) 
            - qpos[13]: right gripper position
    
    Returns:
        numpy array of shape (14,) containing:
            - eef_pos[0:6]: left end-effector pose (x, y, z, roll, pitch, yaw in mm/degrees)
            - eef_pos[6]: left gripper position (normalized 0-1)
            - eef_pos[7:13]: right end-effector pose (x, y, z, roll, pitch, yaw in mm/degrees)
            - eef_pos[13]: right gripper position (normalized 0-1)
    """
    if qpos.shape != (14,):
        raise ValueError(f"Expected qpos to have shape (14,), got {qpos.shape}")
    
    # Initialize forward kinematics for both arms
    fk_left = C_PiperForwardKinematics(dh_is_offset = 0x00)
    fk_right = C_PiperForwardKinematics(dh_is_offset = 0x00)
    
    # Extract joint angles for each arm
    left_joints = qpos[0:6]  # 6 DOF for left arm
    right_joints = qpos[7:13]  # 6 DOF for right arm
    
    # Extract gripper positions
    left_gripper = qpos[6]
    right_gripper = qpos[13]
    
    # Calculate forward kinematics for both arms
    left_eef_poses = fk_left.CalFK(left_joints)
    right_eef_poses = fk_right.CalFK(right_joints)
    
    # Get end-effector pose (last link, index 5)
    left_eef_pose = left_eef_poses[5]  # [x, y, z, roll, pitch, yaw]
    right_eef_pose = right_eef_poses[5]  # [x, y, z, roll, pitch, yaw]
    
    # Combine into final output
    eef_pos = np.zeros(14)
    eef_pos[0:6] = left_eef_pose  # left end-effector pose
    eef_pos[0:3] = eef_pos[0:3] / 1000
    eef_pos[3:6] = eef_pos[3:6] / 180 * np.pi  # convert roll, pitch, yaw to radians
    eef_pos[6] = left_gripper     # left gripper position
    eef_pos[7:13] = right_eef_pose  # right end-effector pose
    eef_pos[7:10] = eef_pos[7:10] / 1000
    eef_pos[10:13] = eef_pos[10:13] / 180 * np.pi  # convert roll, pitch, yaw to radians
    eef_pos[13] = right_gripper     # right gripper position

    
    return eef_pos


def batch_qpos_to_eef_pos(qpos_batch: np.ndarray) -> np.ndarray:
    """
    Convert a batch of 14-dimensional qpos to end-effector positions.
    
    Args:
        qpos_batch: numpy array of shape (batch_size, 14) or (batch_size, timesteps, 14)
    
    Returns:
        numpy array of same shape as input containing end-effector positions
    """
    if len(qpos_batch.shape) == 2:
        # Single batch dimension
        batch_size = qpos_batch.shape[0]
        eef_pos_batch = np.zeros_like(qpos_batch)
        
        for i in range(batch_size):
            eef_pos_batch[i] = qpos_to_eef_pos(qpos_batch[i])
            
    elif len(qpos_batch.shape) == 3:
        # Batch and time dimensions
        batch_size, timesteps = qpos_batch.shape[0], qpos_batch.shape[1]
        eef_pos_batch = np.zeros_like(qpos_batch)
        
        for i in range(batch_size):
            for t in range(timesteps):
                eef_pos_batch[i, t] = qpos_to_eef_pos(qpos_batch[i, t])

    elif len(qpos_batch.shape) == 1:
        # same as qpos_to_eef_pos
        eef_pos_batch = qpos_to_eef_pos(qpos_batch)
        
    else:
        raise ValueError(f"Expected qpos_batch to have 2 or 3 dimensions, got {len(qpos_batch.shape)}")
    
    return eef_pos_batch