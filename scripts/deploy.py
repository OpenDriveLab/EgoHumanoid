#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 Robot Inference Client for OpenPI Policy Server

This script connects to:
1. OpenPI policy server (websocket) for action inference
2. G1 robot via GR00T WBC framework
3. OpenCV camera input for visual perception

Usage:
1. Start the policy server:
   uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05-g1-1224 --policy.dir=xxx

2. Start this client:
   python main.py --host <server_ip> --port 8000

Controls:
  ] - Activate WBC policy / Exit silent mode
  p - Enter preparation phase (move to initial pose)
  c - Toggle left hand open/close (right hand stays open)
  l - Start/pause inference loop
  [ - Enter silent mode (slowly return to initial pose)
  o - Deactivate policy (emergency)
  Ctrl+C - Exit
"""
# TASK = "bed"
TASK = "toy"
# TASK = "cart"
# TASK = "garbage"

import signal
import sys
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro

# OpenPI client
from openpi_client import image_tools, websocket_client_policy
import sys
sys.path.insert(0, '/root/Projects/GR00T-WholeBodyControl')

# GR00T WBC imports
from decoupled_wbc.control.envs.g1.g1_env import G1Env
from decoupled_wbc.control.main.constants import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    JOINT_SAFETY_STATUS_TOPIC,
    LOWER_BODY_POLICY_STATUS_TOPIC,
    ROBOT_CONFIG_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardEStop,
    KeyboardListenerPublisher,
    ROSKeyboardDispatcher,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSServiceServer,
)
from decoupled_wbc.control.utils.telemetry import Telemetry


# ==================== Configuration ====================

@dataclass
class InferenceConfig(ControlLoopConfig):
    """Inference client configuration, inherits from ControlLoopConfig."""
    
    # OpenPI server settings
    host: str = "localhost"
    """Policy server host address"""
    
    port: int = 8000
    """Policy server port"""
    
    # Camera settings
    camera_url: str = "/dev/video0"
    """OpenCV camera source (device path, index, or stream URL)"""
    
    image_size: int = 224
    """Image size for model input"""
    
    # Inference settings
    prompt: str = "grasp the toy and put it on the table"
    """Task prompt for the policy"""
    
    chunk_size: int = 50
    """Action chunk size"""
    
    inference_rate: float = 3.0
    """Inference rate (Hz)"""
    
    gripper_threshold: float = 0.75
    """Gripper binarization threshold (values > threshold = 1, else = 0)"""
    
    # Temporal smoothing settings
    use_temporal_smoothing: bool = True
    """Enable temporal smoothing"""
    
    buffer_max_chunks: int = 10
    """Maximum chunks in buffer"""
    
    latency_k: int = 8
    """Latency compensation steps"""
    
    min_smooth_steps: int = 8
    """Minimum smoothing steps"""
    
    # Preparation settings
    preparation_duration: float = 3.0
    """Preparation phase duration (seconds)"""
    
    # Speed settings
    speed_scale: float = 1.0
    """Speed scale factor (0.5 = half speed, 1.0 = normal speed)"""
    
    # Robot settings (override from parent)
    interface: str = "real"
    """Interface type, 'real' for real robot"""
    
    require_confirmation: bool = True
    """Require confirmation before running on real robot"""


# ==================== Image Stream Client ====================

class MJPEGStreamClient:
    """OpenCV video capture client for getting camera images."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.cap = None
        self.last_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the image capture thread."""
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.stream_url}")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[Camera] Connected to {self.stream_url}")
        
    def _capture_loop(self):
        """Continuously capture frames."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Keep only the left half of the frame; drop the right half.
                frame = frame[:, : frame.shape[1] // 2]
                with self.lock:
                    self.last_frame = frame.copy()
            else:
                # Try to reconnect
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.stream_url)
                
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None
    
    def stop(self):
        """Stop the capture thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# ==================== Temporal Smoothing Buffer ====================

class StreamActionBuffer:
    """
    Buffer for temporal smoothing of action chunks.
    Implements linear blending between old and new action chunks.
    """
    
    def __init__(self, max_chunks: int = 10, state_dim: int = 16, gripper_threshold: float = 0.5):
        self.max_chunks = max_chunks
        self.state_dim = state_dim
        self.gripper_threshold = gripper_threshold  # Threshold for gripper binarization
        self.lock = threading.Lock()
        
        # Current action sequence (smoothed)
        self.cur_chunk = deque()
        self.k = 0  # Published step count
        self.last_action = None
    
    def _binarize_gripper_chunk(self, actions_chunk: np.ndarray) -> np.ndarray:
        """
        Binarize gripper values in a chunk and ensure coherent transitions.
        
        Gripper values are expected at indices [16:18] for left and right hands.
        Each gripper value is binarized based on a threshold, then enforced to have
        at most one transition per chunk. The transition point is chosen to minimize
        the difference from the original continuous values.
        
        Args:
            actions_chunk: Action chunk [chunk_size, state_dim]
                           Expected format: [delta_eef(12), delta_height(1), 
                                           navigate(3), hand_status(2)]
        
        Returns:
            Modified action chunk with binarized and coherent gripper values
        """
        if actions_chunk.ndim != 2:
            return actions_chunk
        
        chunk_size = len(actions_chunk)
        state_dim = actions_chunk.shape[1]
        
        # Check if gripper values exist
        if state_dim < 18:
            print(f"[Warning] state_dim={state_dim} < 18, gripper values may not exist")
            return actions_chunk
        
        actions_chunk = actions_chunk.copy()
        
        # Extract gripper values [chunk_size, 2]
        gripper_values = actions_chunk[:, 16:18].copy()
        
        # Process each hand separately to ensure coherent transitions
        for hand_idx in range(2):
            original_values = gripper_values[:, hand_idx]
            
            if chunk_size == 1:
                # Single step, simple binarization
                binary_val = 1.0 if original_values[0] > self.gripper_threshold else 0.0
                gripper_values[:, hand_idx] = binary_val
                continue
            
            # Try all possible transition points and find the one with minimum error
            # Also try no transition (all 0 or all 1)
            best_error = float('inf')
            best_pattern = None
            best_transition_idx = None
            best_transition_type = None  # 'none', '0to1', '1to0'
            
            # Option 1: All 0
            pattern_all_0 = np.zeros(chunk_size)
            error_all_0 = np.sum((pattern_all_0 - original_values) ** 2)
            if error_all_0 < best_error:
                best_error = error_all_0
                best_pattern = pattern_all_0
                best_transition_idx = None
                best_transition_type = 'none'
            
            # Option 2: All 1
            pattern_all_1 = np.ones(chunk_size)
            error_all_1 = np.sum((pattern_all_1 - original_values) ** 2)
            if error_all_1 < best_error:
                best_error = error_all_1
                best_pattern = pattern_all_1
                best_transition_idx = None
                best_transition_type = 'none'
            
            # Option 3: Transition from 0 to 1 at index i (i=1 to chunk_size-1)
            for transition_idx in range(1, chunk_size):
                pattern = np.zeros(chunk_size)
                pattern[transition_idx:] = 1.0
                error = np.sum((pattern - original_values) ** 2)
                if error < best_error:
                    best_error = error
                    best_pattern = pattern
                    best_transition_idx = transition_idx
                    best_transition_type = '0to1'
            
            # Option 4: Transition from 1 to 0 at index i (i=1 to chunk_size-1)
            for transition_idx in range(1, chunk_size):
                pattern = np.ones(chunk_size)
                pattern[transition_idx:] = 0.0
                error = np.sum((pattern - original_values) ** 2)
                if error < best_error:
                    best_error = error
                    best_pattern = pattern
                    best_transition_idx = transition_idx
                    best_transition_type = '1to0'
            
            # Apply transition delay: increase transition index by 10
            if best_transition_idx is not None:
                delayed_transition_idx = min(best_transition_idx + 0, chunk_size - 1)
                if delayed_transition_idx != best_transition_idx:
                    # Rebuild pattern with delayed transition
                    if best_transition_type == '0to1':
                        best_pattern = np.zeros(chunk_size)
                        best_pattern[delayed_transition_idx:] = 1.0
                    elif best_transition_type == '1to0':
                        best_pattern = np.ones(chunk_size)
                        best_pattern[delayed_transition_idx:] = 0.0
            
            # Apply the best pattern
            gripper_values[:, hand_idx] = best_pattern
        
        # Write back to actions_chunk
        actions_chunk[:, 16:18] = gripper_values
        
        return actions_chunk
        
    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        """
        Integrate a new inference chunk with temporal smoothing.
        
        Args:
            actions_chunk: New action chunk [chunk_size, state_dim]
            max_k: Maximum latency compensation steps
            min_m: Minimum smoothing steps
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            actions_chunk = actions_chunk[:30]
            
            
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            
            # Latency compensation: drop first k actions
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]
            
            # Build old sequence
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            
            new_list = list(new_chunk)
            
            # Compute overlap
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return
            
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)
            
            # Linear blending weights
            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old
            
            # Blend overlapping region
            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            
            # Concatenate with remaining new actions
            combined = smoothed + new_list[overlap_len:]
            # Binarize and ensure coherent gripper transitions
            combined = list(self._binarize_gripper_chunk(np.array(combined)))
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0
    
    def pop_next_action(self) -> Optional[np.ndarray]:
        """Pop and return the next action to execute."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act
    
    def has_any(self) -> bool:
        """Check if buffer has actions."""
        with self.lock:
            return len(self.cur_chunk) > 0
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.cur_chunk.clear()
            self.last_action = None
            self.k = 0
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.cur_chunk)


# ==================== Hand Controller ====================

class HandController:
    """
    Controller that converts hand_status (0=open, 1=close) to hand joint angles.
    Based on G1GripperInverseKinematicsSolver logic.
    """
    
    # Hand joint order in upper_body: index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2
    # IK solver output order: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
    
    def __init__(self, robot_model):
        self.robot_model = robot_model
        
        # Get hand joint indices within upper_body
        upper_body_indices = robot_model.get_joint_group_indices("upper_body")
        left_hand_indices = robot_model.get_joint_group_indices("left_hand")
        right_hand_indices = robot_model.get_joint_group_indices("right_hand")
        
        # Convert to relative indices within upper_body
        upper_body_set = set(upper_body_indices)
        self.left_hand_in_upper = [
            list(upper_body_indices).index(i) for i in left_hand_indices if i in upper_body_set
        ]
        self.right_hand_in_upper = [
            list(upper_body_indices).index(i) for i in right_hand_indices if i in upper_body_set
        ]
        
        # Precompute open and close joint positions
        self.left_open_q = np.zeros(7)
        self.right_open_q = np.zeros(7)
        self.left_close_q = self._get_close_q("left")
        self.right_close_q = self._get_close_q("right")
        
        # Current hand state for smooth transitions
        self.current_left_q = self.left_open_q.copy()
        self.current_right_q = self.right_open_q.copy()
        
    def _get_close_q(self, side: str) -> np.ndarray:
        """
        Get closed hand joint angles (similar to index_close gesture).
        Joint order: index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2
        """
        q = np.zeros(7)
        
        # Thumb joints (indices 4, 5, 6)
        amp0 = 0.5
        if side == "left":
            q[4] = -amp0  # thumb_0
            q[5] = 0.7  # thumb_1
        else:
            q[4] = -amp0  # thumb_0
            q[5] = -0.7  # thumb_1
        
        # q[5] = 0.7  # thumb_1
        q[6] = 0.7  # thumb_2
        
        # Index and middle finger joints (indices 0, 1, 2, 3)
        ampA = 1.5
        ampB = 1.5
        
        if side == "left":
            q[0] = -ampA  # index_0
            q[1] = -ampB  # index_1
            q[2] = -ampA  # middle_0
            q[3] = -ampB  # middle_1
        else:
            q[0] = ampA  # index_0
            q[1] = ampB  # index_1
            q[2] = ampA  # middle_0
            q[3] = ampB  # middle_1
        
        return q
    
    def get_hand_joints(self, hand_status: np.ndarray) -> tuple:
        """
        Get hand joint angles based on hand_status.
        
        Args:
            hand_status: 0.0 = open, 1.0 = close (can be interpolated)
        
        Returns:
            (left_hand_q, right_hand_q): tuple of 7-element arrays
        """
        # Clamp hand_status to [0, 1]
        hand_status = np.clip(hand_status, 0.0, 1.0)

        # if hand_status[0] <=0.5:
        #     hand_status[0] = 1.0
        # else:
        #     hand_status[0] = 0.0
        # if hand_status[1] <=0.5:
        #     hand_status[1] = 1.0
        # else:
        #     hand_status[1] = 0.0
        
        # Interpolate between open and close
        left_q = self.left_open_q * (1 - hand_status[0]) + self.left_close_q * hand_status[0]
        right_q = self.right_open_q * (1 - hand_status[1]) + self.right_close_q * hand_status[1]
        
        return left_q, right_q
    
    def apply_hand_joints_to_upper_body(
        self, 
        upper_body_pose: np.ndarray, 
        left_hand_q: np.ndarray, 
        right_hand_q: np.ndarray
    ) -> np.ndarray:
        """
        Apply hand joint angles to upper body pose array.
        
        Args:
            upper_body_pose: Full upper body joint angles
            left_hand_q: Left hand joint angles (7,)
            right_hand_q: Right hand joint angles (7,)
        
        Returns:
            Modified upper body pose with hand joints
        """
        result = upper_body_pose.copy()
        
        if len(self.left_hand_in_upper) == 7:
            result[self.left_hand_in_upper] = left_hand_q
        if len(self.right_hand_in_upper) == 7:
            result[self.right_hand_in_upper] = right_hand_q
        
        return result


# ==================== Delta EEF Controller ====================

class DeltaEEFController:
    """
    Controller that applies delta EEF actions using IK.
    Converts delta_eef + base commands to upper body joint commands.
    """
    
    def __init__(self, robot_model):
        self.robot_model = robot_model
        
        # Initialize IK solver
        left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
        self.retargeting_ik = TeleopRetargetingIK(
            robot_model=robot_model,
            left_hand_ik_solver=left_hand_ik_solver,
            right_hand_ik_solver=right_hand_ik_solver,
            enable_visualization=False,
            body_active_joint_groups=["upper_body"],
        )
        
        # Get wrist frame names
        self.left_wrist_frame = robot_model.supplemental_info.hand_frame_names["left"]
        self.right_wrist_frame = robot_model.supplemental_info.hand_frame_names["right"]
        
        # Current EEF state
        self.current_eef = None
    
    def _delta_to_transform_matrix(self, delta_xyzrpy: np.ndarray) -> np.ndarray:
        """
        Convert delta xyzrpy to transformation matrix.
        
        Args:
            delta_xyzrpy: (6,) [dx, dy, dz, roll, pitch, yaw]
        
        Returns:
            (4, 4) transformation matrix
        """
        T = np.eye(4)
        T[:3, 3] = delta_xyzrpy[:3]  # translation
        T[:3, :3] = R.from_euler('xyz', delta_xyzrpy[3:6]).as_matrix()  # rotation
        return T
    
    def _eef_to_transformation_matrix(self, eef_data: np.ndarray) -> dict:
        """
        Convert EEF data to transformation matrix dict.
        
        Args:
            eef_data: (14,) [left: x,y,z,qw,qx,qy,qz, right: x,y,z,qw,qx,qy,qz]
                     Note: quaternion format is scalar_first=True, i.e. (w, x, y, z)
        
        Returns:
            dict: {left_wrist_frame: T_left, right_wrist_frame: T_right}
        """
        # Extract left and right hand data
        left_pos = eef_data[:3]
        left_quat_scalar_first = eef_data[3:7]  # (qw, qx, qy, qz) - scalar_first format
        right_pos = eef_data[7:10]
        right_quat_scalar_first = eef_data[10:14]  # (qw, qx, qy, qz) - scalar_first format
        
        # Convert to scipy expected format (x, y, z, w)
        # scalar_first (w, x, y, z) -> vector_first (x, y, z, w)
        left_quat = np.array([left_quat_scalar_first[1], left_quat_scalar_first[2], 
                             left_quat_scalar_first[3], left_quat_scalar_first[0]])
        right_quat = np.array([right_quat_scalar_first[1], right_quat_scalar_first[2], 
                              right_quat_scalar_first[3], right_quat_scalar_first[0]])
        
        # Convert to transformation matrix
        T_left = np.eye(4)
        T_left[:3, 3] = left_pos
        T_left[:3, :3] = R.from_quat(left_quat).as_matrix()
        
        T_right = np.eye(4)
        T_right[:3, 3] = right_pos
        T_right[:3, :3] = R.from_quat(right_quat).as_matrix()
        
        return {
            self.left_wrist_frame: T_left,
            self.right_wrist_frame: T_right,
        }
    
    def _transform_matrix_to_eef(self, T_left: np.ndarray, T_right: np.ndarray) -> np.ndarray:
        """
        Convert transformation matrices to EEF data format.
        
        Args:
            T_left: (4, 4) left hand transformation matrix
            T_right: (4, 4) right hand transformation matrix
        
        Returns:
            (14,) [left: x,y,z,qw,qx,qy,qz, right: x,y,z,qw,qx,qy,qz]
        """
        # Extract position and rotation
        left_pos = T_left[:3, 3]
        left_quat_vector_first = R.from_matrix(T_left[:3, :3]).as_quat()  # (x, y, z, w)
        left_quat_scalar_first = np.array([left_quat_vector_first[3], left_quat_vector_first[0],
                                          left_quat_vector_first[1], left_quat_vector_first[2]])  # (w, x, y, z)
        
        right_pos = T_right[:3, 3]
        right_quat_vector_first = R.from_matrix(T_right[:3, :3]).as_quat()  # (x, y, z, w)
        right_quat_scalar_first = np.array([right_quat_vector_first[3], right_quat_vector_first[0],
                                            right_quat_vector_first[1], right_quat_vector_first[2]])  # (w, x, y, z)
        
        return np.concatenate([
            left_pos, left_quat_scalar_first,
            right_pos, right_quat_scalar_first
        ])
    
    def _apply_delta_to_eef(self, eef_data: np.ndarray, delta_eef: np.ndarray) -> np.ndarray:
        """
        Apply delta_eef to eef_data.
        
        Args:
            eef_data: (14,) current EEF pose
            delta_eef: (12,) [left: dx,dy,dz,roll,pitch,yaw, right: dx,dy,dz,roll,pitch,yaw]
        
        Returns:
            (14,) new EEF pose after applying delta
        """
        # Convert current EEF to transformation matrix
        T_dict = self._eef_to_transformation_matrix(eef_data)
        T_left_curr = T_dict[self.left_wrist_frame]
        T_right_curr = T_dict[self.right_wrist_frame]
        
        # Extract left and right delta
        left_delta = delta_eef[:6]  # [dx, dy, dz, roll, pitch, yaw]
        right_delta = delta_eef[6:12]  # [dx, dy, dz, roll, pitch, yaw]
        
        # Convert delta to transformation matrix
        delta_T_left = self._delta_to_transform_matrix(left_delta)
        delta_T_right = self._delta_to_transform_matrix(right_delta)
        
        # Apply delta: T_new = T_curr @ delta_T
        T_left_new = T_left_curr @ delta_T_left
        T_right_new = T_right_curr @ delta_T_right
        
        # Convert back to EEF format
        return self._transform_matrix_to_eef(T_left_new, T_right_new)
    
    def _compute_upper_body_joints_from_eef(self, eef_data: np.ndarray) -> np.ndarray:
        """
        Compute upper body joint angles from EEF pose.
        
        Uses the same approach as TeleopPolicy: set_goal() + get_action() for state continuity.
        
        Args:
            eef_data: (14,) EEF pose
        
        Returns:
            (N,) upper body joint angles
        """
        # Convert to transformation matrix
        # if TASK == "toy":
        #     if eef_data[0] < 0.15:
        #         eef_data[0] = 0.15
        #     if eef_data[7] < 0.15:
        #         eef_data[7] = 0.15
            
        #     if eef_data[1] < 0.05:
        #         eef_data[1] = 0.05
        #     if eef_data[8] > -0.05:
        #         eef_data[8] = -0.05

        #     if abs(eef_data[2] - eef_data[9]) > 0.05 and eef_data[2] - eef_data[9] > 0.0:
        #         eef_data[2] = eef_data[2] - 0.01
        #         eef_data[9] = eef_data[9] + 0.01
        #     if abs(eef_data[2] - eef_data[9]) > 0.05 and eef_data[2] - eef_data[9] < 0.0:
        #         eef_data[2] = eef_data[2] + 0.01
        #         eef_data[9] = eef_data[9] - 0.01
        
        # print(f"EEF data: {eef_data}")
        body_data = self._eef_to_transformation_matrix(eef_data)
        
        # Use same approach as TeleopPolicy: set_goal() + get_action()
        ik_data = {
            "body_data": body_data,
            "left_hand_data": None,  # No hand IK
            "right_hand_data": None,  # No hand IK
        }
        self.retargeting_ik.set_goal(ik_data)
        target_joints = np.array(self.retargeting_ik.get_action())
        
        return target_joints
    
    def reset_eef(self, initial_eef: np.ndarray):
        """Reset current EEF state and IK solver."""
        self.current_eef = initial_eef.copy()
        # Reset IK solver internal state
        self.retargeting_ik.reset()
        # Warmup IK with initial pose
        _ = self.compute_upper_body_joints(self.current_eef)
    
    def apply_delta_eef(self, delta_eef: np.ndarray) -> np.ndarray:
        """
        Apply delta EEF to current state and return new EEF.
        
        Args:
            delta_eef: [12] - left: dx,dy,dz,roll,pitch,yaw, right: dx,dy,dz,roll,pitch,yaw
        
        Returns:
            New EEF state [14]
        """
        if self.current_eef is None:
            raise RuntimeError("EEF not initialized. Call reset_eef first.")
        
        self.current_eef = self._apply_delta_to_eef(self.current_eef, delta_eef)
        return self.current_eef
    
    def compute_upper_body_joints(self, eef_data: np.ndarray) -> np.ndarray:
        """Compute upper body joint angles from EEF pose using IK."""
        return self._compute_upper_body_joints_from_eef(eef_data)


# ==================== Inference Controller ====================

class InferenceController:
    """
    Main controller that handles:
    - Keyboard input for state transitions
    - Inference loop with the policy server
    - Action execution on the robot
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        robot_model,
        camera_client: MJPEGStreamClient,
        policy_client: websocket_client_policy.WebsocketClientPolicy,
    ):
        self.config = config
        self.robot_model = robot_model
        self.camera_client = camera_client
        self.policy_client = policy_client
        
        # Controllers
        self.delta_eef_controller = DeltaEEFController(robot_model)
        self.hand_controller = HandController(robot_model)
        
        # Initial EEF pose (from first frame)
        # Left hand: pos [0.1924, 0.1499, 0.0796], quat [0.9990, 0.0417, 0.0124, 0.0084] (qw,qx,qy,qz)
        # Right hand: pos [0.1839, -0.1760, 0.0886], quat [0.9983, -0.0160, 0.0138, -0.0540] (qw,qx,qy,qz)
        


        if TASK == "garbage":
            self.initial_eef = np.array([
                # 0.1324, 0.1499, 0.13,  # left position
                0.1924, 0.1499, 0.13,  # left position
                0.9990, 0.0417, -0.1624, 0.0084,  # left quaternion (qw,qx,qy,qz)
                # 0.1239, -0.1760, 0.13,  # right position
                0.1839, -0.1760, 0.13,  # right position
                0.9983, -0.0160, -0.1638, -0.0540,  # right quaternion (qw,qx,qy,qz)
            ])
        elif TASK == "cart":
            self.initial_eef = np.array([
                0.26203959, 0.13982252, 0.22464995,  # left position
                0.72998835, 0.58931441, -0.29015691, 0.18877102,  # left quaternion (qw,qx,qy,qz)
                0.25401768, -0.12831931, 0.22786569,  # right position
                0.73122155, -0.56956442, -0.33943402, -0.16029963,  # right quaternion (qw,qx,qy,qz)
            ])
        # elif TASK == "cart":
        #     self.initial_eef = np.array([
        #         0.19101273,  0.13912974,  0.22924709,  # left position
        #         0.76827627,  0.61089334, -0.17724066,  0.07174011,  # left quaternion (qw,qx,qy,qz)
        #         0.19453715, -0.13912033,  0.23318919,  # right position
        #         0.71888301, -0.6426301 , -0.25615103, -0.06797372,  # right quaternion (qw,qx,qy,qz)
        #     ])
        # elif TASK == "cart":
        #     self.initial_eef = np.array([
        #         0.19101273,  0.13912974,  0.25924709,  # left position
        #         0.76827627,  0.61089334, -0.17724066,  0.07174011,  # left quaternion (qw,qx,qy,qz)
        #         0.19453715, -0.13912033,  0.24318919,  # right position
        #         0.71888301, -0.6426301 , -0.25615103, -0.06797372,  # right quaternion (qw,qx,qy,qz)
        #     ])
        else:
            self.initial_eef = np.array([
                # 0.1924, 0.1499, 0.1,  # left position
                0.2224, 0.1499, 0.155,  # left position
                1, 0.0, -0.2, 0.00,  # left quaternion (qw,qx,qy,qz)
                # 0.1839, -0.1760, 0.1,  # right position
                0.2139, -0.1760, 0.155,  # right position
                1, 0.0, -0.2, -0.00,  # right quaternion (qw,qx,qy,qz)
            ])

        ## toy step 2
        # self.initial_eef = np.array([ 0.16400074,  0.33442484,  0.106349  ,  0.97269609, -0.01624171,
        # 0.00593432,  0.23143746,  0.18256415, -0.29837135,  0.11267524,
        # 0.9798113 , -0.01048086, -0.01416838, -0.19914622])

    #     ## toy  step 3 4
    #     self.initial_eef = np.array([ 0.28991667,  0.05686246,  0.1832942 ,  0.98784922,  0.00666122,
    #    -0.05741888, -0.14426577,  0.28374521, -0.09727426,  0.23295578,
    #     0.99630355, -0.02438568, -0.06667285,  0.04836642])

        ## cart step 3
    #    self.initial_eef = np.array([ 0.34831205,  0.25793398,  0.375012  ,  0.57634596,  0.6712966 ,
    #    -0.31798139,  0.34069642,  0.31526737, -0.1425088 ,  0.2self.initial_eef1015122,
    #        0.81751868, -0.51930896, -0.22841931, -0.0990254 ])

        ## cart step 4
    #    self.initial_eef = np.array([ 0.40475031,  0.13653816,  0.34919792,  0.55300517,  0.595414  ,
    #         -0.51268846,  0.27716059,  0.31991547, -0.13204646,  0.18561096,
    #         0.81997627, -0.51189417, -0.23822185, -0.09409365])

        
        # State
        self.is_running = False
        self.is_preparing = False
        self.preparation_complete = False
        self.vx_zero_detected = False  # Flag: once vx becomes 0, stop forcing hand_status
        self.preparation_start_time = None
        self.preparation_start_upper_body = None
        self.preparation_target_upper_body = None
        self.preparation_target_wrist = None
        
        # Action buffer for temporal smoothing
        self.stream_buffer = StreamActionBuffer(
            max_chunks=config.buffer_max_chunks,
            state_dim=18,  # 12 delta_eef + 1 delta_height + 3 navigate + 2 hand_status
            gripper_threshold=config.gripper_threshold,
        )
        
        # Inference thread
        self.inference_thread = None
        self.shutdown_event = threading.Event()

        # to align control frequency and data frequency, we need to repeat the actions
        # alternates between 2 and 3 repeats
        self.command_buffer = deque()
        self.repeat_count = 2  # starts with 2, alternates: 2, 3, 2, 3, ...
        
        # Silent mode state (slowly interpolate back to initial position)
        self.silent_mode = False
        self.silent_mode_start_time = None
        self.silent_mode_start_upper_body = None
        self.silent_mode_target_upper_body = None
        self.silent_mode_duration = 5.0  # seconds for interpolation (slow and safe)
        self.silent_mode_interpolation_complete = False
        
        # Save last upper body joints for silent mode interpolation
        self.last_upper_body_joints = None
        
        # Gripper state (for manual control before inference)
        self.gripper_closed = False  # Start with open gripper
        self.gripper_current = 0.0   # Current gripper value (0=open, 1=closed), smoothly interpolated
        self.gripper_speed = 0.02    # Gripper change per control cycle (smooth transition)
        
        # Height state (accumulate delta_height)
        self.current_height = DEFAULT_BASE_HEIGHT
        
        # Hand status monitoring for automatic stop
        self.prev_hand_status = None  # Previous hand_status value
        self.hand_status_zero_count = 0  # Count of consecutive zeros after transition from 1->0
        self.hand_status_zero_threshold = 40000
        if TASK == "toy":
            self.hand_status_zero_threshold = 40  # Threshold to trigger stop
        if TASK == "bed":
            self.hand_status_zero_threshold = 100
        self.hand_status_seen_closed = False  # Flag: have we seen hand_status=1 at least once?
        self.prev_left_hand_status = None  # 上一帧左手状态，用于检测闭合→打开
        self.left_hand_open_transition_count = 0  # 左手闭合→打开已遇到的次数，第二次才启动延迟
        self.left_hand_open_delay_count = 0  # 左手闭合→打开后，强制输出1的剩余次数
        self.original_initial_eef = None  # Backup of original initial_eef (for bed task)
        
    def handle_keyboard_button(self, key):
        """Handle keyboard input."""
        if key == "p":
            if not self.is_preparing and not self.is_running:
                self.is_preparing = True
                self.preparation_complete = False
                self.preparation_start_time = None
                self.preparation_start_upper_body = None
                self.preparation_target_upper_body = None
                self.preparation_target_wrist = None
                print("\n[Controller] 🔄 Entering preparation phase...")
                
        elif key == "l":
            if self.is_preparing:
                print("\n[Controller] ⚠️ Preparation not complete, please wait")
                return
            
            if not self.preparation_complete and not self.is_running:
                print("\n[Controller] ⚠️ Press 'p' first to complete preparation")
                return
            
            self.is_running = not self.is_running
            if self.is_running:
                self.preparation_complete = False
                self.vx_zero_detected = False  # Reset vx detection flag when starting
                # Reset hand status monitoring
                self.prev_hand_status = None
                self.hand_status_zero_count = 0
                self.hand_status_seen_closed = False
                self.prev_left_hand_status = None
                self.left_hand_open_transition_count = 0
                self.left_hand_open_delay_count = 0
                # Exit silent mode if active
                if self.silent_mode:
                    self.exit_silent_mode()
                # Reset state before starting
                self.stream_buffer.clear()
                self.command_buffer.clear()
                self.delta_eef_controller.reset_eef(self.initial_eef)
                self.repeat_count = 2
                self.current_height = DEFAULT_BASE_HEIGHT  # Reset height
                print("\n[Controller] ▶ Starting inference (state reset)")
                self._start_inference_thread()
            else:
                # Reset hand status monitoring when pausing
                self.prev_hand_status = None
                self.hand_status_zero_count = 0
                self.hand_status_seen_closed = False
                self.prev_left_hand_status = None
                self.left_hand_open_transition_count = 0
                self.left_hand_open_delay_count = 0
                print("\n[Controller] ⏸ Pausing inference")
                
        elif key == "[":
            # Enter silent mode
            if not self.silent_mode:
                self.enter_silent_mode()
                print("\n[Controller] 🔇 Entering silent mode (press ']' to exit)")
                
        elif key == "]":
            # Exit silent mode
            if self.silent_mode:
                self.exit_silent_mode()
                print("\n[Controller] 🔊 Exiting silent mode")
                
        elif key == "c":
            # Toggle left hand only (right hand保持打开)
            self.gripper_closed = not self.gripper_closed
            state = "闭合" if self.gripper_closed else "打开"
            print(f"\n[Controller] ✋ 左手: {state}")
                
        elif key == "o":
            # Emergency stop
            self.is_running = False
            self.is_preparing = False
            self.preparation_complete = False
            self.silent_mode = False
            self.stream_buffer.clear()
            self.command_buffer.clear()
            # Reset hand status monitoring
            self.prev_hand_status = None
            self.hand_status_zero_count = 0
            self.hand_status_seen_closed = False
            self.prev_left_hand_status = None
            self.left_hand_open_transition_count = 0
            self.left_hand_open_delay_count = 0
            print("\n[Controller] ⏹ Policy deactivated (emergency stop)")
                
    def enter_silent_mode(self):
        """Enter silent mode - slowly interpolate back to initial position."""
        self.silent_mode = True
        self.is_running = False
        self.preparation_complete = False  # Clear preparation state
        self.stream_buffer.clear()
        self.command_buffer.clear()
        # Reset hand status monitoring
        self.prev_hand_status = None
        self.hand_status_zero_count = 0
        self.hand_status_seen_closed = False
        self.prev_left_hand_status = None
        self.left_hand_open_transition_count = 0
        self.left_hand_open_delay_count = 0
        
        # Set interpolation state
        self.silent_mode_start_time = time.time()
        
        # Use saved upper body joints as start position
        if self.last_upper_body_joints is not None:
            self.silent_mode_start_upper_body = self.last_upper_body_joints.copy()
        else:
            # Fallback: compute from initial EEF
            self.delta_eef_controller.reset_eef(self.initial_eef)
            self.silent_mode_start_upper_body = self.delta_eef_controller.compute_upper_body_joints(self.initial_eef)
        
        # Reset IK and compute target upper body joints from initial EEF
        self.delta_eef_controller.reset_eef(self.initial_eef)
        self.silent_mode_target_upper_body = self.delta_eef_controller.compute_upper_body_joints(self.initial_eef)
        self.silent_mode_interpolation_complete = False
        
    def exit_silent_mode(self):
        """Exit silent mode."""
        self.silent_mode = False
        self.preparation_complete = False  # Clear preparation state to require new 'p' press
        self.stream_buffer.clear()
        self.command_buffer.clear()
        # Reset EEF and height state
        self.delta_eef_controller.reset_eef(self.initial_eef)
        self.repeat_count = 2
        self.current_height = DEFAULT_BASE_HEIGHT
        
    def is_silent_mode(self) -> bool:
        """Check if in silent mode."""
        return self.silent_mode
                
    def _start_inference_thread(self):
        """Start the inference thread."""
        if self.inference_thread is not None and self.inference_thread.is_alive():
            return
        
        self.shutdown_event.clear()
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
        )
        self.inference_thread.start()
        
    def _inference_loop(self):
        """Non-blocking inference loop running in a separate thread."""
        rate_interval = 1.0 / self.config.inference_rate
        
        while not self.shutdown_event.is_set() and self.is_running:
            try:
                t_start = time.time()
                
                # Get camera frame
                frame = self.camera_client.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Preprocess image: BGR -> RGB, resize
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resized = image_tools.resize_with_pad(
                    np.array([img_rgb]),
                    self.config.image_size,
                    self.config.image_size,
                )[0]
                
                # Build payload
                # State is set to zeros (mask_state=True in training config)
                payload = {
                    "images": {
                        "head_left": img_resized.transpose(2, 0, 1),  # CHW format
                    },
                    "prompt": self.config.prompt,
                }
                
                # Infer
                t_infer = time.time()
                result = self.policy_client.infer(payload)
                actions = result.get("actions", None).copy()
                
                # Force hand_status to 0 while vx > threshold, until vx becomes 0
                if actions is not None and not self.vx_zero_detected:
                    vx_values = actions[:, 13]  # navigate vx is at index 13
                    vx_threshold = 0.05  # threshold to consider vx as "zero"
                    if np.mean(np.abs(vx_values)) <= vx_threshold:
                        # vx is zero, stop forcing hand_status from now on
                        self.vx_zero_detected = True
                    else:
                        # vx > threshold, force hand_status to 0 (open)
                        actions = actions.copy()  # Make writable copy
                        # print(f"Force hand_status to 0 (open) because vx > threshold: {np.mean(np.abs(vx_values))}")
                        # actions[:, 16:] = 0.0
                
                print(f"[Inference] Time: {(time.time() - t_infer)*1000:.1f}ms")
                
                # Push to buffer
                if actions is not None and len(actions) > 0:
                    self.stream_buffer.integrate_new_chunk(
                        actions,
                        max_k=self.config.latency_k,
                        min_m=self.config.min_smooth_steps,
                    )
                
                # Sleep to maintain rate
                elapsed = time.time() - t_start
                if elapsed < rate_interval:
                    time.sleep(rate_interval - elapsed)
                    
            except Exception as e:
                print(f"[Inference] Error: {e}")
                time.sleep(0.1)
                
    def get_command(self, current_obs: dict) -> Optional[dict]:
        """
        Get the next command to send to the robot.
        
        Returns wbc_goal dict or None.
        """
        # Save current upper body joints for silent mode interpolation
        if current_obs is not None and "q" in current_obs:
            upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")
            self.last_upper_body_joints = current_obs["q"][upper_body_indices].copy()
        
        # Preparation phase: slowly move to initial pose
        if self.is_preparing:
            return self._handle_preparation(current_obs)
        
        # Silent mode: slowly interpolate back to initial position
        if self.silent_mode:
            return self._handle_silent_mode(current_obs)
        
        # Waiting for start after preparation
        if self.preparation_complete and not self.is_running:
            if self.preparation_target_upper_body is not None:
                # Smoothly interpolate gripper to target state
                target_gripper = 1.0 if self.gripper_closed else 0.0
                if self.gripper_current < target_gripper:
                    self.gripper_current = min(self.gripper_current + self.gripper_speed, target_gripper)
                elif self.gripper_current > target_gripper:
                    self.gripper_current = max(self.gripper_current - self.gripper_speed, target_gripper)
                
                # Apply smoothed gripper state
                # 仅左手受 'c' 控制，右手保持打开
                left_hand_q, right_hand_q = self.hand_controller.get_hand_joints(
                    np.array([self.gripper_current, 0.0])
                )
                upper_body = self.hand_controller.apply_hand_joints_to_upper_body(
                    self.preparation_target_upper_body.copy(), left_hand_q, right_hand_q
                )
                return {
                    "target_upper_body_pose": upper_body,
                    "wrist_pose": self.preparation_target_wrist,
                    "navigate_cmd": np.array(DEFAULT_NAV_CMD),
                    "base_height_command": DEFAULT_BASE_HEIGHT,
                }
            return None
        
        # Running: get action from buffer and apply
        if not self.is_running:
            return None
        
        # If we have commands in buffer, use them first
        if len(self.command_buffer) > 0:
            return self.command_buffer.popleft()
        
        # Get new action from inference buffer
        action = self.stream_buffer.pop_next_action()
        if action is None:
            # Buffer is empty (waiting for first inference result)
            # Keep current hand state instead of returning None
            if self.preparation_target_upper_body is not None:
                # 仅左手受 'c' 控制，右手保持打开
                target_gripper = 1.0 if self.gripper_closed else 0.0
                left_hand_q, right_hand_q = self.hand_controller.get_hand_joints(
                    np.array([target_gripper, 0.0])
                )
                upper_body = self.hand_controller.apply_hand_joints_to_upper_body(
                    self.preparation_target_upper_body.copy(), left_hand_q, right_hand_q
                )
                return {
                    "target_upper_body_pose": upper_body,
                    "wrist_pose": self.preparation_target_wrist if self.preparation_target_wrist is not None else self.initial_eef,
                    "navigate_cmd": np.array(DEFAULT_NAV_CMD),
                    "base_height_command": self.current_height,
                }
            return None
        
        # Parse action: [delta_eef(12), delta_height(1), navigate(3), hand_status(2)]
        delta_eef = action[:12]
        delta_height = float(action[12])
        navigate_cmd = action[13:16]

        if TASK == "garbage":
            if navigate_cmd[0] < -0.1:
                navigate_cmd[0] = -0.25
            if navigate_cmd[0] > 0.15:
                navigate_cmd[0] = 0.25
            else:
                navigate_cmd[0] = 0
            navigate_cmd[1] = 0.0
            navigate_cmd[2] = 0.0
        elif TASK == "cart":
            if navigate_cmd[0] < -0.1:
                navigate_cmd[0] = -0.2
            ## 11111111111111111111111111111111111111111111111111111111111111111111111111111
            if navigate_cmd[0] > 0.18:
                navigate_cmd[0] = 0.20
            else:
                navigate_cmd[0] = 0
            navigate_cmd[1] = 0.0
            navigate_cmd[2] = 0.0
        elif TASK == "bed":
            if navigate_cmd[0] < -0.1:
                navigate_cmd[0] = -0.2
            #### xuyaotiao 第一个参数要小于0.2, 第二个参数要小于0.5
            if navigate_cmd[0] > 0.08:
                navigate_cmd[0] = 0.3
            else:
                navigate_cmd[0] = 0
            navigate_cmd[1] = 0.0
            navigate_cmd[2] = 0.0
        elif TASK == "toy":
            if navigate_cmd[0] < -0.1:
                navigate_cmd[0] = -0.2
            #### xuyaotiao 第一个参数要小于0.2, 第二个参数要小于0.5
            if navigate_cmd[0] > 0.1:
                navigate_cmd[0] = 0.3
            else:
                navigate_cmd[0] = 0

            if navigate_cmd[2] < -0.04 :
                navigate_cmd[2] = -0.5
                navigate_cmd[0] = 0
            # if navigate_cmd[2] > 0.14:
            if navigate_cmd[2] > 0.04:
                navigate_cmd[2] = 0.5
                navigate_cmd[0] = 0
                # navigate_cmd[0] = 0.2
            
            # if TASK != "toy":
            #     navigate_cmd[2] = 0.0
            #     navigate_cmd[1] = 0.0
        hand_status = action[16:] 
        
        # Monitor hand_status for automatic stop (both hands averaged)
        avg_hand_status = np.mean(hand_status)
        
        # First, check if hand is closed (mark that we've seen closed state)
        if avg_hand_status > 0.5:
            if not self.hand_status_seen_closed:
                self.hand_status_seen_closed = True
                print(f"\n[Controller] 👋 Hand closed detected, monitoring for release...")
        
        # Only monitor transitions if we've seen the hand closed at least once
        if self.hand_status_seen_closed and self.prev_hand_status is not None:
            # Check for transition from 1 to 0 (closed to open)
            if self.prev_hand_status > 0.5 and avg_hand_status <= 0.5:
                print(f"\n[Controller] 🔍 Detected hand_status transition: 1 -> 0, starting counter")
                self.hand_status_zero_count = 1  # Start counting
            elif avg_hand_status <= 0.5 and self.hand_status_zero_count > 0:
                # Hand is still open, increment count
                self.hand_status_zero_count += 1
                # Print progress every 10 steps
                if self.hand_status_zero_count % 10 == 0:
                    print(f"[Controller] 📊 Hand status 0 count: {self.hand_status_zero_count}/{self.hand_status_zero_threshold}")
                
                if self.hand_status_zero_count >= self.hand_status_zero_threshold:
                    print(f"\n[Controller] 🛑 Hand status 0 for {self.hand_status_zero_count} steps - Auto stopping!")
                    # Stop inference (simulate pressing 'l')
                    self.is_running = False
                    print("[Controller] ⏸ Inference stopped")
                    
                    # Return to initial position (simulate pressing 'p')
                    # 对于bed任务，返回到调整后的位姿
                    if TASK == "bed":
                        # 创建修改后的目标位姿
                        # initial_eef格式: [left_xyz(3), left_quat(4), right_xyz(3), right_quat(4)]
                        # index 1: 左手y位置，index 8: 右手y位置
                        adjusted_eef = self.initial_eef.copy()
                        adjusted_eef[1] += 0.1  # 左手y增加0.1
                        adjusted_eef[8] -= 0.1  # 右手y减少0.1
                        
                        print(f"[Controller] 📍 Bed任务: 返回到调整后的位姿")
                        print(f"  左手y: {self.initial_eef[1]:.3f} -> {adjusted_eef[1]:.3f} (+0.1)")
                        print(f"  右手y: {self.initial_eef[8]:.3f} -> {adjusted_eef[8]:.3f} (-0.1)")
                        
                        # 临时保存原始initial_eef
                        original_initial_eef = self.initial_eef.copy()
                        # 使用调整后的位姿作为目标
                        self.initial_eef = adjusted_eef
                        
                        self.is_preparing = True
                        self.preparation_complete = False
                        self.preparation_start_time = None
                        self.preparation_start_upper_body = None
                        self.preparation_target_upper_body = None
                        self.preparation_target_wrist = None
                        self.hand_status_zero_count = 0  # Reset counter
                        self.hand_status_seen_closed = False  # Reset flag
                        self.prev_left_hand_status = None
                        self.left_hand_open_transition_count = 0
                        self.left_hand_open_delay_count = 0
                        
                        # 恢复原始initial_eef（在准备阶段完成后会用到）
                        self.original_initial_eef = original_initial_eef
                    else:
                        # 其他任务直接返回初始位置
                        self.is_preparing = True
                        self.preparation_complete = False
                        self.preparation_start_time = None
                        self.preparation_start_upper_body = None
                        self.preparation_target_upper_body = None
                        self.preparation_target_wrist = None
                        self.hand_status_zero_count = 0  # Reset counter
                        self.hand_status_seen_closed = False  # Reset flag
                        self.prev_left_hand_status = None
                        self.left_hand_open_transition_count = 0
                        self.left_hand_open_delay_count = 0
                        print("[Controller] 🔄 Returning to initial position...")
            elif avg_hand_status > 0.5:
                # Hand is closed, reset count
                if self.hand_status_zero_count > 0:
                    print(f"[Controller] Hand status returned to closed, resetting counter (was at {self.hand_status_zero_count})")
                self.hand_status_zero_count = 0
                self.left_hand_open_transition_count = 0
                self.left_hand_open_delay_count = 0  # 手重新闭合时也取消左手延迟
        
        self.prev_hand_status = avg_hand_status
        
        # 左手从闭合变为打开时：第一次不作用，第二次才延迟 N 次再真正打开
        raw_left = float(hand_status[0])
        if self.prev_left_hand_status is not None and self.prev_left_hand_status > 0.1 and raw_left <= 0.1:
            self.left_hand_open_transition_count += 1
            if TASK == "cart":
                if self.left_hand_open_transition_count == 2:
                    self.left_hand_open_delay_count = 52  # 延迟次数，改这里即可
            elif TASK == "garbage":
                if self.left_hand_open_transition_count == 1:
                    self.left_hand_open_delay_count = 0  # 延迟次数，改这里即可
        if self.left_hand_open_delay_count > 0:
            hand_status[0] = 1.0
            self.left_hand_open_delay_count -= 1
        self.prev_left_hand_status = raw_left
        
        # Apply speed scale to all delta values (0.5 = half speed)
        # speed_scale = self.config.speed_scale
        # delta_eef = delta_eef * speed_scale
        # delta_height = delta_height * speed_scale
        # navigate_cmd = navigate_cmd * speed_scale
        
        # Accumulate delta_height to get absolute height
        delta_raw = delta_height
        delta_height = np.clip(delta_height, -0.005, 0.005)  # Max 5mm per step
        # self.current_height += delta_height
        self.current_height = np.clip(self.current_height, 0.36, 0.74)
        
        print(f"hand_status: {hand_status}, height: {self.current_height:.3f}, delta: {delta_height:.4f}, navigate: {navigate_cmd}")

        # Apply delta EEF
        try:
            new_eef = self.delta_eef_controller.apply_delta_eef(delta_eef)
            upper_body_joints = self.delta_eef_controller.compute_upper_body_joints(new_eef)
        except Exception as e:
            print(f"[Controller] IK error: {e}")
            return None
        
        # Apply hand control based on hand_status
        # hand_status: 0 = open, 1 = close (both hands same state)
        try:
            left_hand_q, right_hand_q = self.hand_controller.get_hand_joints(hand_status)
            upper_body_joints = self.hand_controller.apply_hand_joints_to_upper_body(
                upper_body_joints, left_hand_q, right_hand_q
            )
        except Exception as e:
            print(f"[Controller] Hand control error: {e}")
            # Continue without hand control
        
        command = {
            "target_upper_body_pose": upper_body_joints,
            "wrist_pose": new_eef,
            "navigate_cmd": navigate_cmd,
            "base_height_command": float(self.current_height),
            "hand_status": hand_status,
        }
        # print(f"command: {command}")
        
        # Add command to buffer (repeat_count - 1) times, since we return it once now
        for _ in range(self.repeat_count - 1):
            self.command_buffer.append(command.copy())
        
        # Toggle repeat_count between 2 and 3: 5-2=3, 5-3=2
        self.repeat_count = 5 - self.repeat_count
        # self.repeat_count = 5
        
        return command
    
    def _handle_preparation(self, current_obs: dict) -> Optional[dict]:
        """Handle preparation phase: slowly move to initial pose."""
        if self.preparation_start_time is None:
            self.preparation_start_time = time.time()
        
        now = time.time()
        elapsed = now - self.preparation_start_time
        
        # First call: record start position and compute target position
        if self.preparation_start_upper_body is None:
            upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")
            self.preparation_start_upper_body = current_obs["q"][upper_body_indices].copy()
            
            # Reset EEF controller FIRST (this also resets IK solver)
            self.delta_eef_controller.reset_eef(self.initial_eef)
            
            # Then compute target upper body joints from initial EEF
            self.preparation_target_upper_body = self.delta_eef_controller.compute_upper_body_joints(self.initial_eef)
            self.preparation_target_wrist = self.initial_eef.copy()
            
            # Print initial EEF info
            left_pos = self.initial_eef[:3]
            left_quat = self.initial_eef[3:7]
            right_pos = self.initial_eef[7:10]
            right_quat = self.initial_eef[10:14]
            print(f"[Controller] Initial position recorded")
            print(f"[Controller] Target left hand position (m): [{left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f}]")
            print(f"[Controller] Target left hand quaternion (qw,qx,qy,qz): [{left_quat[0]:.4f}, {left_quat[1]:.4f}, {left_quat[2]:.4f}, {left_quat[3]:.4f}]")
            print(f"[Controller] Target right hand position (m): [{right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f}]")
            print(f"[Controller] Target right hand quaternion (qw,qx,qy,qz): [{right_quat[0]:.4f}, {right_quat[1]:.4f}, {right_quat[2]:.4f}, {right_quat[3]:.4f}]")
        
        # Progress
        if elapsed >= self.config.preparation_duration:
            progress = 1.0
            self.is_preparing = False
            self.preparation_complete = True
            print(f"\n[Controller] ✓ Preparation complete! Press 'l' to start")
            
            # 如果是bed任务的自动返回，恢复原始initial_eef
            if self.original_initial_eef is not None:
                self.initial_eef = self.original_initial_eef
                self.original_initial_eef = None
                print(f"[Controller] 已恢复原始初始位姿")
        else:
            t = elapsed / self.config.preparation_duration
            progress = t * t * (3.0 - 2.0 * t)  # smoothstep
        
        # Interpolate upper body joints from start to target
        if (self.preparation_start_upper_body is not None and 
            self.preparation_target_upper_body is not None):
            interp_upper_body = (
                self.preparation_start_upper_body * (1 - progress) +
                self.preparation_target_upper_body * progress
            )
        else:
            upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")
            interp_upper_body = current_obs["q"][upper_body_indices]
        
        if int(elapsed * 10) % 10 == 0:
            print(f"\r[Controller] Preparing... {elapsed:.1f}/{self.config.preparation_duration:.1f}s ({progress*100:.1f}%)", 
                  end="", flush=True)
        
        # Smoothly interpolate gripper to target state
        target_gripper = 1.0 if self.gripper_closed else 0.0
        if self.gripper_current < target_gripper:
            self.gripper_current = min(self.gripper_current + self.gripper_speed, target_gripper)
        elif self.gripper_current > target_gripper:
            self.gripper_current = max(self.gripper_current - self.gripper_speed, target_gripper)
        
        # 仅左手受 'c' 控制，右手保持打开
        left_hand_q, right_hand_q = self.hand_controller.get_hand_joints(
            np.array([self.gripper_current, 0.0])
        )
        interp_upper_body = self.hand_controller.apply_hand_joints_to_upper_body(
            interp_upper_body, left_hand_q, right_hand_q
        )
        
        return {
            "target_upper_body_pose": interp_upper_body,
            "wrist_pose": self.preparation_target_wrist,
            "navigate_cmd": np.array(DEFAULT_NAV_CMD),
            "base_height_command": DEFAULT_BASE_HEIGHT,
        }
    
    def _handle_silent_mode(self, current_obs: dict) -> Optional[dict]:
        """Handle silent mode: slowly interpolate back to initial position."""
        # Calculate interpolation progress
        if self.silent_mode_start_time is not None and self.silent_mode_start_upper_body is not None:
            elapsed = time.time() - self.silent_mode_start_time
            
            if elapsed >= self.silent_mode_duration:
                # Interpolation complete
                progress = 1.0
                if not self.silent_mode_interpolation_complete:
                    self.silent_mode_interpolation_complete = True
                    print(f"\n[Controller] ✓ Silent mode interpolation complete")
            else:
                # Use smoothstep interpolation
                t = elapsed / self.silent_mode_duration
                progress = t * t * (3.0 - 2.0 * t)  # smoothstep
                
                # Print progress every 0.5 seconds
                if int(elapsed * 2) != int((elapsed - 0.02) * 2):
                    print(f"\r[Controller] Silent mode interpolating... {progress*100:.0f}%", end="", flush=True)
            
            # Interpolate upper body joints
            if self.silent_mode_target_upper_body is not None:
                upper_body_joints = (
                    self.silent_mode_start_upper_body * (1 - progress) +
                    self.silent_mode_target_upper_body * progress
                )
            else:
                upper_body_joints = self.delta_eef_controller.compute_upper_body_joints(self.initial_eef)
        else:
            # No start state, use target directly
            upper_body_joints = self.delta_eef_controller.compute_upper_body_joints(self.initial_eef)
        
        return {
            "target_upper_body_pose": upper_body_joints,
            "wrist_pose": self.initial_eef,
            "navigate_cmd": np.array(DEFAULT_NAV_CMD),
            "base_height_command": DEFAULT_BASE_HEIGHT,
            "hand_status": 0.0,  # Open hands
        }
    
    def stop(self):
        """Stop the controller."""
        self.is_running = False
        self.shutdown_event.set()
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=2.0)


# ==================== Main Entry Point ====================

def print_safety_warning():
    """Print safety warning."""
    print("\n" + "!" * 70)
    print("!" + " " * 68 + "!")
    print("!" + "  ⚠️  WARNING: About to run inference on real robot!  ".center(68) + "!")
    print("!" + " " * 68 + "!")
    print("!" * 70)
    print("""
Please ensure:
  1. The robot area is clear of obstacles and people
  2. Emergency stop is within reach
  3. You are familiar with emergency stop procedures

Emergency stop:
  - Press 'o' key to deactivate policy
  - Press physical E-stop button
  - Ctrl+C to terminate program
""")
    print("!" * 70 + "\n")


def main(config: InferenceConfig):
    # Safety warning
    print_safety_warning()
    
    if config.require_confirmation:
        confirm = input("Confirm running on real robot? Enter 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Cancelled")
            return
    
    # Initialize ROS
    ros_manager = ROSManager(node_name="G1InferenceClient")
    node = ros_manager.node
    
    # Start config service
    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())
    
    # Status publishers
    lower_body_policy_status_pub = ROSMsgPublisher(LOWER_BODY_POLICY_STATUS_TOPIC)
    joint_safety_status_pub = ROSMsgPublisher(JOINT_SAFETY_STATUS_TOPIC)
    data_exp_pub = ROSMsgPublisher(STATE_TOPIC_NAME)
    
    telemetry = Telemetry(window_size=100)

    # Initialize robot model
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=config.high_elbow_pose,
    )
    
    # Load WBC config
    wbc_config = config.load_wbc_yaml()
    # breakpoint()
    wbc_config['with_hands'] = True
    # if TASK == "cart" or TASK == "garbage":
    #     wbc_config['enable_waist'] = False
    # else:
    #     wbc_config['enable_waist'] = True
    wbc_config['enable_waist'] = False
    wbc_config['upper_body_max_joint_speed'] = 2000

    # Initialize environment
    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    
    # Initialize WBC policy
    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)

    # Initialize camera client
    camera_client = MJPEGStreamClient(config.camera_url)
    camera_client.start()
    
    # Initialize policy client
    print(f"[Main] Connecting to policy server at {config.host}:{config.port}...")
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=config.host,
        port=config.port,
    )
    print(f"[Main] Server metadata: {policy_client.get_server_metadata()}")
    
    # Warmup inference
    print("[Main] Warming up model...")
    dummy_payload = {
        "images": {"head_left": np.zeros((3, 224, 224), dtype=np.uint8)},
        "prompt": config.prompt,
    }
    for _ in range(2):
        policy_client.infer(dummy_payload)
    print("[Main] Warmup complete")
    
    # Initialize controller
    controller = InferenceController(
        config=config,
        robot_model=robot_model,
        camera_client=camera_client,
        policy_client=policy_client,
    )
    
    # Setup keyboard dispatcher
    keyboard_listener_pub = KeyboardListenerPublisher()
    keyboard_estop = KeyboardEStop()
    if config.keyboard_dispatcher_type == "raw":
        dispatcher = KeyboardDispatcher()
    elif config.keyboard_dispatcher_type == "ros":
        dispatcher = ROSKeyboardDispatcher()
    else:
        raise ValueError(f"Invalid keyboard dispatcher: {config.keyboard_dispatcher_type}")
    
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_listener_pub)
    dispatcher.register(keyboard_estop)
    dispatcher.register(controller)
    dispatcher.start()
    rate = node.create_rate(config.control_frequency)
    
    print("\n" + "=" * 60)
    print("G1 Robot Inference Client")
    print("=" * 60)
    print(f"Policy Server: {config.host}:{config.port}")
    print(f"Camera: {config.camera_url}")
    print(f"Prompt: {config.prompt}")
    print("-" * 60)
    print("Controls:")
    print("  ] - Activate WBC policy / Exit silent mode")
    print("  p - Enter preparation phase")
    print("  c - Toggle left hand open/close (right hand stays open)")
    print("  l - Start/pause inference")
    print("  [ - Enter silent mode (return to initial pose)")
    print("  o - Deactivate policy (emergency)")
    print("  Ctrl+C - Exit")
    print("=" * 60 + "\n")
    
    last_cmd = None
    
    try:
        while ros_manager.ok():
            t_start = time.monotonic()
            
            with telemetry.timer("total_loop"):
                # Get observation with error handling
                with telemetry.timer("observe"):
                    try:
                        obs = env.observe()
                        if obs is None:
                            print("\n[ERROR] Robot observation returned None - communication lost!")
                            print("[ERROR] Stopping control loop for safety.")
                            break
                    except Exception as e:
                        print(f"\n[ERROR] Failed to get robot observation: {e}")
                        print("[ERROR] Stopping control loop for safety.")
                        break
                    wbc_policy.set_observation(obs)
                
                t_now = time.monotonic()
                
                # Get command from controller
                with telemetry.timer("get_command"):
                    cmd = controller.get_command(obs)
                
                # Build WBC goal
                with telemetry.timer("policy_setup"):
                    if cmd is not None:
                        wbc_goal = cmd.copy()
                        last_cmd = cmd.copy()
                    else:
                        upper_body_indices = robot_model.get_joint_group_indices("upper_body")
                        wbc_goal = {
                            "wrist_pose": [0.1924, 0.1499, 0.0796,0.9990, 0.0417, 0.0124, 0.0084,0.1839, -0.1760, 0.0886,0.9983, -0.0160, 0.0138, -0.0540],
                            "navigate_cmd": np.array(DEFAULT_NAV_CMD),
                            "base_height_command": DEFAULT_BASE_HEIGHT,
                            "target_upper_body_pose": obs["q"][upper_body_indices],
                        }
                    
                    wbc_goal["target_time"] = t_now + (1 / config.control_frequency)
                    wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * (
                        1 / config.control_frequency
                    )
                    wbc_policy.set_goal(wbc_goal)
                
                # Get action
                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)
                
                # Execute action
                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)
                
                # Publish status
                with telemetry.timer("publish_status"):
                    policy_use_action = False
                    try:
                        if hasattr(wbc_policy, "lower_body_policy"):
                            policy_use_action = getattr(
                                wbc_policy.lower_body_policy, "use_policy_action", False
                            )
                    except (AttributeError, TypeError):
                        pass
                    
                    policy_status_msg = {"use_policy_action": policy_use_action, "timestamp": t_now}
                    lower_body_policy_status_pub.publish(policy_status_msg)
                    
                    joint_safety_ok = env.get_joint_safety_status()
                    joint_safety_status_msg = {
                        "joint_safety_ok": joint_safety_ok,
                        "timestamp": t_now,
                    }
                    joint_safety_status_pub.publish(joint_safety_status_msg)
                    
                    if not joint_safety_ok:
                        print("\n⚠️ Joint safety warning!")
                
                # Export data
                msg = deepcopy(obs)
                for key in list(obs.keys()):
                    if key.endswith("_image"):
                        del msg[key]
                
                if last_cmd is not None:
                    msg.update({
                        "action": wbc_action["q"],
                        "action.eef": last_cmd.get("wrist_pose", DEFAULT_WRIST_POSE),
                        "base_height_command": last_cmd.get("base_height_command", DEFAULT_BASE_HEIGHT),
                        "navigate_command": last_cmd.get("navigate_cmd", DEFAULT_NAV_CMD),
                        "timestamps": {
                            "main_loop": time.time(),
                            "proprio": time.time(),
                        },
                    })
                data_exp_pub.publish(msg)
                
                end_time = time.monotonic()
            
            rate.sleep()
            
            if config.verbose_timing:
                telemetry.log_timing_info(context="Inference Control Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.control_frequency):
                telemetry.log_timing_info(context="Control Loop Missed", threshold=0.001)
    
    except ros_manager.exceptions() as e:
        print(f"\nROSManager interrupted: {e}")
    except KeyboardInterrupt:
        print("\nUser interrupt")
    finally:
        print("\nCleaning up...")
        controller.stop()
        camera_client.stop()
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()
        print("Exited safely")


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
