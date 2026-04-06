"""
HDF5 Delta EEF Replayer (IK Version) - Real Robot Version

Based on replay_hdf5_eef_ik.py, but uses action_delta_eef for cumulative replay

Differences from replay_hdf5_eef_ik.py:
- Reads action_eef from the first frame as the initial position
- Reads action_delta_eef (incremental EEF)
- Cumulatively applies delta_eef to the initial action_eef
- Uses IK to calculate joint angles corresponding to the cumulative EEF pose

⚠️ Warning: This script controls a real robot, ensure safety measures are in place

Usage:
1. Run the script
2. Press ] to activate policy
3. Press p to enter preparation phase (slowly move to first frame position)
4. After preparation completes, press l to start replay

Usage Example:
    python decoupled_wbc/control/main/teleop/replay_hdf5_delta_eef_ik.py --dataset-dir data_backup/test --episode 0 --no-with-hands
"""
import sys
sys.path.insert(0, '/root/Projects/GR00T-WholeBodyControl')
from copy import deepcopy
from dataclasses import dataclass
import glob
import os
import time
from typing import Optional

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro

from decoupled_wbc.control.envs.g1.g1_env import G1Env
from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
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


@dataclass
class ReplayDeltaEEFIKConfig(ControlLoopConfig):
    """Delta EEF IK replay configuration, inherits from ControlLoopConfig"""

    dataset_dir: str = "outputs/data_1224_0"
    """Dataset directory path"""

    episode: Optional[int] = None
    """Episode index to replay, None means replay all"""

    speed: float = 1.0
    """Replay speed multiplier"""

    loop: bool = False
    """Whether to loop replay"""

    # Real robot default settings
    interface: str = "real"
    """Interface type, use 'real' for real robot"""

    require_confirmation: bool = True
    """Whether confirmation is required to start"""

    preparation_duration: float = 3.0
    """Preparation phase duration (seconds), time for arm to slowly move to first frame position"""

    disable_lower_body: bool = False
    """Disable lower body movement, force all lower body joints to 0, keep lower body stationary (takes effect after WBC policy starts)"""


class HDF5DeltaEEFProvider:
    """
    Provides Delta EEF from HDF5 dataset, cumulatively applies to initial action_eef

    Press 'p' key to enter preparation phase (slowly move to first frame position)
    Press 'l' key to start replay
    """
    
    def __init__(
        self,
        dataset_dir: str,
        robot_model,
        episode: Optional[int] = None,
        speed: float = 1.0,
        preparation_duration: float = 3.0,
        disable_lower_body: bool = False,
    ):
        self.robot_model = robot_model
        self.episodes = self._load_episodes(dataset_dir, episode)
        self.speed = speed
        self.preparation_duration = preparation_duration
        self.disable_lower_body = disable_lower_body
        
        # 初始化 IK 求解器
        left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
        self.retargeting_ik = TeleopRetargetingIK(
            robot_model=robot_model,
            left_hand_ik_solver=left_hand_ik_solver,
            right_hand_ik_solver=right_hand_ik_solver,
            enable_visualization=False,
            body_active_joint_groups=["upper_body"],
        )
        
        # 获取手腕 frame 名称
        self.left_wrist_frame = robot_model.supplemental_info.hand_frame_names["left"]
        self.right_wrist_frame = robot_model.supplemental_info.hand_frame_names["right"]
        
        # 回放状态
        self.is_replaying = False
        self.is_preparing = False
        self.preparation_complete = False
        self.current_episode_idx = 0
        self.current_frame_idx = 0
        self.last_frame_time = None
        self.frame_interval = None
        
        # 准备阶段相关
        self.preparation_start_time = None
        self.preparation_start_upper_body = None
        self.preparation_target_upper_body = None
        self.preparation_target_wrist = None
        
        # Cumulative EEF pose (for delta accumulation)
        self.current_cumulative_eef = None  # (14,) current cumulative EEF pose

        print(f"[HDF5 Delta EEF] Loaded {len(self.episodes)} episodes")
        print(f"[HDF5 Delta EEF] Replay speed: {speed}x")
        print(f"[HDF5 Delta EEF] Preparation time: {preparation_duration}s")
        if disable_lower_body:
            print(f"[HDF5 Delta EEF] ⚠️ Lower body disabled, all lower body joints will remain at 0")
        print(f"[HDF5 Delta EEF] Press 'p' key to enter preparation phase (slowly move to first frame)")
        print(f"[HDF5 Delta EEF] Press 'l' key to start/pause replay")
    
    def _load_episodes(self, dataset_dir: str, episode: Optional[int]) -> list:
        """Load episode data"""
        print(f"Loading episodes from {dataset_dir}")
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Data directory does not exist: {dataset_dir}")
        
        hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
        if len(hdf5_files) == 0:
            single_file = os.path.join(dataset_dir, "data.hdf5")
            if os.path.exists(single_file):
                hdf5_files = [single_file]
            else:
                raise FileNotFoundError(f"HDF5 files not found in {dataset_dir}")

        episodes = []
        for hdf5_file in hdf5_files:
            try:
                with h5py.File(hdf5_file, "r") as f:
                    # Load action_delta_eef (load first to get frame count)
                    if "action_delta_eef" not in f:
                        raise ValueError(f"action_delta_eef not found in {hdf5_file}, please run add_delta_eef_to_hdf5.py first")
                    delta_eef_data = np.array(f["action_delta_eef"])

                    ep = {
                        "file": os.path.basename(hdf5_file),
                        "fps": f.attrs.get("fps", 20),
                        "num_frames": f.attrs.get("num_frames", len(delta_eef_data)),
                    }

                    # Use manually specified fixed initial EEF pose (not reading from HDF5 file)
                    # ep["initial_eef"] = np.array([
                    #     0.21101273,  0.13912974,  0.21924709,  # left position
                    #     0.76827627,  0.61089334, -0.17724066,  0.07174011,  # left quaternion (qw,qx,qy,qz)
                    #     # 0.18453715, -0.13912033,  0.25318919,  # right position
                    #     0.20453715, -0.13912033,  0.20318919,  # right position
                    #     0.71888301, -0.6426301 , -0.25615103, -0.06797372,  # right quaternion (qw,qx,qy,qz)
                    # ])
                    # #toy
                    ep["initial_eef"] = np.array([
                        0.1924, 0.1499, 0.1,  # left position
                        0.9990, 0.0417, 0.0124, 0.0084,  # left quaternion (qw,qx,qy,qz)
                        0.1839, -0.1760, 0.1,  # right position
                        0.9983, -0.0160, 0.0138, -0.0540,  # right quaternion (qw,qx,qy,qz)
                    ])
                    # Use delta_eef data already loaded above
                    ep["delta_eef"] = delta_eef_data

                    # Load navigation and height commands (optional)
                    if "teleop_navigate_command" in f:
                        ep["navigate_cmd"] = np.array(f["teleop_navigate_command"])

                    if "teleop_base_height_command" in f:
                        ep["base_height_command"] = np.array(f["teleop_base_height_command"])

                    episodes.append(ep)
                    print(f"  Loaded: {ep['file']} - {ep['num_frames']} frames, {ep['fps']} FPS")
                    print(f"    Initial EEF: {ep['initial_eef']}")
                    print(f"    Delta EEF shape: {ep['delta_eef'].shape}")
            except Exception as e:
                print(f"  [WARN] Failed to load {hdf5_file}: {e}")
        # breakpoint()
        if episode is not None:
            if episode >= len(episodes):
                raise ValueError(f"Episode {episode} does not exist, there are {len(episodes)} episodes")
            episodes = [episodes[episode]]

        return episodes
    
    def _delta_to_transform_matrix(self, delta_xyzrpy: np.ndarray) -> np.ndarray:
        """
        Convert delta xyzrpy to transformation matrix

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
        Convert EEF data to transformation matrix dictionary

        Args:
            eef_data: (14,) [left: x,y,z,qw,qx,qy,qz, right: x,y,z,qw,qx,qy,qz]
                     Note: quaternion format is scalar_first=True, i.e., (w, x, y, z)

        Returns:
            dict: {left_wrist_frame: T_left, right_wrist_frame: T_right}
        """
        # Extract left and right hand data
        left_pos = eef_data[:3]
        left_quat_scalar_first = eef_data[3:7]  # (qw, qx, qy, qz) - scalar_first format
        right_pos = eef_data[7:10]
        right_quat_scalar_first = eef_data[10:14]  # (qw, qx, qy, qz) - scalar_first format

        # Convert to format expected by scipy (x, y, z, w)
        # scalar_first (w, x, y, z) -> vector_first (x, y, z, w)
        left_quat = np.array([left_quat_scalar_first[1], left_quat_scalar_first[2],
                             left_quat_scalar_first[3], left_quat_scalar_first[0]])
        right_quat = np.array([right_quat_scalar_first[1], right_quat_scalar_first[2],
                              right_quat_scalar_first[3], right_quat_scalar_first[0]])

        # Convert to transformation matrices
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
        将变换矩阵转换为 EEF 数据格式
        
        Args:
            T_left: (4, 4) 左手变换矩阵
            T_right: (4, 4) 右手变换矩阵
        
        Returns:
            (14,) [left: x,y,z,qw,qx,qy,qz, right: x,y,z,qw,qx,qy,qz]
        """
        # 提取位置和旋转
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
        将 delta_eef 应用到 eef_data 上
        
        Args:
            eef_data: (14,) 当前 EEF 位姿
            delta_eef: (12,) [left: dx,dy,dz,roll,pitch,yaw, right: dx,dy,dz,roll,pitch,yaw]
        
        Returns:
            (14,) 应用 delta 后的新 EEF 位姿
        """
        # 将当前 EEF 转换为变换矩阵
        T_dict = self._eef_to_transformation_matrix(eef_data)
        T_left_curr = T_dict[self.left_wrist_frame]
        T_right_curr = T_dict[self.right_wrist_frame]
        
        # 提取左右手的 delta
        left_delta = delta_eef[:6]  # [dx, dy, dz, roll, pitch, yaw]
        right_delta = delta_eef[6:12]  # [dx, dy, dz, roll, pitch, yaw]
        
        # 将 delta 转换为变换矩阵
        delta_T_left = self._delta_to_transform_matrix(left_delta)
        delta_T_right = self._delta_to_transform_matrix(right_delta)
        
        # 应用 delta：T_new = T_curr @ delta_T
        T_left_new = T_left_curr @ delta_T_left
        T_right_new = T_right_curr @ delta_T_right
        
        # 转换回 EEF 格式
        return self._transform_matrix_to_eef(T_left_new, T_right_new)
    
    def _compute_upper_body_joints_from_eef(self, eef_data: np.ndarray) -> np.ndarray:
        """
        从 EEF 位姿计算上半身关节角度
        
        使用与 TeleopPolicy 相同的方式：通过 set_goal() 和 get_action() 来维护状态连续性
        
        Args:
            eef_data: (14,) EEF 位姿
        
        Returns:
            (N,) 上半身关节角度
        """
        # 转换为变换矩阵
        body_data = self._eef_to_transformation_matrix(eef_data)
        
        # 使用与 TeleopPolicy 相同的方式：set_goal() + get_action()
        ik_data = {
            "body_data": body_data,
            "left_hand_data": None,  # type: ignore  # 不使用手部 IK
            "right_hand_data": None,  # type: ignore  # 不使用手部 IK
        }
        self.retargeting_ik.set_goal(ik_data)
        target_joints = np.array(self.retargeting_ik.get_action())  # type: ignore
        
        return target_joints
    
    def handle_keyboard_button(self, key):
        """处理键盘按键"""
        if key == "p":
            if not self.is_preparing and not self.is_replaying:
                self.is_preparing = True
                self.preparation_complete = False
                self.preparation_start_time = time.time()
                self.preparation_start_upper_body = None
                self.preparation_target_upper_body = None
                self.preparation_target_wrist = None
                print("\n[HDF5 Delta EEF] 🔄 Entering preparation phase: slowly moving to first frame position...")

                # Print first frame left and right hand position and quaternion
                if len(self.episodes) > 0:
                    ep = self.episodes[self.current_episode_idx]
                    initial_eef = ep["initial_eef"]
                    left_pos = initial_eef[:3]
                    left_quat = initial_eef[3:7]  # (qw, qx, qy, qz)
                    right_pos = initial_eef[7:10]
                    right_quat = initial_eef[10:14]  # (qw, qx, qy, qz)
                    print(f"[HDF5 Delta EEF] First frame left hand position (m): [{left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f}]")
                    print(f"[HDF5 Delta EEF] First frame left hand quaternion (qw,qx,qy,qz): [{left_quat[0]:.4f}, {left_quat[1]:.4f}, {left_quat[2]:.4f}, {left_quat[3]:.4f}]")
                    print(f"[HDF5 Delta EEF] First frame right hand position (m): [{right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f}]")
                    print(f"[HDF5 Delta EEF] First frame right hand quaternion (qw,qx,qy,qz): [{right_quat[0]:.4f}, {right_quat[1]:.4f}, {right_quat[2]:.4f}, {right_quat[3]:.4f}]")

                print(f"[HDF5 Delta EEF] Preparation time: {self.preparation_duration}s, press 'l' to start replay after completion")
        elif key == "l":
            if self.is_preparing:
                print("\n[HDF5 Delta EEF] ⚠️ Preparation phase not complete, please wait for completion before pressing 'l'")
                return

            if not self.preparation_complete and not self.is_replaying:
                print("\n[HDF5 Delta EEF] ⚠️ Please press 'p' to complete preparation phase first")
                return

            self.is_replaying = not self.is_replaying
            if self.is_replaying:
                self.preparation_complete = False
                print("\n[HDF5 Delta EEF] ▶ Starting replay")
                self.current_frame_idx = 0
                self.last_frame_time = time.time()
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
                # Reset cumulative EEF to initial value
                self.current_cumulative_eef = ep["initial_eef"].copy()
                print(f"[HDF5 Delta EEF] Reset cumulative EEF to initial value")
            else:
                print("\n[HDF5 Delta EEF] ⏸ Paused replay")
    
    def get_command(self, robot_model, current_obs=None) -> Optional[dict]:
        """
        Get current frame command (calculated through cumulative delta_eef)
        """
        if len(self.episodes) == 0:
            return None

        # Preparation phase: slowly move to first frame position
        if self.is_preparing:
            if current_obs is None:
                print("[HDF5 Delta EEF] Warning: preparation phase requires current_obs, but not provided")
                return None

            if self.preparation_start_time is None:
                self.preparation_start_time = time.time()

            now = time.time()
            elapsed = now - self.preparation_start_time

            # First call, record starting position and target position
            if self.preparation_start_upper_body is None:
                upper_body_indices = robot_model.get_joint_group_indices("upper_body")
                self.preparation_start_upper_body = current_obs["q"][upper_body_indices].copy()

                ep = self.episodes[self.current_episode_idx]
                # Calculate target joint angles from first frame EEF
                initial_eef = ep["initial_eef"]
                self.preparation_target_upper_body = self._compute_upper_body_joints_from_eef(initial_eef)
                self.preparation_target_wrist = initial_eef.copy()

                print(f"[HDF5 Delta EEF] Starting position recorded, target position set")

            # Calculate interpolation progress
            if elapsed >= self.preparation_duration:
                progress = 1.0
                self.is_preparing = False
                self.preparation_complete = True
                print(f"\n[HDF5 Delta EEF] ✓ Preparation complete! Press 'l' to start replay")
            else:
                t = elapsed / self.preparation_duration
                progress = t * t * (3.0 - 2.0 * t)  # smoothstep

            # Interpolate upper arm position
            if (self.preparation_start_upper_body is not None and
                self.preparation_target_upper_body is not None):
                interp_upper_body = (
                    self.preparation_start_upper_body * (1 - progress) +
                    self.preparation_target_upper_body * progress
                )
            else:
                upper_body_indices = robot_model.get_joint_group_indices("upper_body")
                interp_upper_body = current_obs["q"][upper_body_indices]

            cmd = {
                "target_upper_body_pose": interp_upper_body,
                "wrist_pose": self.preparation_target_wrist,  # Use first frame wrist position
                "navigate_cmd": DEFAULT_NAV_CMD,
                "base_height_command": DEFAULT_BASE_HEIGHT,
            }

            if int(elapsed * 10) % 10 == 0:
                print(f"\r[HDF5 Delta EEF] Preparing... {elapsed:.1f}/{self.preparation_duration:.1f}s ({progress*100:.1f}%)",
                      end="", flush=True)

            return cmd

        # Preparation complete, waiting to start replay: continue sending first frame command
        if self.preparation_complete and not self.is_replaying:
            if (self.preparation_target_upper_body is not None and
                self.preparation_target_wrist is not None):
                cmd = {
                    "target_upper_body_pose": self.preparation_target_upper_body,
                    "wrist_pose": self.preparation_target_wrist,
                    "navigate_cmd": DEFAULT_NAV_CMD,
                    "base_height_command": DEFAULT_BASE_HEIGHT,
                }
                return cmd

        # Replay phase
        if not self.is_replaying:
            return None

        ep = self.episodes[self.current_episode_idx]

        # Check if need to advance to next frame
        now = time.time()
        if self.last_frame_time is not None and self.frame_interval is not None:
            if now - self.last_frame_time >= self.frame_interval:
                self.current_frame_idx += 1
                self.last_frame_time = now

                # Apply delta to cumulative EEF (start applying delta from second frame)
                # delta_eef[0] is 0 (first frame), delta_eef[i] is increment from frame i-1 to frame i
                if self.current_cumulative_eef is not None and self.current_frame_idx > 0:
                    if self.current_frame_idx < len(ep["delta_eef"]):
                        delta = ep["delta_eef"][self.current_frame_idx]
                        self.current_cumulative_eef = self._apply_delta_to_eef(
                            self.current_cumulative_eef, delta
                        )

        # Check if current episode ended
        print(f"current_frame_idx: {self.current_frame_idx}, num_frames: {ep['num_frames']}")
        if self.current_frame_idx >= ep["num_frames"]:
            self.current_frame_idx = 0
            self.current_episode_idx += 1

            if self.current_episode_idx >= len(self.episodes):
                print("\n[HDF5 Delta EEF] ✓ All episodes replay complete")
                self.is_replaying = False
                self.current_episode_idx = 0
                return None
            else:
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
                # Reset cumulative EEF to new initial value
                self.current_cumulative_eef = ep["initial_eef"].copy()
                print(f"\n[HDF5 Delta EEF] Switched to Episode {self.current_episode_idx + 1}/{len(self.episodes)}")

        # Get current cumulative EEF (if not initialized yet, use initial value)
        if self.current_cumulative_eef is None:
            self.current_cumulative_eef = ep["initial_eef"].copy()

        # Use IK to calculate upper body joint angles
        upper_body_joints = self._compute_upper_body_joints_from_eef(self.current_cumulative_eef)

        # Build command
        # If lower body disabled, force navigate_cmd to [0, 0, 0]
        if self.disable_lower_body:
            navigate_cmd = np.array(DEFAULT_NAV_CMD)
        else:
            navigate_cmd = ep.get("navigate_cmd", DEFAULT_NAV_CMD)[self.current_frame_idx] if "navigate_cmd" in ep else DEFAULT_NAV_CMD

        cmd = {
            "target_upper_body_pose": upper_body_joints,
            "wrist_pose": self.current_cumulative_eef,
            "navigate_cmd": navigate_cmd,
            "base_height_command": ep.get("base_height_command", DEFAULT_BASE_HEIGHT)[self.current_frame_idx] if "base_height_command" in ep else DEFAULT_BASE_HEIGHT,
        }

        if isinstance(cmd["base_height_command"], np.ndarray):
            cmd["base_height_command"] = cmd["base_height_command"].item() if cmd["base_height_command"].size == 1 else cmd["base_height_command"][0]

        # Print frame number and progress
        progress = (self.current_frame_idx + 1) / ep["num_frames"] * 100
        nav = cmd["navigate_cmd"]
        print(
            f"\r[HDF5 Delta EEF] Ep {self.current_episode_idx + 1}/{len(self.episodes)} | "
            f"Frame {self.current_frame_idx + 1}/{ep['num_frames']} ({progress:.1f}%) | "
            f"Nav: [{nav[0]:+.2f}, {nav[1]:+.2f}, {nav[2]:+.2f}]",
            end="", flush=True
        )

        return cmd


def print_safety_warning():
    """Print safety warning"""
    print("\n" + "!" * 70)
    print("!" + " " * 68 + "!")
    print("!" + "  ⚠️  Warning: About to replay Delta EEF data on real robot (IK version)!".center(50) + "!")
    print("!" + " " * 68 + "!")
    print("!" * 70)
    print("""
Please ensure:
  1. No obstacles or people around the robot
  2. Emergency stop switch is within reach and functional
  3. Delta EEF data motion range is within safe limits
  4. Replay speed is set appropriately
  5. You are familiar with emergency stop procedures

Emergency stop methods:
  - Press keyboard 'o' to deactivate policy
  - Press physical emergency stop button
  - Ctrl+C to terminate program
""")
    print("!" * 70 + "\n")


def main(config: ReplayDeltaEEFIKConfig):
    # Safety confirmation
    print_safety_warning()

    if config.require_confirmation:
        confirm = input("Confirm running replay on real robot? Enter 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Cancelled")
            return
    
    ros_manager = ROSManager(node_name="ReplayDeltaEEFIKControlLoop")
    node = ros_manager.node
    
    # 启动配置服务器
    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())
    
    # 状态发布器
    data_exp_pub = ROSMsgPublisher(STATE_TOPIC_NAME)
    lower_body_policy_status_pub = ROSMsgPublisher(LOWER_BODY_POLICY_STATUS_TOPIC)
    joint_safety_status_pub = ROSMsgPublisher(JOINT_SAFETY_STATUS_TOPIC)
    
    telemetry = Telemetry(window_size=100)
    
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )
    
    wbc_config = config.load_wbc_yaml()
    
    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    
    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)
    
    # Delta EEF 提供器（包含 IK 求解器）
    delta_eef_provider = HDF5DeltaEEFProvider(
        dataset_dir=config.dataset_dir,
        robot_model=robot_model,
        episode=config.episode,
        speed=config.speed,
        preparation_duration=config.preparation_duration,
        disable_lower_body=config.disable_lower_body,
    )
    
    # 键盘调度器
    keyboard_listener_pub = KeyboardListenerPublisher()
    keyboard_estop = KeyboardEStop()
    if config.keyboard_dispatcher_type == "raw":
        dispatcher = KeyboardDispatcher()
    elif config.keyboard_dispatcher_type == "ros":
        dispatcher = ROSKeyboardDispatcher()
    else:
        raise ValueError(
            f"Invalid keyboard dispatcher: {config.keyboard_dispatcher_type}, please use 'raw' or 'ros'"
        )
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_listener_pub)
    dispatcher.register(keyboard_estop)
    dispatcher.register(delta_eef_provider)
    dispatcher.start()
    
    rate = node.create_rate(config.control_frequency)
    
    print("\n" + "=" * 60)
    print("HDF5 Delta EEF IK Replay Control Loop")
    print("=" * 60)
    print(f"Dataset: {config.dataset_dir}")
    print(f"Replay speed: {config.speed}x")
    print("-" * 60)
    print("Operation steps:")
    print("  1. Press ] to activate WBC policy")
    print("  2. Press p to enter preparation phase (slowly move to first frame position)")
    print("  3. After preparation completes, press l to start/pause replay")
    print("-" * 60)
    print("Safety keys:")
    print("  o - Deactivate policy (use in emergency)")
    print("  Ctrl+C - Exit program")
    print("=" * 60 + "\n")
    
    last_teleop_cmd = None
    
    try:
        while ros_manager.ok():
            t_start = time.monotonic()
            
            with telemetry.timer("total_loop"):
                # 获取观测
                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)
                
                t_now = time.monotonic()
                
                # 获取 command（通过累积 delta_eef 计算）
                with telemetry.timer("get_command"):
                    replay_cmd = delta_eef_provider.get_command(robot_model, current_obs=obs)
                
                # 构建 wbc_goal
                with telemetry.timer("policy_setup"):
                    if replay_cmd:
                        wbc_goal = replay_cmd.copy()
                        last_teleop_cmd = replay_cmd.copy()
                    else:
                        upper_body_indices = robot_model.get_joint_group_indices("upper_body")
                        wbc_goal = {
                            "wrist_pose": DEFAULT_WRIST_POSE,
                            "navigate_cmd": np.array(DEFAULT_NAV_CMD),
                            "base_height_command": DEFAULT_BASE_HEIGHT,
                            "target_upper_body_pose": obs["q"][upper_body_indices],
                        }
                    
                    wbc_goal["target_time"] = t_now + (1 / config.control_frequency)
                    wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * (
                        1 / config.control_frequency
                    )
                    wbc_policy.set_goal(wbc_goal)
                
                # 获取动作
                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)
                
                # 如果禁用下半身，强制将下半身关节设置为 0
                if config.disable_lower_body:
                    lower_body_indices = robot_model.get_joint_group_indices("lower_body")
                    wbc_action["q"][lower_body_indices] = 0.0
                
                # 执行动作
                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)
                
                # 发布状态
                with telemetry.timer("publish_status"):
                    policy_use_action = False
                    try:
                        if hasattr(wbc_policy, "lower_body_policy"):
                            policy_use_action = getattr(
                                wbc_policy.lower_body_policy, "use_policy_action", False
                            )
                    except (AttributeError, TypeError):
                        policy_use_action = False
                    
                    policy_status_msg = {"use_policy_action": policy_use_action, "timestamp": t_now}
                    lower_body_policy_status_pub.publish(policy_status_msg)
                    
                    joint_safety_ok = env.get_joint_safety_status()
                    joint_safety_status_msg = {
                        "joint_safety_ok": joint_safety_ok,
                        "timestamp": t_now,
                    }
                    joint_safety_status_pub.publish(joint_safety_status_msg)
                    
                    if not joint_safety_ok:
                        print("\n⚠️ Joint safety warning! Please check robot status")

                # Export data
                msg = deepcopy(obs)
                for key in list(obs.keys()):
                    if key.endswith("_image"):
                        del msg[key]
                
                if last_teleop_cmd:
                    msg.update({
                        "action": wbc_action["q"],
                        "action.eef": last_teleop_cmd.get("wrist_pose", DEFAULT_WRIST_POSE),
                        "base_height_command": last_teleop_cmd.get(
                            "base_height_command", DEFAULT_BASE_HEIGHT
                        ),
                        "navigate_command": last_teleop_cmd.get(
                            "navigate_cmd", DEFAULT_NAV_CMD
                        ),
                        "timestamps": {
                            "main_loop": time.time(),
                            "proprio": time.time(),
                        },
                    })
                data_exp_pub.publish(msg)
                
                end_time = time.monotonic()
            
            rate.sleep()
            
            if config.verbose_timing:
                telemetry.log_timing_info(context="Replay Delta EEF IK Control Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.control_frequency):
                telemetry.log_timing_info(context="Replay Delta EEF IK Control Loop Missed", threshold=0.001)
    
    except ros_manager.exceptions() as e:
        print(f"\nROSManager interrupted: {e}")
    except KeyboardInterrupt:
        print("\nUser interrupted")
    finally:
        print("\nCleaning up...")
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()
        print("Safely exited")


if __name__ == "__main__":
    config = tyro.cli(ReplayDeltaEEFIKConfig)
    main(config)

