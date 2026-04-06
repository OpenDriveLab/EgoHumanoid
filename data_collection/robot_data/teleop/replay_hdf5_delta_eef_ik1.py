"""
HDF5 Delta EEF 回放器（IK 版本）- 真机版本

基于 replay_hdf5_eef_ik.py，但使用 action_delta_eef 进行累积回放

与 replay_hdf5_eef_ik.py 的区别：
- 读取第一帧的 action_eef 作为初始位置
- 读取 action_delta_eef（增量 EEF）
- 将 delta_eef 累积应用到初始 action_eef 上
- 使用 IK 计算累积后的 EEF 位姿对应的关节角度

⚠️ 警告：此脚本用于控制真实机器人，请确保安全措施到位

使用方式：
1. 运行脚本
2. 按 ] 激活策略
3. 按 p 进入准备阶段（缓慢移动到第一帧位置）
4. 等待准备完成后，按 l 开始回放

使用示例：
    python decoupled_wbc/control/main/teleop/replay_hdf5_delta_eef_ik.py --dataset-dir outputs/test --episode 0 --no-with-hands
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
    """Delta EEF IK 回放配置，继承自 ControlLoopConfig"""
    
    dataset_dir: str = "outputs/data_1224_0"
    """数据集目录路径"""
    
    episode: Optional[int] = None
    """要回放的 episode 索引，None 表示回放所有"""
    
    speed: float = 1.0
    """回放速度倍率"""
    
    loop: bool = False
    """是否循环回放"""
    
    # 真机默认设置
    interface: str = "real"
    """接口类型，真机使用 'real'"""
    
    require_confirmation: bool = True
    """是否需要确认才能开始"""
    
    preparation_duration: float = 3.0
    """准备阶段持续时间（秒），手臂缓慢移动到第一帧位置的时间"""


class HDF5DeltaEEFProvider:
    """
    从 HDF5 数据集提供 Delta EEF，累积应用到初始 action_eef 上
    
    按 'p' 键进入准备阶段（缓慢移动到第一帧位置）
    按 'l' 键开始回放
    """
    
    def __init__(
        self,
        dataset_dir: str,
        robot_model,
        episode: Optional[int] = None,
        speed: float = 1.0,
        preparation_duration: float = 3.0,
    ):
        self.robot_model = robot_model
        self.episodes = self._load_episodes(dataset_dir, episode)
        self.speed = speed
        self.preparation_duration = preparation_duration
        
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
        
        # 累积的 EEF 位姿（用于 delta 累积）
        self.current_cumulative_eef = None  # (14,) 当前累积的 EEF 位姿
        
        print(f"[HDF5 Delta EEF] 已加载 {len(self.episodes)} 个 episode")
        print(f"[HDF5 Delta EEF] 回放速度: {speed}x")
        print(f"[HDF5 Delta EEF] 准备时间: {preparation_duration}s")
        print(f"[HDF5 Delta EEF] 按 'p' 键进入准备阶段（缓慢移动到第一帧）")
        print(f"[HDF5 Delta EEF] 按 'l' 键开始/暂停回放")
    
    def _load_episodes(self, dataset_dir: str, episode: Optional[int]) -> list:
        """加载 episode 数据"""
        print(f"Loading episodes from {dataset_dir}")
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"数据目录不存在: {dataset_dir}")
        
        hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
        if len(hdf5_files) == 0:
            single_file = os.path.join(dataset_dir, "data.hdf5")
            if os.path.exists(single_file):
                hdf5_files = [single_file]
            else:
                raise FileNotFoundError(f"在 {dataset_dir} 中未找到 HDF5 文件")
        
        episodes = []
        for hdf5_file in hdf5_files:
            try:
                with h5py.File(hdf5_file, "r") as f:
                    ep = {
                        "file": os.path.basename(hdf5_file),
                        "fps": f.attrs.get("fps", 20),
                        "num_frames": f.attrs.get("num_frames", len(f["observation_state"])),
                    }
                    
                    # 加载第一帧的 action_eef（作为初始位置）
                    if "action_eef" not in f:
                        raise ValueError(f"在 {hdf5_file} 中未找到 action_eef")
                    action_eef_all = np.array(f["action_eef"])
                    ep["initial_eef"] = action_eef_all[0].copy()  # 第一帧作为初始位置
                    
                    # 加载 action_delta_eef
                    if "action_delta_eef" not in f:
                        raise ValueError(f"在 {hdf5_file} 中未找到 action_delta_eef，请先运行 add_delta_eef_to_hdf5.py")
                    ep["delta_eef"] = np.array(f["action_delta_eef"])
                    
                    # 加载导航和高度命令（可选）
                    if "teleop_navigate_command" in f:
                        ep["navigate_cmd"] = np.array(f["teleop_navigate_command"])
                    
                    if "teleop_base_height_command" in f:
                        ep["base_height_command"] = np.array(f["teleop_base_height_command"])
                    
                    episodes.append(ep)
                    print(f"  已加载: {ep['file']} - {ep['num_frames']} 帧, {ep['fps']} FPS")
                    print(f"    初始 EEF: {ep['initial_eef']}")
                    print(f"    Delta EEF shape: {ep['delta_eef'].shape}")
            except Exception as e:
                print(f"  [WARN] 加载 {hdf5_file} 失败: {e}")
        
        if episode is not None:
            if episode >= len(episodes):
                raise ValueError(f"Episode {episode} 不存在，共有 {len(episodes)} 个")
            episodes = [episodes[episode]]
        
        return episodes
    
    def _delta_to_transform_matrix(self, delta_xyzrpy: np.ndarray) -> np.ndarray:
        """
        将 delta xyzrpy 转换为变换矩阵
        
        Args:
            delta_xyzrpy: (6,) [dx, dy, dz, roll, pitch, yaw]
        
        Returns:
            (4, 4) 变换矩阵
        """
        T = np.eye(4)
        T[:3, 3] = delta_xyzrpy[:3]  # 平移
        T[:3, :3] = R.from_euler('xyz', delta_xyzrpy[3:6]).as_matrix()  # 旋转
        return T
    
    def _eef_to_transformation_matrix(self, eef_data: np.ndarray) -> dict:
        """
        将 EEF 数据转换为变换矩阵字典
        
        Args:
            eef_data: (14,) [left: x,y,z,qw,qx,qy,qz, right: x,y,z,qw,qx,qy,qz]
                     注意：四元数格式是 scalar_first=True，即 (w, x, y, z)
        
        Returns:
            dict: {left_wrist_frame: T_left, right_wrist_frame: T_right}
        """
        # 提取左右手数据
        left_pos = eef_data[:3]
        left_quat_scalar_first = eef_data[3:7]  # (qw, qx, qy, qz) - scalar_first 格式
        right_pos = eef_data[7:10]
        right_quat_scalar_first = eef_data[10:14]  # (qw, qx, qy, qz) - scalar_first 格式
        
        # 转换为 scipy 期望的格式 (x, y, z, w)
        # scalar_first (w, x, y, z) -> vector_first (x, y, z, w)
        left_quat = np.array([left_quat_scalar_first[1], left_quat_scalar_first[2], 
                             left_quat_scalar_first[3], left_quat_scalar_first[0]])
        right_quat = np.array([right_quat_scalar_first[1], right_quat_scalar_first[2], 
                              right_quat_scalar_first[3], right_quat_scalar_first[0]])
        
        # 转换为变换矩阵
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
                print("\n[HDF5 Delta EEF] 🔄 进入准备阶段：缓慢移动到第一帧位置...")
                
                # 打印第一帧左右手位置和四元数
                if len(self.episodes) > 0:
                    ep = self.episodes[self.current_episode_idx]
                    initial_eef = ep["initial_eef"]
                    left_pos = initial_eef[:3]
                    left_quat = initial_eef[3:7]  # (qw, qx, qy, qz)
                    right_pos = initial_eef[7:10]
                    right_quat = initial_eef[10:14]  # (qw, qx, qy, qz)
                    print(f"[HDF5 Delta EEF] 第一帧左手位置 (m): [{left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f}]")
                    print(f"[HDF5 Delta EEF] 第一帧左手四元数 (qw,qx,qy,qz): [{left_quat[0]:.4f}, {left_quat[1]:.4f}, {left_quat[2]:.4f}, {left_quat[3]:.4f}]")
                    print(f"[HDF5 Delta EEF] 第一帧右手位置 (m): [{right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f}]")
                    print(f"[HDF5 Delta EEF] 第一帧右手四元数 (qw,qx,qy,qz): [{right_quat[0]:.4f}, {right_quat[1]:.4f}, {right_quat[2]:.4f}, {right_quat[3]:.4f}]")
                
                print(f"[HDF5 Delta EEF] 准备时间: {self.preparation_duration}s，完成后按 'l' 开始回放")
        elif key == "l":
            if self.is_preparing:
                print("\n[HDF5 Delta EEF] ⚠️ 准备阶段未完成，请等待完成后再按 'l'")
                return
            
            if not self.preparation_complete and not self.is_replaying:
                print("\n[HDF5 Delta EEF] ⚠️ 请先按 'p' 完成准备阶段")
                return
            
            self.is_replaying = not self.is_replaying
            if self.is_replaying:
                self.preparation_complete = False
                print("\n[HDF5 Delta EEF] ▶ 开始回放")
                self.current_frame_idx = 0
                self.last_frame_time = time.time()
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
                # 重置累积 EEF 为初始值
                self.current_cumulative_eef = ep["initial_eef"].copy()
                print(f"[HDF5 Delta EEF] 重置累积 EEF 为初始值")
            else:
                print("\n[HDF5 Delta EEF] ⏸ 暂停回放")
    
    def get_command(self, robot_model, current_obs=None) -> Optional[dict]:
        """
        获取当前帧的 command（通过累积 delta_eef 计算）
        """
        if len(self.episodes) == 0:
            return None
        
        # 准备阶段：缓慢移动到第一帧位置
        if self.is_preparing:
            if current_obs is None:
                print("[HDF5 Delta EEF] 警告: 准备阶段需要 current_obs，但未提供")
                return None
            
            if self.preparation_start_time is None:
                self.preparation_start_time = time.time()
            
            now = time.time()
            elapsed = now - self.preparation_start_time
            
            # 首次调用，记录起始位置和目标位置
            if self.preparation_start_upper_body is None:
                upper_body_indices = robot_model.get_joint_group_indices("upper_body")
                self.preparation_start_upper_body = current_obs["q"][upper_body_indices].copy()
                
                ep = self.episodes[self.current_episode_idx]
                # 从第一帧 EEF 计算目标关节角度
                initial_eef = ep["initial_eef"]
                self.preparation_target_upper_body = self._compute_upper_body_joints_from_eef(initial_eef)
                self.preparation_target_wrist = initial_eef.copy()
                
                print(f"[HDF5 Delta EEF] 开始位置已记录，目标位置已设置")
            
            # 计算插值进度
            if elapsed >= self.preparation_duration:
                progress = 1.0
                self.is_preparing = False
                self.preparation_complete = True
                print(f"\n[HDF5 Delta EEF] ✓ 准备完成！按 'l' 开始回放")
            else:
                t = elapsed / self.preparation_duration
                progress = t * t * (3.0 - 2.0 * t)  # smoothstep
            
            # 插值上臂位置
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
                "wrist_pose": self.preparation_target_wrist,  # 使用第一帧的手腕位置
                "navigate_cmd": DEFAULT_NAV_CMD,
                "base_height_command": DEFAULT_BASE_HEIGHT,
            }
            
            if int(elapsed * 10) % 10 == 0:
                print(f"\r[HDF5 Delta EEF] 准备中... {elapsed:.1f}/{self.preparation_duration:.1f}s ({progress*100:.1f}%)", 
                      end="", flush=True)
            
            return cmd
        
        # 准备完成，等待开始回放：持续发送第一帧命令
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
        
        # 回放阶段
        if not self.is_replaying:
            return None
        
        ep = self.episodes[self.current_episode_idx]
        
        # 检查是否需要推进到下一帧
        now = time.time()
        if self.last_frame_time is not None and self.frame_interval is not None:
            if now - self.last_frame_time >= self.frame_interval:
                self.current_frame_idx += 1
                self.last_frame_time = now
                
                # 应用 delta 到累积 EEF（从第二帧开始应用 delta）
                # delta_eef[0] 是 0（第一帧），delta_eef[i] 是从 frame i-1 到 frame i 的增量
                if self.current_cumulative_eef is not None and self.current_frame_idx > 0:
                    if self.current_frame_idx < len(ep["delta_eef"]):
                        delta = ep["delta_eef"][self.current_frame_idx]
                        self.current_cumulative_eef = self._apply_delta_to_eef(
                            self.current_cumulative_eef, delta
                        )
        
        # 检查是否当前 episode 结束
        if self.current_frame_idx >= ep["num_frames"]:
            self.current_frame_idx = 0
            self.current_episode_idx += 1
            
            if self.current_episode_idx >= len(self.episodes):
                print("\n[HDF5 Delta EEF] ✓ 所有 episode 回放完成")
                self.is_replaying = False
                self.current_episode_idx = 0
                return None
            else:
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
                # 重置累积 EEF 为新的初始值
                self.current_cumulative_eef = ep["initial_eef"].copy()
                print(f"\n[HDF5 Delta EEF] 切换到 Episode {self.current_episode_idx + 1}/{len(self.episodes)}")
        
        # 获取当前累积的 EEF（如果还没初始化，使用初始值）
        if self.current_cumulative_eef is None:
            self.current_cumulative_eef = ep["initial_eef"].copy()
        
        # 使用 IK 计算上半身关节角度
        upper_body_joints = self._compute_upper_body_joints_from_eef(self.current_cumulative_eef)
        
        # 构建 command
        cmd = {
            "target_upper_body_pose": upper_body_joints,
            "wrist_pose": self.current_cumulative_eef,
            "navigate_cmd": ep.get("navigate_cmd", DEFAULT_NAV_CMD)[self.current_frame_idx] if "navigate_cmd" in ep else DEFAULT_NAV_CMD,
            "base_height_command": ep.get("base_height_command", DEFAULT_BASE_HEIGHT)[self.current_frame_idx] if "base_height_command" in ep else DEFAULT_BASE_HEIGHT,
        }
        
        if isinstance(cmd["base_height_command"], np.ndarray):
            cmd["base_height_command"] = cmd["base_height_command"].item() if cmd["base_height_command"].size == 1 else cmd["base_height_command"][0]
        
        # 打印帧序号和进度
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
    """打印安全警告"""
    print("\n" + "!" * 70)
    print("!" + " " * 68 + "!")
    print("!" + "  ⚠️  警告：即将在真机上回放 Delta EEF 数据（IK 版本）！".center(50) + "!")
    print("!" + " " * 68 + "!")
    print("!" * 70)
    print("""
请确保：
  1. 机器人周围无障碍物和人员
  2. 急停开关在手边且可用
  3. Delta EEF 数据的动作幅度在安全范围内
  4. 回放速度设置合理
  5. 你已熟悉如何紧急停止
  
紧急停止方式：
  - 按键盘 'o' 停用策略
  - 按物理急停按钮
  - Ctrl+C 终止程序
""")
    print("!" * 70 + "\n")


def main(config: ReplayDeltaEEFIKConfig):
    # 安全确认
    print_safety_warning()
    
    if config.require_confirmation:
        confirm = input("确认在真机上运行回放？输入 'yes' 继续: ").strip().lower()
        if confirm != "yes":
            print("已取消")
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
    print("HDF5 Delta EEF IK 回放控制循环")
    print("=" * 60)
    print(f"数据集: {config.dataset_dir}")
    print(f"回放速度: {config.speed}x")
    print("-" * 60)
    print("操作步骤:")
    print("  1. 按 ] 激活 WBC 策略")
    print("  2. 按 p 进入准备阶段（缓慢移动到第一帧位置）")
    print("  3. 等待准备完成后，按 l 开始/暂停回放")
    print("-" * 60)
    print("安全按键:")
    print("  o - 停用策略（紧急时使用）")
    print("  Ctrl+C - 退出程序")
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
                        print("\n⚠️ 关节安全警告！请检查机器人状态")
                
                # 导出数据
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
        print("\n用户中断")
    finally:
        print("\nCleaning up...")
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()
        print("已安全退出")


if __name__ == "__main__":
    config = tyro.cli(ReplayDeltaEEFIKConfig)
    main(config)

