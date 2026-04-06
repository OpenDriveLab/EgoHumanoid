"""
HDF5 数据回放器 - 真机版本

基于 run_g1_control_loop.py 的结构，在真机上回放 HDF5 数据集

⚠️ 警告：此脚本用于控制真实机器人，请确保：
   1. 机器人周围无障碍物
   2. 急停开关在手边
   3. 数据集速度和幅度在安全范围内

使用方式：
1. 运行脚本
2. 按 ] 激活策略
3. 按 p 开始回放数据集
4. 按 o 可以随时停用策略

使用示例：
    python replay_hdf5_real.py --dataset-dir outputs/data_1224_0
    python replay_hdf5_real.py --dataset-dir outputs/data_1224_0 --episode 0
    python replay_hdf5_real.py --dataset-dir outputs/data_1224_0 --speed 0.5
"""

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
class ReplayRealConfig(ControlLoopConfig):
    """真机回放配置，继承自 ControlLoopConfig"""
    
    dataset_dir: str = "outputs/data_1224_0"
    """数据集目录路径"""
    
    episode: Optional[int] = None
    """要回放的 episode 索引，None 表示回放所有"""
    
    speed: float = 1.0
    """回放速度倍率（真机默认 0.5x 更安全）"""
    
    loop: bool = False
    """是否循环回放"""
    
    # 真机默认设置
    interface: str = "real"
    """接口类型，真机使用 'real'"""
    
    require_confirmation: bool = True
    """是否需要确认才能开始"""
    
    preparation_duration: float = 3.0
    """准备阶段持续时间（秒），手臂缓慢移动到第一帧位置的时间"""


class HDF5CommandProvider:
    """
    从 HDF5 数据集提供 command，替代 ROS 订阅
    
    按 'p' 键进入准备阶段（缓慢移动到第一帧位置）
    按 'l' 键开始回放
    """
    
    def __init__(self, dataset_dir: str, episode: Optional[int] = None, speed: float = 1.0,
                 preparation_duration: float = 3.0):
        self.episodes = self._load_episodes(dataset_dir, episode)
        self.speed = speed
        self.preparation_duration = preparation_duration  # 准备阶段持续时间（秒）
        
        # 回放状态
        self.is_replaying = False
        self.is_preparing = False  # 准备阶段（缓慢移动到第一帧）
        self.preparation_complete = False  # 准备阶段完成，等待开始回放
        self.current_episode_idx = 0
        self.current_frame_idx = 0
        self.last_frame_time = None
        self.frame_interval = None
        
        # 准备阶段相关
        self.preparation_start_time = None
        self.preparation_start_upper_body = None  # 准备开始时的上臂位置
        self.preparation_target_upper_body = None  # 第一帧的目标上臂位置
        self.preparation_target_wrist = None  # 第一帧的目标手腕位置
        
        print(f"[HDF5] 已加载 {len(self.episodes)} 个 episode")
        print(f"[HDF5] 回放速度: {speed}x")
        print(f"[HDF5] 准备时间: {preparation_duration}s")
        print(f"[HDF5] 按 'p' 键进入准备阶段（缓慢移动到第一帧）")
        print(f"[HDF5] 按 'l' 键开始/暂停回放")
    
    def _load_episodes(self, dataset_dir: str, episode: Optional[int]) -> list:
        """加载 episode 数据"""
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
                    
                    # 加载控制指令
                    if "action_eef" in f:
                        ep["wrist_pose"] = np.array(f["action_eef"])
                    elif "observation_eef_state" in f:
                        ep["wrist_pose"] = np.array(f["observation_eef_state"])
                    
                    if "teleop_navigate_command" in f:
                        ep["navigate_cmd"] = np.array(f["teleop_navigate_command"])
                    
                    if "teleop_base_height_command" in f:
                        ep["base_height_command"] = np.array(f["teleop_base_height_command"])
                    
                    if "observation_state" in f:
                        ep["observation_state"] = np.array(f["observation_state"])
                    
                    episodes.append(ep)
                    print(f"  已加载: {ep['file']} - {ep['num_frames']} 帧, {ep['fps']} FPS")
            except Exception as e:
                print(f"  [WARN] 加载 {hdf5_file} 失败: {e}")
        
        if episode is not None:
            if episode >= len(episodes):
                raise ValueError(f"Episode {episode} 不存在，共有 {len(episodes)} 个")
            episodes = [episodes[episode]]
        
        return episodes
    
    def _interpolate_wrist_pose(self, start_pose: np.ndarray, target_pose: np.ndarray, t: float) -> np.ndarray:
        """
        插值手腕位姿（左右手各 7 维: [x, y, z, qx, qy, qz, qw]）
        
        使用 SLERP 插值四元数
        """
        result = np.zeros_like(start_pose)
        
        # 处理左右手
        for hand_idx in range(2):
            start_idx = hand_idx * 7
            end_idx = start_idx + 7
            
            # 位置（线性插值）
            start_pos = start_pose[start_idx:start_idx + 3]
            target_pos = target_pose[start_idx:start_idx + 3]
            result[start_idx:start_idx + 3] = start_pos * (1 - t) + target_pos * t
            
            # 四元数（简单线性插值后归一化，对于准备阶段足够）
            start_quat = start_pose[start_idx + 3:end_idx]  # (qx, qy, qz, qw)
            target_quat = target_pose[start_idx + 3:end_idx]
            
            # 简单的线性插值（然后归一化）
            interp_quat = start_quat * (1 - t) + target_quat * t
            # 归一化
            norm = np.linalg.norm(interp_quat)
            if norm > 1e-6:
                interp_quat = interp_quat / norm
            result[start_idx + 3:end_idx] = interp_quat
        
        return result
    
    def handle_keyboard_button(self, key):
        """处理键盘按键"""
        if key == "p":
            # 进入准备阶段
            if not self.is_preparing and not self.is_replaying:
                self.is_preparing = True
                self.preparation_complete = False  # 重置准备完成标志
                self.preparation_start_time = time.time()
                # 重置准备状态变量
                self.preparation_start_upper_body = None
                self.preparation_start_wrist = None
                self.preparation_target_upper_body = None
                self.preparation_target_wrist = None
                print("\n[HDF5] 🔄 进入准备阶段：缓慢移动到第一帧位置...")
                print(f"[HDF5] 准备时间: {self.preparation_duration}s，完成后按 'l' 开始回放")
        elif key == "l":
            # 开始/暂停回放
            if self.is_preparing:
                print("\n[HDF5] ⚠️ 准备阶段未完成，请等待完成后再按 'l'")
                return
            
            if not self.preparation_complete and not self.is_replaying:
                print("\n[HDF5] ⚠️ 请先按 'p' 完成准备阶段")
                return
            
            self.is_replaying = not self.is_replaying
            if self.is_replaying:
                self.preparation_complete = False  # 开始回放后清除准备完成标志
                print("\n[HDF5] ▶ 开始回放")
                self.current_frame_idx = 0  # 重置到第一帧
                self.last_frame_time = time.time()
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
            else:
                print("\n[HDF5] ⏸ 暂停回放")
    
    def get_command(self, robot_model, current_obs=None) -> Optional[dict]:
        """
        获取当前帧的 command
        
        Args:
            robot_model: 机器人模型
            current_obs: 当前观测（用于准备阶段获取当前手臂位置）
        
        返回 None 表示使用默认 command（未开始回放或回放结束）
        返回 dict 表示数据集中的 command 或准备阶段的插值 command
        """
        if len(self.episodes) == 0:
            return None
        
        # 准备阶段：缓慢移动到第一帧位置
        if self.is_preparing:
            if current_obs is None:
                print("[HDF5] 警告: 准备阶段需要 current_obs，但未提供")
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
                if "observation_state" in ep:
                    self.preparation_target_upper_body = ep["observation_state"][0][upper_body_indices].copy()
                else:
                    # 如果没有观测数据，使用当前位置
                    self.preparation_target_upper_body = self.preparation_start_upper_body.copy()
                
                # 记录第一帧的手腕位置
                if "wrist_pose" in ep:
                    self.preparation_target_wrist = ep["wrist_pose"][0].copy()
                else:
                    self.preparation_target_wrist = np.array(DEFAULT_WRIST_POSE).copy()
                
                # 记录起始手腕位置（从当前观测获取）
                self.preparation_start_wrist = current_obs.get("wrist_pose", np.array(DEFAULT_WRIST_POSE)).copy()
                
                print(f"[HDF5] 开始位置已记录，目标位置已设置")
            
            # 计算插值进度（使用平滑曲线）
            if elapsed >= self.preparation_duration:
                # 准备完成
                progress = 1.0
                self.is_preparing = False
                self.preparation_complete = True  # 标记准备完成
                print(f"\n[HDF5] ✓ 准备完成！按 'l' 开始回放")
            else:
                # 使用平滑插值（ease-in-out）
                t = elapsed / self.preparation_duration
                progress = t * t * (3.0 - 2.0 * t)  # smoothstep
            
            # 插值上臂位置
            upper_body_indices = robot_model.get_joint_group_indices("upper_body")
            if (self.preparation_start_upper_body is not None and 
                self.preparation_target_upper_body is not None):
                interp_upper_body = (
                    self.preparation_start_upper_body * (1 - progress) +
                    self.preparation_target_upper_body * progress
                )
            else:
                # 如果还没初始化，使用目标位置
                interp_upper_body = (
                    self.preparation_target_upper_body 
                    if self.preparation_target_upper_body is not None 
                    else current_obs["q"][upper_body_indices]
                )
            
            # 插值手腕位置（使用 SLERP 处理四元数）
            if (self.preparation_start_wrist is not None and 
                self.preparation_target_wrist is not None):
                interp_wrist = self._interpolate_wrist_pose(
                    self.preparation_start_wrist,
                    self.preparation_target_wrist,
                    progress
                )
            else:
                # 如果还没初始化，使用目标位置
                interp_wrist = self.preparation_target_wrist if self.preparation_target_wrist is not None else np.array(DEFAULT_WRIST_POSE)
            
            # 构建 command（只移动上臂和手腕，其他保持默认）
            cmd = {
                "target_upper_body_pose": interp_upper_body,
                "wrist_pose": interp_wrist,
                "navigate_cmd": DEFAULT_NAV_CMD,  # 准备阶段不移动底盘
                "base_height_command": DEFAULT_BASE_HEIGHT,
            }
            
            # 打印进度
            if int(elapsed * 10) % 10 == 0:  # 每0.1秒打印一次
                print(f"\r[HDF5] 准备中... {elapsed:.1f}/{self.preparation_duration:.1f}s ({progress*100:.1f}%)", 
                      end="", flush=True)
            
            return cmd
        
        # 准备完成，等待开始回放：持续发送第一帧命令
        if self.preparation_complete and not self.is_replaying:
            # 使用准备阶段已经设置好的目标位置
            if (self.preparation_target_upper_body is not None and 
                self.preparation_target_wrist is not None):
                cmd = {
                    "target_upper_body_pose": self.preparation_target_upper_body,
                    "wrist_pose": self.preparation_target_wrist,
                    "navigate_cmd": DEFAULT_NAV_CMD,
                    "base_height_command": DEFAULT_BASE_HEIGHT,
                }
                return cmd
            else:
                # 如果目标位置未设置（不应该发生），使用默认值
                upper_body_indices = robot_model.get_joint_group_indices("upper_body")
                cmd = {
                    "target_upper_body_pose": current_obs["q"][upper_body_indices] if current_obs else robot_model.get_default_q()[upper_body_indices],
                    "wrist_pose": DEFAULT_WRIST_POSE,
                    "navigate_cmd": DEFAULT_NAV_CMD,
                    "base_height_command": DEFAULT_BASE_HEIGHT,
                }
                return cmd
        
        # 回放阶段
        if not self.is_replaying:
            return None
        
        # 检查是否需要推进到下一帧
        now = time.time()
        if self.last_frame_time is not None and self.frame_interval is not None:
            if now - self.last_frame_time < self.frame_interval:
                # 还没到下一帧的时间，返回当前帧的 command
                pass
            else:
                # 推进到下一帧
                self.current_frame_idx += 1
                self.last_frame_time = now
        
        ep = self.episodes[self.current_episode_idx]
        
        # 检查是否当前 episode 结束
        if self.current_frame_idx >= ep["num_frames"]:
            self.current_frame_idx = 0
            self.current_episode_idx += 1
            
            if self.current_episode_idx >= len(self.episodes):
                # 所有 episode 结束
                print("\n[HDF5] ✓ 所有 episode 回放完成")
                self.is_replaying = False
                self.current_episode_idx = 0
                return None
            else:
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
                print(f"\n[HDF5] 切换到 Episode {self.current_episode_idx + 1}/{len(self.episodes)}")
        
        # 构建 command
        frame_idx = self.current_frame_idx
        cmd = {}
        
        if "wrist_pose" in ep:
            cmd["wrist_pose"] = ep["wrist_pose"][frame_idx]
        else:
            cmd["wrist_pose"] = DEFAULT_WRIST_POSE
        
        if "navigate_cmd" in ep:
            cmd["navigate_cmd"] = ep["navigate_cmd"][frame_idx]
        else:
            cmd["navigate_cmd"] = DEFAULT_NAV_CMD
        
        if "base_height_command" in ep:
            height = ep["base_height_command"][frame_idx]
            if isinstance(height, np.ndarray):
                height = height.item() if height.size == 1 else height[0]
            cmd["base_height_command"] = height
        else:
            cmd["base_height_command"] = DEFAULT_BASE_HEIGHT
        
        if "observation_state" in ep:
            upper_body_indices = robot_model.get_joint_group_indices("upper_body")
            cmd["target_upper_body_pose"] = ep["observation_state"][frame_idx][upper_body_indices]
        
        # 打印进度
        if frame_idx % 20 == 0:
            progress = (frame_idx + 1) / ep["num_frames"] * 100
            nav = cmd["navigate_cmd"]
            print(
                f"\r[HDF5] Ep {self.current_episode_idx + 1}/{len(self.episodes)} | "
                f"Frame {frame_idx + 1}/{ep['num_frames']} ({progress:.1f}%) | "
                f"Nav: [{nav[0]:+.2f}, {nav[1]:+.2f}, {nav[2]:+.2f}]",
                end="", flush=True
            )
        
        return cmd


def print_safety_warning():
    """打印安全警告"""
    print("\n" + "!" * 70)
    print("!" + " " * 68 + "!")
    print("!" + "  ⚠️  警告：即将在真机上回放数据！".center(58) + "!")
    print("!" + " " * 68 + "!")
    print("!" * 70)
    print("""
请确保：
  1. 机器人周围无障碍物和人员
  2. 急停开关在手边且可用
  3. 数据集的动作幅度在安全范围内
  4. 回放速度设置合理（建议先用 0.5x）
  5. 你已熟悉如何紧急停止
  
紧急停止方式：
  - 按键盘 'o' 停用策略
  - 按物理急停按钮
  - Ctrl+C 终止程序
""")
    print("!" * 70 + "\n")


def main(config: ReplayRealConfig):
    # 安全确认
    print_safety_warning()
    
    if config.require_confirmation:
        confirm = input("确认在真机上运行回放？输入 'yes' 继续: ").strip().lower()
        if confirm != "yes":
            print("已取消")
            return
    
    ros_manager = ROSManager(node_name="ReplayRealControlLoop")
    node = ros_manager.node
    
    # 启动配置服务器
    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())
    
    wbc_config = config.load_wbc_yaml()
    
    # 状态发布器（用于数据采集和监控）
    data_exp_pub = ROSMsgPublisher(STATE_TOPIC_NAME)
    lower_body_policy_status_pub = ROSMsgPublisher(LOWER_BODY_POLICY_STATUS_TOPIC)
    joint_safety_status_pub = ROSMsgPublisher(JOINT_SAFETY_STATUS_TOPIC)
    
    telemetry = Telemetry(window_size=100)
    
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )
    
    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    
    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)
    
    # HDF5 command 提供器
    hdf5_provider = HDF5CommandProvider(
        dataset_dir=config.dataset_dir,
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
    dispatcher.register(hdf5_provider)
    dispatcher.start()
    
    rate = node.create_rate(config.control_frequency)
    
    print("\n" + "=" * 60)
    print("HDF5 真机回放控制循环")
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
                
                # 获取 command（传入当前观测用于准备阶段）
                with telemetry.timer("get_command"):
                    replay_cmd = hdf5_provider.get_command(robot_model, current_obs=obs)
                
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
                
                # 发布状态（与 run_g1_control_loop.py 相同）
                with telemetry.timer("publish_status"):
                    # 策略状态
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
                    
                    # 关节安全状态
                    joint_safety_ok = env.get_joint_safety_status()
                    joint_safety_status_msg = {
                        "joint_safety_ok": joint_safety_ok,
                        "timestamp": t_now,
                    }
                    joint_safety_status_pub.publish(joint_safety_status_msg)
                    
                    # 安全警告
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
            
            # 时间日志
            if config.verbose_timing:
                telemetry.log_timing_info(context="Replay Real Control Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.control_frequency):
                telemetry.log_timing_info(context="Replay Real Control Loop Missed", threshold=0.001)
    
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
    config = tyro.cli(ReplayRealConfig)
    main(config)

