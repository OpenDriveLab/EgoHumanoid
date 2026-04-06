"""
HDF5 数据回放器 - 基于 run_g1_control_loop.py 的结构

与 run_g1_control_loop.py 几乎相同，唯一区别是：
- 用 HDF5 数据集中的 command 替代 ROS 订阅的 teleop command

使用方式：
1. 运行脚本
2. 按 ] 激活策略
3. 按 9 释放机器人
4. 按 p 开始回放数据集

使用示例：
    python replay_hdf5_with_policy.py --dataset-dir outputs/data_1224_0
    python replay_hdf5_with_policy.py --dataset-dir outputs/data_1224_0 --episode 0
    python replay_hdf5_with_policy.py --dataset-dir outputs/data_1224_0 --speed 0.5
"""

from dataclasses import dataclass
import glob
import os
import time
from typing import Optional

import h5py
import numpy as np
import tyro

from decoupled_wbc.control.envs.g1.g1_env import G1Env
from decoupled_wbc.control.main.constants import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
)
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardEStop,
    KeyboardListenerPublisher,
)
from decoupled_wbc.control.utils.ros_utils import ROSManager
from decoupled_wbc.control.utils.telemetry import Telemetry


@dataclass
class ReplayConfig(ControlLoopConfig):
    """回放配置，继承自 ControlLoopConfig"""
    
    dataset_dir: str = "outputs/data_1224_0"
    """数据集目录路径"""
    
    episode: Optional[int] = None
    """要回放的 episode 索引，None 表示回放所有"""
    
    speed: float = 1.0
    """回放速度倍率"""
    
    loop: bool = False
    """是否循环回放"""
    
    # 默认使用仿真模式
    interface: str = "sim"


class HDF5CommandProvider:
    """
    从 HDF5 数据集提供 command，替代 ROS 订阅
    
    按 'p' 键开始回放，回放期间提供数据集中的 command
    """
    
    def __init__(self, dataset_dir: str, episode: Optional[int] = None, speed: float = 1.0):
        self.episodes = self._load_episodes(dataset_dir, episode)
        self.speed = speed
        
        # 回放状态
        self.is_replaying = False
        self.current_episode_idx = 0
        self.current_frame_idx = 0
        self.last_frame_time = None
        self.frame_interval = None
        
        print(f"[HDF5] 已加载 {len(self.episodes)} 个 episode")
        print(f"[HDF5] 按 'p' 键开始/暂停回放")
    
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
    
    def handle_keyboard_button(self, key):
        """处理键盘按键"""
        if key == "p":
            self.is_replaying = not self.is_replaying
            if self.is_replaying:
                print("\n[HDF5] ▶ 开始回放")
                self.last_frame_time = time.time()
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
            else:
                print("\n[HDF5] ⏸ 暂停回放")
    
    def get_command(self, robot_model) -> Optional[dict]:
        """
        获取当前帧的 command
        
        返回 None 表示使用默认 command（未开始回放或回放结束）
        返回 dict 表示数据集中的 command
        """
        if not self.is_replaying or len(self.episodes) == 0:
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


def main(config: ReplayConfig):
    ros_manager = ROSManager(node_name="ReplayControlLoop")
    node = ros_manager.node
    
    wbc_config = config.load_wbc_yaml()
    
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
    if env.sim and not config.sim_sync_mode:
        env.start_simulator()
    
    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)
    
    # === 关键区别：创建 HDF5 command 提供器 ===
    hdf5_provider = HDF5CommandProvider(
        dataset_dir=config.dataset_dir,
        episode=config.episode,
        speed=config.speed,
    )
    
    # 键盘调度器（与 run_g1_control_loop.py 相同）
    keyboard_listener_pub = KeyboardListenerPublisher()
    keyboard_estop = KeyboardEStop()
    dispatcher = KeyboardDispatcher()
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_listener_pub)
    dispatcher.register(keyboard_estop)
    dispatcher.register(hdf5_provider)  # 注册 HDF5 provider 以接收 'p' 键
    dispatcher.start()
    
    rate = node.create_rate(config.control_frequency)
    
    print("\n" + "=" * 60)
    print("HDF5 回放控制循环")
    print("=" * 60)
    print("操作步骤:")
    print("  1. 按 ] 激活 WBC 策略")
    print("  2. 按 9 释放机器人（禁用弹性绳索）")
    print("  3. 按 p 开始/暂停回放")
    print("-" * 60)
    print("其他按键:")
    print("  w/s/a/d/q/e - 手动控制移动")
    print("  1/2 - 调整高度")
    print("  o - 停用策略")
    print("  Ctrl+C - 退出")
    print("=" * 60 + "\n")
    
    last_teleop_cmd = None
    
    try:
        while ros_manager.ok():
            t_start = time.monotonic()
            
            with telemetry.timer("total_loop"):
                # Step simulator if in sync mode
                if env.sim and config.sim_sync_mode:
                    env.step_simulator()
                
                # 获取观测
                obs = env.observe()
                wbc_policy.set_observation(obs)
                
                t_now = time.monotonic()
                
                # === 关键区别：从 HDF5 获取 command，而非 ROS 订阅 ===
                replay_cmd = hdf5_provider.get_command(robot_model)
                
                # 构建 wbc_goal：回放时用数据集的 command，否则用默认值
                if replay_cmd:
                    # 回放模式：使用数据集中的 command
                    wbc_goal = replay_cmd.copy()
                    last_teleop_cmd = replay_cmd.copy()
                else:
                    # 待机模式：使用默认 command（速度=0，高度=0.74）
                    # 这样机器人会保持站立不动
                    upper_body_indices = robot_model.get_joint_group_indices("upper_body")
                    wbc_goal = {
                        "wrist_pose": DEFAULT_WRIST_POSE,
                        "navigate_cmd": np.array(DEFAULT_NAV_CMD),  # [0, 0, 0]
                        "base_height_command": DEFAULT_BASE_HEIGHT,  # 0.74
                        "target_upper_body_pose": obs["q"][upper_body_indices],
                    }
                
                # 关键：必须设置 target_time，否则 InterpolationPolicy 不会处理目标！
                wbc_goal["target_time"] = t_now + (1 / config.control_frequency)
                wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * (
                    1 / config.control_frequency
                )
                wbc_policy.set_goal(wbc_goal)
                
                # 获取动作
                wbc_action = wbc_policy.get_action(time=t_now)
                
                # 执行动作
                env.queue_action(wbc_action)
            
            # 检查仿真器线程是否存活
            if env.sim and (not env.sim.sim_thread or not env.sim.sim_thread.is_alive()):
                raise RuntimeError("Simulator thread is not alive")
            
            rate.sleep()
            
            end_time = time.monotonic()
            if config.verbose_timing:
                telemetry.log_timing_info(context="Replay Control Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.control_frequency) and not config.sim_sync_mode:
                telemetry.log_timing_info(context="Replay Control Loop Missed", threshold=0.001)
    
    except ros_manager.exceptions() as e:
        print(f"\nROSManager interrupted: {e}")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print("\nCleaning up...")
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()


if __name__ == "__main__":
    config = tyro.cli(ReplayConfig)
    main(config)
