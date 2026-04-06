import time
from collections import deque

import numpy as np
import rclpy
import tyro

from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC
from decoupled_wbc.control.main.teleop.configs.configs import TeleopConfig
from decoupled_wbc.control.policy.lerobot_replay_policy import LerobotReplayPolicy
from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from decoupled_wbc.control.utils.telemetry import Telemetry

TELEOP_NODE_NAME = "TeleopPolicy"


class VRDataDebugger:
    """VR数据调试器 - 用于检查VR指令接收情况"""
    
    def __init__(self, enable_debug=True, print_interval=1.0):
        """
        Args:
            enable_debug: 是否启用调试输出
            print_interval: 打印间隔（秒）
        """
        self.enable_debug = enable_debug
        self.print_interval = print_interval
        self.last_print_time = 0.0
        
        # 数据统计
        self.data_received_count = 0
        self.last_data = None
        self.data_change_count = 0
        
        # 数据有效性检查
        self.wrist_pos_history = {"left": deque(maxlen=10), "right": deque(maxlen=10)}
        self.nav_cmd_history = deque(maxlen=10)
        
    def check_data(self, data: dict, iteration: int):
        """检查并显示VR数据
        
        Args:
            data: teleop_policy返回的数据字典
            iteration: 当前迭代次数
        """
        if not self.enable_debug:
            return
        
        self.data_received_count += 1
        t_now = time.monotonic()
        
        # 检查数据是否发生变化
        data_changed = False
        if self.last_data is not None:
            data_changed = self._check_data_changed(data, self.last_data)
            if data_changed:
                self.data_change_count += 1
        
        self.last_data = data.copy() if data else None
        
        # 定期打印详细信息
        if t_now - self.last_print_time >= self.print_interval:
            self._print_debug_info(data, iteration, data_changed)
            self.last_print_time = t_now
    
    def _check_data_changed(self, current: dict, last: dict) -> bool:
        """检查数据是否发生变化"""
        # 检查关键字段
        key_fields = ["wrist_pose", "navigate_cmd", "base_height_command", "target_upper_body_pose"]
        
        for key in key_fields:
            if key in current and key in last:
                curr_val = np.array(current[key])
                last_val = np.array(last[key])
                if not np.allclose(curr_val, last_val, atol=1e-4):
                    return True
        
        return False
    
    def _print_debug_info(self, data: dict, iteration: int, data_changed: bool):
        """打印调试信息"""
        print("\n" + "="*60)
        print(f"[VR调试] 迭代次数: {iteration} | 接收数据次数: {self.data_received_count}")
        print(f"[VR调试] 数据变化次数: {self.data_change_count} | 本次数据变化: {'是' if data_changed else '否'}")
        print("-"*60)
        
        if data is None:
            print("[警告] 未接收到任何数据！")
            return
        
        # 检查teleop_policy激活状态
        if hasattr(data, 'get'):
            is_active = getattr(data, 'is_active', None)
            if is_active is not None:
                print(f"[状态] Teleop Policy 激活状态: {'已激活' if is_active else '未激活'}")
        
        # 检查手腕位置
        if "wrist_pose" in data:
            wrist_pose = np.array(data["wrist_pose"])
            if len(wrist_pose) >= 14:
                left_pos = wrist_pose[0:3]
                right_pos = wrist_pose[7:10]
                print(f"[手腕位置] 左手: [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]")
                print(f"[手腕位置] 右手: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")
                
                # 检查位置是否有效（非零且不在原点）
                left_valid = not np.allclose(left_pos, 0, atol=0.01)
                right_valid = not np.allclose(right_pos, 0, atol=0.01)
                print(f"[有效性] 左手位置: {'有效' if left_valid else '无效（接近原点）'}")
                print(f"[有效性] 右手位置: {'有效' if right_valid else '无效（接近原点）'}")
            else:
                print(f"[警告] wrist_pose长度异常: {len(wrist_pose)} (期望14)")
        else:
            print("[警告] 缺少 wrist_pose 字段")
        
        # 检查手指数据
        left_fingers_present = "left_fingers" in data
        right_fingers_present = "right_fingers" in data
        print(f"[手指数据] 左手: {'存在' if left_fingers_present else '缺失'} | 右手: {'存在' if right_fingers_present else '缺失'}")
        
        if left_fingers_present and isinstance(data["left_fingers"], dict):
            if "position" in data["left_fingers"]:
                left_fingers = np.array(data["left_fingers"]["position"])
                # 检查是否有手指关闭（扳机按下）
                index_finger = left_fingers[4 + 5, 0, 3] if left_fingers.shape[0] > 9 else 0
                print(f"[手指状态] 左手食指: {'关闭' if index_finger > 0.5 else '打开'}")
        
        if right_fingers_present and isinstance(data["right_fingers"], dict):
            if "position" in data["right_fingers"]:
                right_fingers = np.array(data["right_fingers"]["position"])
                index_finger = right_fingers[4 + 5, 0, 3] if right_fingers.shape[0] > 9 else 0
                print(f"[手指状态] 右手食指: {'关闭' if index_finger > 0.5 else '打开'}")
        
        # 检查导航命令
        if "navigate_cmd" in data:
            nav_cmd = np.array(data["navigate_cmd"])
            if len(nav_cmd) >= 3:
                print(f"[导航命令] 线性速度: [{nav_cmd[0]:.3f}, {nav_cmd[1]:.3f}] | 角速度: {nav_cmd[2]:.3f}")
                nav_active = not np.allclose(nav_cmd, 0, atol=0.01)
                print(f"[有效性] 导航命令: {'活跃' if nav_active else '静止'}")
            else:
                print(f"[警告] navigate_cmd长度异常: {len(nav_cmd)} (期望3)")
        else:
            print("[警告] 缺少 navigate_cmd 字段")
        
        # 检查底盘高度命令
        if "base_height_command" in data:
            height_cmd = data["base_height_command"]
            if isinstance(height_cmd, (list, np.ndarray)):
                height_val = height_cmd[0] if len(height_cmd) > 0 else height_cmd
            else:
                height_val = height_cmd
            print(f"[底盘高度] 命令值: {height_val:.3f}")
        else:
            print("[警告] 缺少 base_height_command 字段")
        
        # 检查目标关节角度
        if "target_upper_body_pose" in data:
            target_pose = np.array(data["target_upper_body_pose"])
            print(f"[目标关节] 关节数量: {len(target_pose)}")
            if len(target_pose) > 0:
                print(f"[目标关节] 范围: [{target_pose.min():.3f}, {target_pose.max():.3f}]")
                print(f"[目标关节] 均值: {target_pose.mean():.3f}")
        else:
            print("[警告] 缺少 target_upper_body_pose 字段")
        
        # 检查时间戳
        if "timestamp" in data:
            print(f"[时间戳] {data['timestamp']:.6f}")
        
        # 检查数据源
        if hasattr(data, 'get'):
            source = data.get('source', 'unknown')
            print(f"[数据源] {source}")
        
        print("="*60 + "\n")
    
    def print_summary(self):
        """打印统计摘要"""
        if not self.enable_debug:
            return
        
        print("\n" + "="*60)
        print("[VR调试摘要]")
        print(f"  总接收数据次数: {self.data_received_count}")
        print(f"  数据变化次数: {self.data_change_count}")
        if self.data_received_count > 0:
            change_rate = self.data_change_count / self.data_received_count * 100
            print(f"  数据变化率: {change_rate:.2f}%")
        print("="*60 + "\n")


def main(config: TeleopConfig):
    ros_manager = ROSManager(node_name=TELEOP_NODE_NAME)
    node = ros_manager.node

    if config.robot == "g1":
        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        robot_model = instantiate_g1_robot_model(
            waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
        )
        left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
    else:
        raise ValueError(f"Unsupported robot name: {config.robot}")

    if config.lerobot_replay_path:
        teleop_policy = LerobotReplayPolicy(
            robot_model=robot_model, parquet_path=config.lerobot_replay_path
        )
    else:
        print("running teleop policy, waiting teleop policy to be initialized...")
        retargeting_ik = TeleopRetargetingIK(
            robot_model=robot_model,
            left_hand_ik_solver=left_hand_ik_solver,
            right_hand_ik_solver=right_hand_ik_solver,
            enable_visualization=config.enable_visualization,
            body_active_joint_groups=["upper_body"],
        )
        teleop_policy = TeleopPolicy(
            robot_model=robot_model,
            retargeting_ik=retargeting_ik,
            body_control_device=config.body_control_device,
            hand_control_device=config.hand_control_device,
            body_streamer_ip=config.body_streamer_ip,  # vive tracker, leap motion does not require
            body_streamer_keyword=config.body_streamer_keyword,
            enable_real_device=config.enable_real_device,
            replay_data_path=config.teleop_replay_path,
        )

    # Create a publisher for the navigation commands
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)

    # Create rate controller
    rate = node.create_rate(config.teleop_frequency)
    iteration = 0
    time_to_get_to_initial_pose = 2  # seconds

    telemetry = Telemetry(window_size=100)
    
    # 初始化VR数据调试器
    # 可以通过环境变量或配置来控制调试输出
    import os
    enable_debug = os.getenv("VR_DEBUG", "true").lower() == "true"
    debug_print_interval = float(os.getenv("VR_DEBUG_INTERVAL", "1.0"))  # 默认每秒打印一次
    vr_debugger = VRDataDebugger(enable_debug=enable_debug, print_interval=debug_print_interval)
    
    print("\n" + "="*60)
    print("[VR调试] 调试模式已启用")
    print(f"[VR调试] 打印间隔: {debug_print_interval}秒")
    print(f"[VR调试] 控制频率: {config.teleop_frequency}Hz")
    print("="*60 + "\n")

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                # Get the current teleop action
                with telemetry.timer("get_action"):
                    data = teleop_policy.get_action()
                    
                    # 检查VR数据
                    vr_debugger.check_data(data, iteration)
                
                # Add timing information to the message
                t_now = time.monotonic()
                if data:
                    data["timestamp"] = t_now

                    # Set target completion time - longer for initial pose, then match control frequency
                    if iteration == 0:
                        data["target_time"] = t_now + time_to_get_to_initial_pose
                    else:
                        data["target_time"] = t_now + (1 / config.teleop_frequency)

                    # Publish the teleop command
                    with telemetry.timer("publish_teleop_command"):
                        control_publisher.publish(data)
                else:
                    print("[警告] teleop_policy.get_action() 返回了 None 或空数据！")

                # For the initial pose, wait the full duration before continuing
                if iteration == 0:
                    print(f"Moving to initial pose for {time_to_get_to_initial_pose} seconds")
                    time.sleep(time_to_get_to_initial_pose)
                iteration += 1
            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.teleop_frequency):
                telemetry.log_timing_info(context="Teleop Policy Loop Missed", threshold=0.001)
            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")

    finally:
        print("Cleaning up...")
        # 打印调试摘要
        vr_debugger.print_summary()
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(TeleopConfig)
    main(config)
