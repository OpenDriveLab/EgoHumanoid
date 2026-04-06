"""
G1 数据采集器 - HDF5 格式
使用 HDF5 格式保存数据，每帧带有 time.time() 时间戳
"""
from collections import deque
from datetime import datetime
import os
import socket
import threading
import time

import h5py
import numpy as np
import rclpy
import tyro

import sys
sys.path.insert(0, '/root/Projects/GR00T-WholeBodyControl')
from decoupled_wbc.control.main.constants import ROBOT_CONFIG_TOPIC, STATE_TOPIC_NAME, CONTROL_GOAL_TOPIC
from decoupled_wbc.control.main.teleop.configs.configs import DataExporterConfig
from decoupled_wbc.control.robot_model.instantiation import g1
from decoupled_wbc.control.utils.episode_state import EpisodeState
from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber
from decoupled_wbc.control.utils.ros_utils import ROSMsgSubscriber, ROSServiceClient
from decoupled_wbc.control.utils.telemetry import Telemetry
from decoupled_wbc.control.utils.text_to_speech import TextToSpeech


class HDF5DataExporter:
    """
    HDF5 数据导出器 - 每个 episode 保存为单独的文件
    
    数据结构:
    dataset_dir/
    ├── episode_0.hdf5
    │   ├── timestamp (N,) - time.time() 时间戳
    │   ├── observation_state (N, num_joints) - 关节角度
    │   ├── observation_eef_state (N, 14) - 末端执行器位姿
    │   ├── action (N, num_joints) - 动作
    │   ├── action_eef (N, 14) - 末端执行器动作
    │   ├── teleop_navigate_command (N, 3) - 导航命令
    │   ├── teleop_base_height_command (N, 1) - 底盘高度命令
    │   ├── hand_status (N, 2) - 手部状态（扳机状态）：[左手状态, 右手状态]，按下扳机为1，否则为0
    │   └── metadata (属性)
    │       ├── task_prompt
    │       ├── robot_id
    │       ├── fps
    │       └── joint_names
    ├── episode_1.hdf5
    │   └── ...
    └── episode_N.hdf5
    """
    
    def __init__(
        self,
        save_dir: str,
        task_prompt: str = "",
        robot_id: str = "sim",
        fps: int = 20,
        joint_names: list = None,
    ):
        self.save_dir = save_dir
        self.task_prompt = task_prompt
        self.robot_id = robot_id
        self.fps = fps
        self.joint_names = joint_names if joint_names is not None else []
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # Episode 缓冲区
        self.episode_buffer = {
            "episode_index": 0,
            "size": 0,
            "timestamp": [],
            "observation_state": [],
            "observation_eef_state": [],
            "action": [],
            "action_eef": [],
            "teleop_navigate_command": [],
            "teleop_base_height_command": [],
            "hand_status": [],
        }
        
        # 标志：是否被B键中断（需要保持index）
        self.interrupted_by_b_button = False
        
        # 扫描目录，找到下一个 episode 索引
        self._init_episode_index()
        
    def _init_episode_index(self):
        """扫描目录，找到下一个可用的 episode 索引"""
        existing_episodes = []
        for filename in os.listdir(self.save_dir):
            if filename.startswith("episode_") and filename.endswith(".hdf5"):
                try:
                    # 从文件名提取索引: episode_X.hdf5 -> X
                    index = int(filename.replace("episode_", "").replace(".hdf5", ""))
                    existing_episodes.append(index)
                except ValueError:
                    continue
        
        if existing_episodes:
            self.episode_buffer["episode_index"] = max(existing_episodes) + 1
            print(f"继续添加到现有数据集，当前 episode 索引: {self.episode_buffer['episode_index']}")
        else:
            print(f"创建新数据集目录: {self.save_dir}")
    
    def add_frame(self, frame_data: dict):
        """添加一帧数据到缓冲区"""
        # 添加时间戳
        self.episode_buffer["timestamp"].append(time.time())
        
        # 添加各个数据字段
        self.episode_buffer["observation_state"].append(
            np.array(frame_data["observation.state"], dtype=np.float64)
        )
        self.episode_buffer["observation_eef_state"].append(
            np.array(frame_data["observation.eef_state"], dtype=np.float64)
        )
        self.episode_buffer["action"].append(
            np.array(frame_data["action"], dtype=np.float64)
        )
        self.episode_buffer["action_eef"].append(
            np.array(frame_data["action.eef"], dtype=np.float64)
        )
        self.episode_buffer["teleop_navigate_command"].append(
            np.array(frame_data["teleop.navigate_command"], dtype=np.float64)
        )
        self.episode_buffer["teleop_base_height_command"].append(
            np.array(frame_data["teleop.base_height_command"], dtype=np.float64)
        )
        # 添加手部状态（扳机状态）：[左手状态, 右手状态]，按下扳机为1，否则为0
        hand_status = frame_data.get("hand_status", np.array([0, 0], dtype=np.int32))
        # 确保hand_status是shape为(2,)的数组
        if isinstance(hand_status, (int, float)):
            hand_status = np.array([hand_status, hand_status], dtype=np.int32)
        elif isinstance(hand_status, (list, tuple)):
            if len(hand_status) == 1:
                hand_status = np.array([hand_status[0], hand_status[0]], dtype=np.int32)
            else:
                hand_status = np.array(hand_status[:2], dtype=np.int32)
        elif isinstance(hand_status, np.ndarray):
            if hand_status.shape == ():
                hand_status = np.array([hand_status.item(), hand_status.item()], dtype=np.int32)
            elif len(hand_status.shape) == 1 and hand_status.shape[0] == 1:
                hand_status = np.array([hand_status[0], hand_status[0]], dtype=np.int32)
            elif len(hand_status.shape) == 1 and hand_status.shape[0] >= 2:
                hand_status = np.array([hand_status[0], hand_status[1]], dtype=np.int32)
        self.episode_buffer["hand_status"].append(hand_status)
        
        self.episode_buffer["size"] += 1
    
    def save_episode(self):
        """保存当前 episode 到单独的 HDF5 文件"""
        if self.episode_buffer["size"] == 0:
            print("警告: Episode 为空，跳过保存")
            return
        
        episode_index = self.episode_buffer['episode_index']
        episode_filename = f"episode_{episode_index}.hdf5"
        episode_path = os.path.join(self.save_dir, episode_filename)
        
        with h5py.File(episode_path, "w") as f:
            # 保存时间戳
            f.create_dataset(
                "timestamp",
                data=np.array(self.episode_buffer["timestamp"], dtype=np.float64),
                compression="gzip"
            )
            
            # 保存观测状态
            f.create_dataset(
                "observation_state",
                data=np.stack(self.episode_buffer["observation_state"]),
                compression="gzip"
            )
            
            # 保存末端执行器状态
            f.create_dataset(
                "observation_eef_state",
                data=np.stack(self.episode_buffer["observation_eef_state"]),
                compression="gzip"
            )
            
            # 保存动作
            f.create_dataset(
                "action",
                data=np.stack(self.episode_buffer["action"]),
                compression="gzip"
            )
            
            # 保存末端执行器动作
            f.create_dataset(
                "action_eef",
                data=np.stack(self.episode_buffer["action_eef"]),
                compression="gzip"
            )
            
            # 保存导航命令
            f.create_dataset(
                "teleop_navigate_command",
                data=np.stack(self.episode_buffer["teleop_navigate_command"]),
                compression="gzip"
            )
            
            # 保存底盘高度命令
            f.create_dataset(
                "teleop_base_height_command",
                data=np.stack(self.episode_buffer["teleop_base_height_command"]),
                compression="gzip"
            )
            
            # 保存手部状态（扳机状态）：[左手状态, 右手状态]
            hand_status_data = np.stack(self.episode_buffer["hand_status"])
            # 确保shape为(N, 2)
            if len(hand_status_data.shape) == 1:
                # 如果只有一维，reshape为(N, 1)然后扩展为(N, 2)
                hand_status_data = hand_status_data.reshape(-1, 1)
                hand_status_data = np.repeat(hand_status_data, 2, axis=1)
            elif hand_status_data.shape[1] == 1:
                # 如果是(N, 1)，扩展为(N, 2)
                hand_status_data = np.repeat(hand_status_data, 2, axis=1)
            f.create_dataset(
                "hand_status",
                data=hand_status_data,
                compression="gzip"
            )
            
            # 添加元数据属性
            f.attrs["episode_index"] = episode_index
            f.attrs["num_frames"] = self.episode_buffer["size"]
            f.attrs["start_time"] = self.episode_buffer["timestamp"][0]
            f.attrs["end_time"] = self.episode_buffer["timestamp"][-1]
            f.attrs["duration"] = (
                self.episode_buffer["timestamp"][-1] - self.episode_buffer["timestamp"][0]
            )
            f.attrs["task_prompt"] = self.task_prompt
            f.attrs["robot_id"] = self.robot_id
            f.attrs["fps"] = self.fps
            f.attrs["creation_time"] = datetime.now().isoformat()
            if self.joint_names:
                f.attrs["joint_names"] = self.joint_names
        
        duration = self.episode_buffer["timestamp"][-1] - self.episode_buffer["timestamp"][0]
        print(f"已保存 {episode_filename}: {self.episode_buffer['size']} 帧, 时长 {duration:.2f}s")
        
        # 重置缓冲区
        self._reset_buffer()
    
    def save_episode_as_discarded(self, keep_index=False):
        """丢弃当前 episode
        
        Args:
            keep_index: 如果为True，保持episode_index不变（用于B键中断的情况）
        """
        print(f"丢弃 episode_{self.episode_buffer['episode_index']}: {self.episode_buffer['size']} 帧")
        self._reset_buffer(keep_index=keep_index)
    
    def _reset_buffer(self, keep_index=False):
        """重置缓冲区
        
        Args:
            keep_index: 如果为True，保持episode_index不变
        """
        if not keep_index:
            self.episode_buffer["episode_index"] += 1
        self.episode_buffer["size"] = 0
        self.episode_buffer["timestamp"] = []
        self.episode_buffer["observation_state"] = []
        self.episode_buffer["observation_eef_state"] = []
        self.episode_buffer["action"] = []
        self.episode_buffer["action_eef"] = []
        self.episode_buffer["teleop_navigate_command"] = []
        self.episode_buffer["teleop_base_height_command"] = []
        self.episode_buffer["hand_status"] = []


class Gr00tDataCollector:
    def __init__(
        self,
        node,
        state_topic_name: str,
        data_exporter: HDF5DataExporter,
        text_to_speech=None,
        frequency=20,
        zed_camera_host=None,
        zed_camera_port=8888,
    ):
        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.data_exporter = data_exporter

        self.node = node

        thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        thread.start()
        time.sleep(0.5)

        self._episode_state = EpisodeState()
        self._keyboard_listener = KeyboardListenerSubscriber()
        self._state_subscriber = ROSMsgSubscriber(state_topic_name)
        # 订阅teleop命令以获取手指数据，用于推断扳机状态
        self._teleop_cmd_subscriber = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)
        self.rate = self.node.create_rate(self.frequency)

        self.latest_proprio_msg = None
        self.latest_teleop_cmd = None

        self.telemetry = Telemetry(window_size=100)

        # ZED 相机 TCP 客户端
        self.zed_camera_host = zed_camera_host
        self.zed_camera_port = zed_camera_port
        self.zed_camera_socket = None
        if self.zed_camera_host:
            self._connect_zed_camera()

        print(f"Recording to {self.data_exporter.save_dir}")
    
    def _connect_zed_camera(self):
        """连接到 ZED 相机服务器"""
        try:
            self.zed_camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.zed_camera_socket.connect((self.zed_camera_host, self.zed_camera_port))
            print(f"[INFO] 已连接到 ZED 相机服务器: {self.zed_camera_host}:{self.zed_camera_port}")
        except Exception as e:
            print(f"[WARN] 无法连接到 ZED 相机服务器 {self.zed_camera_host}:{self.zed_camera_port}: {e}")
            self.zed_camera_socket = None
    
    def _send_zed_command(self, command: str):
        """向 ZED 相机服务器发送命令"""
        if self.zed_camera_socket is None:
            return
        
        try:
            self.zed_camera_socket.send(command.encode('utf-8'))
            response = self.zed_camera_socket.recv(1024).decode('utf-8').strip()
            print(f"[INFO] ZED 相机响应: {response}")
            return response
        except Exception as e:
            print(f"[WARN] 发送 ZED 相机命令失败: {e}")
            # 尝试重新连接
            if self.zed_camera_host:
                self._connect_zed_camera()

    @property
    def current_episode_index(self):
        return self.data_exporter.episode_buffer["episode_index"]

    def _print_and_say(self, message: str, say: bool = True):
        """Helper to use TextToSpeech print_and_say or fallback to print."""
        if self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say)
        else:
            print(message)

    def _check_keyboard_input(self):
        key = self._keyboard_listener.read_msg()
        if key == "c":
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                episode_index = self.current_episode_index
                # 如果之前被B键中断，保持index不变
                if self.data_exporter.interrupted_by_b_button:
                    self.data_exporter.interrupted_by_b_button = False
                    self._print_and_say(f"恢复录制 episode {episode_index} (保持index不变)")
                else:
                    self._print_and_say(f"Started recording episode {episode_index}")
                # 向 ZED 相机发送开始采集命令
                self._send_zed_command(f"start_{episode_index}")
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._print_and_say("Stopping recording, preparing to save")
                # 向 ZED 相机发送停止采集命令
                self._send_zed_command("STOP")
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state")
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                # B键中断：丢弃当前episode，保持index不变，等待下次A键恢复
                self.data_exporter.save_episode_as_discarded(keep_index=True)
                self.data_exporter.interrupted_by_b_button = True
                self._episode_state.reset_state()
                self._print_and_say("B键中断：已丢弃当前episode，等待下次A键恢复（保持index不变）")
                # 向 ZED 相机发送丢弃指令
                self._send_zed_command("DISCARD")
                # 向 ZED 相机发送停止采集命令
                self._send_zed_command("STOP")

    def _get_hand_status_from_fingers(self, left_fingers=None, right_fingers=None):
        """从手指数据推断扳机状态
        
        Args:
            left_fingers: 左手手指数据，shape为(25, 4, 4)的numpy数组
            right_fingers: 右手手指数据，shape为(25, 4, 4)的numpy数组
            
        Returns:
            np.ndarray: shape为(2,)的数组，[左手状态, 右手状态]，每个值为0或1
                       1表示按下扳机，0表示未按下
        """
        # 根据pico_streamer的逻辑：
        # - 只按下扳机（trigger > 0.5 且 grip <= 0.5）时，食指关闭：fingertips[4 + index, 0, 3] = 1.0
        # - 同时按下扳机和握把（trigger > 0.5 且 grip > 0.5）时，中指关闭：fingertips[4 + middle, 0, 3] = 1.0
        # 所以检查食指或中指是否关闭来判断扳机是否按下
        
        index = 5  # 食指索引
        middle = 10  # 中指索引
        
        left_status = 0
        right_status = 0
        
        # 检查左手是否按下扳机
        if left_fingers is not None:
            try:
                left_index_closed = left_fingers[4 + index, 0, 3] > 0.5  # 食指关闭
                left_middle_closed = left_fingers[4 + middle, 0, 3] > 0.5  # 中指关闭
                if left_index_closed or left_middle_closed:
                    left_status = 1
            except (IndexError, TypeError):
                left_status = 0
        
        # 检查右手是否按下扳机
        if right_fingers is not None:
            try:
                right_index_closed = right_fingers[4 + index, 0, 3] > 0.5  # 食指关闭
                right_middle_closed = right_fingers[4 + middle, 0, 3] > 0.5  # 中指关闭
                if right_index_closed or right_middle_closed:
                    right_status = 1
            except (IndexError, TypeError):
                right_status = 0
        
        return np.array([left_status, right_status], dtype=np.int32)

    def _add_data_frame(self):
        t_start = time.monotonic()

        if self.latest_proprio_msg is None:
            self._print_and_say(
                f"Waiting for proprio message...",
                say=False,
            )
            return False

        # 获取最新的teleop命令以获取手指数据
        teleop_cmd = self._teleop_cmd_subscriber.get_msg()
        if teleop_cmd is not None:
            self.latest_teleop_cmd = teleop_cmd

        if self._episode_state.get_state() == self._episode_state.RECORDING:
            # 从teleop命令中获取手指数据，推断扳机状态
            hand_status = np.array([0, 0], dtype=np.int32)  # [左手状态, 右手状态]
            if self.latest_teleop_cmd is not None:
                # 尝试从teleop命令中获取手指数据
                # 注意：teleop命令可能包含left_fingers和right_fingers，但格式可能不同
                # 这里需要根据实际的数据结构来调整
                left_fingers = None
                right_fingers = None
                
                # 检查是否有手指数据（可能在ik_data或其他字段中）
                if "left_fingers" in self.latest_teleop_cmd:
                    left_fingers = self.latest_teleop_cmd["left_fingers"]
                    if isinstance(left_fingers, dict) and "position" in left_fingers:
                        left_fingers = np.array(left_fingers["position"])
                
                if "right_fingers" in self.latest_teleop_cmd:
                    right_fingers = self.latest_teleop_cmd["right_fingers"]
                    if isinstance(right_fingers, dict) and "position" in right_fingers:
                        right_fingers = np.array(right_fingers["position"])
                
                # 如果状态消息中有手指数据，也尝试使用
                if left_fingers is None and "left_fingers" in self.latest_proprio_msg:
                    left_fingers_data = self.latest_proprio_msg["left_fingers"]
                    if isinstance(left_fingers_data, dict) and "position" in left_fingers_data:
                        left_fingers = np.array(left_fingers_data["position"])
                
                if right_fingers is None and "right_fingers" in self.latest_proprio_msg:
                    right_fingers_data = self.latest_proprio_msg["right_fingers"]
                    if isinstance(right_fingers_data, dict) and "position" in right_fingers_data:
                        right_fingers = np.array(right_fingers_data["position"])
                
                hand_status = self._get_hand_status_from_fingers(left_fingers, right_fingers)
            
            frame_data = {
                "observation.state": self.latest_proprio_msg["q"],
                "observation.eef_state": self.latest_proprio_msg["wrist_pose"],
                "action": self.latest_proprio_msg["action"],
                "action.eef": self.latest_proprio_msg["action.eef"],
                "teleop.navigate_command": np.array(
                    self.latest_proprio_msg["navigate_command"], dtype=np.float64
                ),
                "teleop.base_height_command": np.array(
                    [self.latest_proprio_msg["base_height_command"]], dtype=np.float64
                ),
                "hand_status": hand_status,
            }

            self.data_exporter.add_frame(frame_data)

        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start:.4f} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            self.data_exporter.save_episode()
            self._print_and_say("Finished saving episode")
            # 确保 ZED 相机已停止（可能在键盘输入时已发送，但这里再次确保）
            self._send_zed_command("STOP")
            self._episode_state.change_state()

        return True

    def save_and_cleanup(self):
        try:
            # 保存正在进行的 episode（如果有）
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
            self._print_and_say(f"Recording complete: {self.data_exporter.save_dir}", say=False)
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}")

        # 向 ZED 相机发送退出命令
        self._send_zed_command("QUIT")
        
        # 关闭 ZED 相机连接
        if self.zed_camera_socket:
            try:
                self.zed_camera_socket.close()
            except Exception as e:
                print(f"[WARN] 关闭 ZED 相机连接时出错: {e}")

        self.node.destroy_node()
        rclpy.shutdown()
        self._print_and_say("Shutting down data exporter...", say=False)

    def run(self):
        try:
            while rclpy.ok():
                t_start = time.monotonic()

                with self.telemetry.timer("total_loop"):
                    # 1. poll proprio msg
                    with self.telemetry.timer("poll_state"):
                        msg = self._state_subscriber.get_msg()
                        if msg is not None:
                            self.latest_proprio_msg = msg

                    # 2. check keyboard input
                    with self.telemetry.timer("check_keyboard"):
                        self._check_keyboard_input()

                    # 3. add frame
                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    end_time = time.monotonic()

                self.rate.sleep()

                # Log timing information if we missed our target frequency
                # if (end_time - t_start) > (1 / self.frequency):
                #     self.telemetry.log_timing_info(
                #         context="Data Exporter Loop Missed", threshold=0.001
                #     )

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            # 用户触发键盘中断，将正在进行的 episode 标记为丢弃
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()
            # 向 ZED 相机发送停止命令
            self._send_zed_command("STOP")

        finally:
            self.save_and_cleanup()


def main(config: DataExporterConfig):
    rclpy.init(args=None)
    node = rclpy.create_node("data_exporter")

    # 初始化机器人模型以获取关节名称
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    g1_rm = g1.instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    # 创建 HDF5 数据导出器
    save_dir = f"{config.root_output_dir}/{config.dataset_name}"
    
    data_exporter = HDF5DataExporter(
        save_dir=save_dir,
        task_prompt=config.task_prompt,
        robot_id=config.robot_id or "sim",
        fps=config.data_collection_frequency,
        joint_names=g1_rm.joint_names,
    )

    data_collector = Gr00tDataCollector(
        node=node,
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        state_topic_name=STATE_TOPIC_NAME,
        text_to_speech=text_to_speech,
        zed_camera_host="192.158.1.100",
        zed_camera_port=8888,
    )
    
    print("\n" + "="*50)
    print("HDF5 数据采集器已启动")
    print("="*50)
    print(f"保存目录: {save_dir}")
    print(f"文件格式: episode_X.hdf5 (每个episode单独保存)")
    print(f"任务描述: {config.task_prompt}")
    print(f"采集频率: {config.data_collection_frequency} Hz")
    print("-"*50)
    print("操作说明:")
    print("  按 'c' - 开始/停止录制")
    print("  按 'x' - 丢弃当前 episode")
    print("  Ctrl+C - 退出程序")
    print("="*50 + "\n")
    
    data_collector.run()


if __name__ == "__main__":
    # config = tyro.cli(DataExporterConfig)
    # config.task_prompt = input("Enter the task prompt: ").strip()
    # add_to_existing_dataset = input("Add to existing dataset? (y/n): ").strip().lower()

    # if add_to_existing_dataset == "y":
    #     config.dataset_name = input("Enter the dataset name: ").strip()
    # elif add_to_existing_dataset == "n":
    #     config.robot_id = "sim"
    #     config.dataset_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-G1-{config.robot_id}"
    #     config.teleoperator_username = "NEW_USER"
    #     config.support_operator_username = "NEW_USER"

    config = tyro.cli(DataExporterConfig)
    config.task_prompt = "None"




    main(config)
