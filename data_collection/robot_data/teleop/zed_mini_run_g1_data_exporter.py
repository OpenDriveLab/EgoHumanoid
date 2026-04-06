"""
G1 数据采集器 - HDF5 格式 + ZED Mini 视频录制
在PC端同时实现 Pico 采集和 ZED Mini 相机录制
保存两个文件：episode_X.hdf5（机器人数据）和 episode_X.svo2（视频）
支持实时可视化显示
"""
from collections import deque
from datetime import datetime
import os
import threading
import time

import h5py
import numpy as np
import rclpy
import tyro

# OpenCV 用于可视化
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[WARN] cv2 未安装，可视化功能将被禁用")
    CV2_AVAILABLE = False

# ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    print("[WARN] pyzed.sl 未安装，ZED相机功能将被禁用")
    ZED_AVAILABLE = False

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


class ZEDMiniRecorder:
    """
    ZED Mini 相机录制器
    直接在PC端控制ZED Mini相机，录制SVO2格式视频
    支持实时可视化显示（仅左眼图像）
    """
    
    def __init__(self, data_dir: str, resolution: str = "HD720", fps: int = 30, 
                 record_depth: bool = True, enable_visualization: bool = False):
        """
        Args:
            data_dir: 数据保存目录
            resolution: 分辨率设置 ("HD2K", "HD1080", "HD720", "VGA")
            fps: 帧率 (15, 30, 60, 100 - 取决于分辨率)
            record_depth: 是否录制深度信息
            enable_visualization: 是否启用实时可视化（默认关闭，使用 --visualize 开启）
        """
        if not ZED_AVAILABLE:
            raise RuntimeError("ZED SDK (pyzed.sl) 未安装，无法使用ZED相机功能")
        
        self.data_dir = data_dir
        self.resolution_str = resolution    
        self.fps = fps
        self.record_depth = record_depth
        self.enable_visualization = enable_visualization and CV2_AVAILABLE
        
        self.zed = None
        self.is_recording = False
        self.is_camera_open = False
        
        self.episode = None
        self.svo_path = None
        
        # 录制统计
        self.recording_start_time = None
        self.recording_stop_time = None
        self.frame_count = 0
        
        # 录制线程
        self.recording_thread = None
        self.stop_recording_flag = False
        
        # 可视化相关（仅左眼图像）
        self.left_image = None
        self.image_lock = threading.Lock()
        self.visualization_thread = None
        self.stop_visualization_flag = False
        
        # ZED 图像对象（预分配）
        self.zed_left_image = None
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"[ZED] 深度信息录制: {'启用' if self.record_depth else '禁用'}")
        print(f"[ZED] 实时可视化: {'启用' if self.enable_visualization else '禁用'}")
    
    def _get_resolution(self):
        """获取分辨率枚举值"""
        resolution_map = {
            "HD2K": sl.RESOLUTION.HD2K,      # 2208x1242
            "HD1080": sl.RESOLUTION.HD1080,  # 1920x1080
            "HD720": sl.RESOLUTION.HD720,    # 1280x720
            "VGA": sl.RESOLUTION.VGA,        # 672x376
        }
        return resolution_map.get(self.resolution_str, sl.RESOLUTION.HD720)
    
    def init_camera(self):
        """初始化ZED Mini相机"""
        if self.is_camera_open:
            print("[ZED] 相机已经打开")
            return True
        
        self.zed = sl.Camera()
        
        # 设置初始化参数
        init_params = sl.InitParameters()
        init_params.camera_resolution = self._get_resolution()
        init_params.camera_fps = self.fps
        
        # 深度模式
        if self.record_depth:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE
        
        init_params.coordinate_units = sl.UNIT.METER
        
        # 打开相机
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] 无法打开 ZED Mini 相机: {status}")
            return False
        
        self.is_camera_open = True
        print("[ZED] ZED Mini 相机已成功打开")
        
        # 获取相机信息
        camera_info = self.zed.get_camera_information()
        print(f"[ZED] 相机型号: {camera_info.camera_model}")
        print(f"[ZED] 序列号: {camera_info.serial_number}")
        
        # 获取分辨率信息
        resolution = camera_info.camera_configuration.resolution
        self.image_width = resolution.width
        self.image_height = resolution.height
        print(f"[ZED] 分辨率: {self.image_width}x{self.image_height}")
        print(f"[ZED] 帧率: {self.fps} FPS")
        
        # 预分配 ZED 图像对象（用于可视化，仅左眼）
        if self.enable_visualization:
            self.zed_left_image = sl.Mat()
            
            # 启动可视化线程
            self.stop_visualization_flag = False
            self.visualization_thread = threading.Thread(target=self._visualization_loop, daemon=True)
            self.visualization_thread.start()
            print("[ZED] 可视化线程已启动")
        
        return True
    
    def _set_episode_path(self, episode_index):
        """根据episode序号设置SVO文件路径"""
        self.episode = episode_index
        self.svo_path = os.path.join(self.data_dir, f"episode_{episode_index}.svo2")
        print(f"[ZED] SVO文件路径: {self.svo_path}")
    
    def start_recording(self, episode_index):
        """开始录制SVO"""
        if not self.is_camera_open:
            print("[ZED] 相机未打开，无法开始录制")
            return False
        
        if self.is_recording:
            print("[ZED] 录制已在进行中，先停止当前录制")
            self.stop_recording()
        
        # 设置episode路径
        self._set_episode_path(episode_index)
        
        # 检查文件是否已存在
        if os.path.exists(self.svo_path):
            print(f"[ZED] SVO文件已存在: {self.svo_path}，将覆盖")
        
        # 设置录制参数
        recording_params = sl.RecordingParameters()
        recording_params.video_filename = self.svo_path
        recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264
        
        # 启用录制
        err = self.zed.enable_recording(recording_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] 无法启用录制: {err}")
            return False
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.frame_count = 0
        self.stop_recording_flag = False
        
        # 启动录制线程
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        
        print(f"[ZED] 开始录制 episode_{episode_index}")
        return True
    
    def _recording_loop(self):
        """录制循环：持续grab帧以写入SVO文件，带帧率控制以避免CPU占用过高"""
        # 计算目标帧间隔（稍微短一点确保能达到目标帧率）
        target_frame_interval = 1.0 / self.fps * 0.95
        
        try:
            while self.is_recording and not self.stop_recording_flag:
                frame_start = time.time()
                
                status = self.zed.grab()
                if status == sl.ERROR_CODE.SUCCESS:
                    self.frame_count += 1
                    
                    # 获取图像用于可视化
                    if self.enable_visualization:
                        self._update_visualization_images()
                    
                    if self.frame_count % 100 == 0:
                        elapsed = time.time() - self.recording_start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        print(f"[ZED] 已录制 {self.frame_count} 帧，当前帧率: {fps:.2f} Hz")
                    
                    # 帧率控制：确保不超过目标帧率，减少CPU占用
                    frame_elapsed = time.time() - frame_start
                    sleep_time = target_frame_interval - frame_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                elif status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    print("[ZED] 达到SVO文件大小限制")
                    break
                else:
                    # grab失败时短暂休眠
                    time.sleep(0.001)
        except Exception as e:
            print(f"[ZED] 录制循环出错: {e}")
    
    def _update_visualization_images(self):
        """从 ZED 相机获取左眼图像用于可视化"""
        try:
            # 获取左眼图像
            self.zed.retrieve_image(self.zed_left_image, sl.VIEW.LEFT)
            left_np = self.zed_left_image.get_data()
            
            # 更新共享图像（线程安全）
            with self.image_lock:
                self.left_image = left_np.copy() if left_np is not None else None
        except Exception as e:
            pass  # 静默处理可视化图像获取错误
    
    def _visualization_loop(self):
        """可视化循环：显示左眼实时画面"""
        if not CV2_AVAILABLE:
            print("[ZED] cv2 不可用，跳过可视化")
            return
        
        window_name = "ZED Mini - Left Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 设置窗口大小
        display_width = self.image_width
        display_height = self.image_height
        # 限制最大窗口大小
        max_width = 1280
        if display_width > max_width:
            scale = max_width / display_width
            display_width = int(display_width * scale)
            display_height = int(display_height * scale)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        print(f"[ZED] 可视化窗口已创建: {window_name}")
        print("[ZED] 按 'q' 键关闭可视化窗口")
        
        last_fps_update = time.time()
        frame_times = []
        display_fps = 0.0
        
        while not self.stop_visualization_flag:
            frame_start = time.time()
            
            # 获取当前图像
            with self.image_lock:
                left = self.left_image.copy() if self.left_image is not None else None
            
            if left is None:
                # 如果没有图像，显示等待画面
                placeholder = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...", (50, self.image_height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow(window_name, placeholder)
            else:
                # 转换颜色格式 (BGRA -> BGR)
                if left.shape[2] == 4:
                    left = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
                
                # 添加 OSD 信息
                display_image = self._add_osd_info(left, display_fps)
                
                cv2.imshow(window_name, display_image)
            
            # 计算显示帧率
            frame_times.append(time.time() - frame_start)
            if time.time() - last_fps_update > 0.5:
                if frame_times:
                    display_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
                    frame_times = []
                last_fps_update = time.time()
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[ZED] 用户请求关闭可视化窗口")
                self.stop_visualization_flag = True
                break
            
            # 控制显示帧率（约30fps）
            elapsed = time.time() - frame_start
            sleep_time = max(0, (1.0 / 30) - elapsed)
            time.sleep(sleep_time)
        
        cv2.destroyWindow(window_name)
        print("[ZED] 可视化窗口已关闭")
    
    def _add_osd_info(self, image, display_fps):
        """在图像上添加 OSD (On-Screen Display) 信息"""
        # 创建半透明覆盖层
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # OSD 背景区域
        osd_height = 90
        cv2.rectangle(overlay, (0, 0), (w, osd_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # 文字参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25
        
        # 录制状态
        if self.is_recording:
            status_text = f"REC Episode {self.episode}"
            status_color = (0, 0, 255)  # 红色
            # 录制时间和帧数
            elapsed = time.time() - self.recording_start_time if self.recording_start_time else 0
            rec_fps = self.frame_count / elapsed if elapsed > 0 else 0
            info_text = f"Frames: {self.frame_count} | {elapsed:.1f}s | {rec_fps:.1f} FPS"
        else:
            status_text = "IDLE"
            status_color = (0, 255, 0)  # 绿色
            info_text = "Press 'c' to start"
        
        # 绘制状态
        cv2.putText(image, status_text, (10, line_height), font, font_scale, status_color, thickness)
        cv2.putText(image, info_text, (10, line_height * 2), font, font_scale, (255, 255, 255), thickness)
        
        # 显示帧率
        fps_text = f"Display: {display_fps:.1f} FPS | Camera: {self.fps} FPS"
        cv2.putText(image, fps_text, (10, line_height * 3), font, font_scale, (200, 200, 200), thickness)
        
        # 录制指示器（录制时闪烁的红点）
        if self.is_recording:
            if int(time.time() * 2) % 2 == 0:  # 每0.5秒闪烁
                cv2.circle(image, (w - 25, 25), 12, (0, 0, 255), -1)
        
        return image
    
    def stop_recording(self):
        """停止录制SVO"""
        if not self.is_recording:
            return
        
        self.stop_recording_flag = True
        self.is_recording = False
        
        # 等待录制线程结束
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # 停止录制
        self.zed.disable_recording()
        self.recording_stop_time = time.time()
        
        # 计算录制统计信息
        duration_sec = self.recording_stop_time - self.recording_start_time if self.recording_start_time else 0
        fps = self.frame_count / duration_sec if duration_sec > 0 else 0
        
        print(f"[ZED] 录制已停止")
        print(f"[ZED] 录制时长: {duration_sec:.3f} 秒, 总帧数: {self.frame_count}, 平均帧率: {fps:.2f} Hz")
        
        if os.path.exists(self.svo_path):
            file_size_mb = os.path.getsize(self.svo_path) / (1024 * 1024)
            print(f"[ZED] SVO文件大小: {file_size_mb:.2f} MB")
    
    def discard_recording(self):
        """丢弃当前录制：停止录制并删除SVO文件"""
        if not self.is_recording:
            print("[ZED] 当前没有正在进行的录制")
            return
        
        self.stop_recording_flag = True
        self.is_recording = False
        
        # 等待录制线程结束
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # 停止录制
        self.zed.disable_recording()
        self.recording_stop_time = time.time()
        
        print(f"[ZED] 丢弃录制 episode_{self.episode}")
        
        # 删除SVO文件
        if self.svo_path and os.path.exists(self.svo_path):
            try:
                os.remove(self.svo_path)
                print(f"[ZED] 已删除SVO文件: {self.svo_path}")
            except Exception as e:
                print(f"[ZED] 删除SVO文件失败: {e}")
        
        self.episode = None
        self.svo_path = None
    
    def close(self):
        """关闭相机"""
        if self.is_recording:
            self.stop_recording()
        
        # 停止可视化线程
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.stop_visualization_flag = True
            self.visualization_thread.join(timeout=2.0)
            print("[ZED] 可视化线程已停止")
        
        if self.zed and self.is_camera_open:
            self.zed.close()
            self.is_camera_open = False
            print("[ZED] 相机已关闭")
    
    def grab_frame_for_preview(self):
        """
        抓取一帧用于预览（非录制时使用）
        用于在非录制状态下也能显示实时画面
        """
        if not self.is_camera_open or self.is_recording:
            return
        
        status = self.zed.grab()
        if status == sl.ERROR_CODE.SUCCESS and self.enable_visualization:
            self._update_visualization_images()


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
    ├── episode_0.svo2  (ZED Mini 录制的视频)
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


class Gr00tDataCollectorWithZED:
    """
    G1 数据采集器 + ZED Mini 相机录制
    在PC端同时采集Pico数据和ZED Mini视频
    支持实时可视化显示（仅左眼图像）
    """
    
    def __init__(
        self,
        node,
        state_topic_name: str,
        data_exporter: HDF5DataExporter,
        text_to_speech=None,
        frequency=20,
        # ZED Mini 相机参数
        enable_zed: bool = True,
        zed_resolution: str = "HD720",
        zed_fps: int = 30,
        zed_record_depth: bool = True,
        # 可视化参数
        enable_visualization: bool = False,
    ):
        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.data_exporter = data_exporter
        self.enable_zed = enable_zed
        self.enable_visualization = enable_visualization

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

        # 初始化 ZED Mini 相机
        self.zed_recorder = None
        if self.enable_zed and ZED_AVAILABLE:
            try:
                self.zed_recorder = ZEDMiniRecorder(
                    data_dir=self.data_exporter.save_dir,
                    resolution=zed_resolution,
                    fps=zed_fps,
                    record_depth=zed_record_depth,
                    enable_visualization=enable_visualization,
                )
                if not self.zed_recorder.init_camera():
                    print("[WARN] ZED Mini 相机初始化失败，将禁用相机功能")
                    self.zed_recorder = None
            except Exception as e:
                print(f"[WARN] ZED Mini 相机初始化出错: {e}")
                self.zed_recorder = None
        elif self.enable_zed and not ZED_AVAILABLE:
            print("[WARN] ZED SDK 不可用，将禁用相机功能")

        print(f"Recording to {self.data_exporter.save_dir}")
        print(f"ZED Mini 相机: {'已启用' if self.zed_recorder else '已禁用'}")
        print(f"实时可视化: {'已启用' if (self.zed_recorder and self.enable_visualization) else '已禁用'}")

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
                # 开始 ZED 相机录制
                if self.zed_recorder:
                    self.zed_recorder.start_recording(episode_index)
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._print_and_say("Stopping recording, preparing to save")
                # 停止 ZED 相机录制
                if self.zed_recorder:
                    self.zed_recorder.stop_recording()
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state")
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                # B键中断：丢弃当前episode，保持index不变，等待下次A键恢复
                self.data_exporter.save_episode_as_discarded(keep_index=True)
                self.data_exporter.interrupted_by_b_button = True
                self._episode_state.reset_state()
                self._print_and_say("B键中断：已丢弃当前episode，等待下次A键恢复（保持index不变）")
                # 丢弃 ZED 相机录制
                if self.zed_recorder:
                    self.zed_recorder.discard_recording()

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

        # 关闭 ZED 相机
        if self.zed_recorder:
            self.zed_recorder.close()

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
                    
                    # 4. 如果未录制，仍然抓取帧用于预览
                    if self.zed_recorder and not self.zed_recorder.is_recording:
                        self.zed_recorder.grab_frame_for_preview()

                    end_time = time.monotonic()

                self.rate.sleep()
                
                # 检查可视化窗口是否被关闭
                if self.zed_recorder and self.zed_recorder.stop_visualization_flag:
                    print("[INFO] 可视化窗口已关闭，继续运行...")
                    # 可视化被关闭不影响数据采集

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            # 用户触发键盘中断，将正在进行的 episode 标记为丢弃
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()
            # 丢弃 ZED 相机录制
            if self.zed_recorder and self.zed_recorder.is_recording:
                self.zed_recorder.discard_recording()

        finally:
            self.save_and_cleanup()


def main(config: DataExporterConfig, enable_visualization: bool = False):
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

    data_collector = Gr00tDataCollectorWithZED(
        node=node,
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        state_topic_name=STATE_TOPIC_NAME,
        text_to_speech=text_to_speech,
        # ZED Mini 相机参数
        enable_zed=True,
        zed_resolution="HD720",  # ZED Mini 支持的分辨率: HD2K, HD1080, HD720, VGA
        zed_fps=30,             # 帧率: 15, 30, 60 (取决于分辨率)
        zed_record_depth=True,   # 是否录制深度信息
        # 可视化参数
        enable_visualization=enable_visualization,
    )
    
    print("\n" + "="*60)
    print("G1 数据采集器 + ZED Mini 相机录制 已启动")
    print("="*60)
    print(f"保存目录: {save_dir}")
    print(f"文件格式:")
    print(f"  - episode_X.hdf5 (机器人状态数据)")
    print(f"  - episode_X.svo2 (ZED Mini 视频)")
    print(f"任务描述: {config.task_prompt}")
    print(f"采集频率: {config.data_collection_frequency} Hz")
    print(f"ZED 分辨率: HD720, 帧率: 30 FPS")
    print(f"实时可视化: {'启用' if enable_visualization else '禁用'}")
    print("-"*60)
    print("操作说明:")
    print("  按 'c' - 开始/停止录制 (同时控制Pico采集和ZED录制)")
    print("  按 'x' - 丢弃当前 episode (同时丢弃h5和svo2)")
    if enable_visualization:
        print("  按 'q' - 关闭可视化窗口（不影响录制）")
    print("  Ctrl+C - 退出程序")
    print("="*60 + "\n")
    
    data_collector.run()


if __name__ == "__main__":
    import argparse
    
    # 首先解析可视化相关参数
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--visualize", action="store_true", 
                        help="启用实时可视化显示（左眼图像）")
    
    # 解析已知参数，其余传给 tyro
    args, remaining = parser.parse_known_args()
    
    # 使用 tyro 解析 DataExporterConfig
    import sys
    sys.argv = [sys.argv[0]] + remaining  # 移除已解析的参数
    config = tyro.cli(DataExporterConfig)
    
    # 如果未指定 dataset_name，使用时间戳作为默认名称
    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("zed_mini_dataset_%Y%m%d_%H%M%S")
    
    # 如果未指定 task_prompt，设置默认值
    if config.task_prompt == "demo":
        config.task_prompt = "ZED Mini 数据采集"

    main(
        config, 
        enable_visualization=args.visualize,
    )

