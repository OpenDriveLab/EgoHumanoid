"""
G1 Data Collector - HDF5 format + ZED Mini video recording
Simultaneous Pico data collection and ZED Mini camera recording on PC
Saves two files: episode_X.hdf5 (robot data) and episode_X.svo2 (video)
Supports real-time visualization
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

# OpenCV for visualization
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[WARN] cv2 not installed, visualization will be disabled")
    CV2_AVAILABLE = False

# ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    print("[WARN] pyzed.sl not installed, ZED camera functionality will be disabled")
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
    ZED Mini Camera Recorder
    Direct control of ZED Mini camera on PC, recording SVO2 format video
    Supports real-time visualization (left eye image only)
    """
    
    def __init__(self, data_dir: str, resolution: str = "HD720", fps: int = 30,
                 record_depth: bool = True, enable_visualization: bool = False):
        """
        Args:
            data_dir: Data save directory
            resolution: Resolution setting ("HD2K", "HD1080", "HD720", "VGA")
            fps: Frame rate (15, 30, 60, 100 - depends on resolution)
            record_depth: Whether to record depth information
            enable_visualization: Whether to enable real-time visualization (default off, use --visualize to enable)
        """
        if not ZED_AVAILABLE:
            raise RuntimeError("ZED SDK (pyzed.sl) not installed, cannot use ZED camera functionality")
        
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
        
        # Recording statistics
        self.recording_start_time = None
        self.recording_stop_time = None
        self.frame_count = 0

        # Recording thread
        self.recording_thread = None
        self.stop_recording_flag = False

        # Visualization related (left eye image only)
        self.left_image = None
        self.image_lock = threading.Lock()
        self.visualization_thread = None
        self.stop_visualization_flag = False

        # ZED image object (pre-allocated)
        self.zed_left_image = None

        # Ensure directory exists
        os.makedirs(data_dir, exist_ok=True)

        print(f"[ZED] Depth recording: {'Enabled' if self.record_depth else 'Disabled'}")
        print(f"[ZED] Real-time visualization: {'Enabled' if self.enable_visualization else 'Disabled'}")
    
    def _get_resolution(self):
        """Get resolution enum value"""
        resolution_map = {
            "HD2K": sl.RESOLUTION.HD2K,      # 2208x1242
            "HD1080": sl.RESOLUTION.HD1080,  # 1920x1080
            "HD720": sl.RESOLUTION.HD720,    # 1280x720
            "VGA": sl.RESOLUTION.VGA,        # 672x376
        }
        return resolution_map.get(self.resolution_str, sl.RESOLUTION.HD720)
    
    def init_camera(self):
        """Initialize ZED Mini camera"""
        if self.is_camera_open:
            print("[ZED] Camera already open")
            return True

        self.zed = sl.Camera()

        # Set initialization parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = self._get_resolution()
        init_params.camera_fps = self.fps

        # Depth mode
        if self.record_depth:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE

        init_params.coordinate_units = sl.UNIT.METER

        # Open camera
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] Unable to open ZED Mini camera: {status}")
            return False

        self.is_camera_open = True
        print("[ZED] ZED Mini camera successfully opened")

        # Get camera information
        camera_info = self.zed.get_camera_information()
        print(f"[ZED] Camera model: {camera_info.camera_model}")
        print(f"[ZED] Serial number: {camera_info.serial_number}")

        # Get resolution information
        resolution = camera_info.camera_configuration.resolution
        self.image_width = resolution.width
        self.image_height = resolution.height
        print(f"[ZED] Resolution: {self.image_width}x{self.image_height}")
        print(f"[ZED] Frame rate: {self.fps} FPS")

        # Pre-allocate ZED image object (for visualization, left eye only)
        if self.enable_visualization:
            self.zed_left_image = sl.Mat()

            # Start visualization thread
            self.stop_visualization_flag = False
            self.visualization_thread = threading.Thread(target=self._visualization_loop, daemon=True)
            self.visualization_thread.start()
            print("[ZED] Visualization thread started")

        return True
    
    def _set_episode_path(self, episode_index):
        """Set SVO file path based on episode index"""
        self.episode = episode_index
        self.svo_path = os.path.join(self.data_dir, f"episode_{episode_index}.svo2")
        print(f"[ZED] SVO file path: {self.svo_path}")
    
    def start_recording(self, episode_index):
        """Start recording SVO"""
        if not self.is_camera_open:
            print("[ZED] Camera not open, cannot start recording")
            return False

        if self.is_recording:
            print("[ZED] Recording already in progress, stopping current recording")
            self.stop_recording()

        # Set episode path
        self._set_episode_path(episode_index)

        # Check if file already exists
        if os.path.exists(self.svo_path):
            print(f"[ZED] SVO file already exists: {self.svo_path}, will overwrite")

        # Set recording parameters
        recording_params = sl.RecordingParameters()
        recording_params.video_filename = self.svo_path
        recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264

        # Enable recording
        err = self.zed.enable_recording(recording_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] Unable to enable recording: {err}")
            return False

        self.is_recording = True
        self.recording_start_time = time.time()
        self.frame_count = 0
        self.stop_recording_flag = False

        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()

        print(f"[ZED] Started recording episode_{episode_index}")
        return True
    
    def _recording_loop(self):
        """Recording loop: continuously grab frames to write to SVO file, with frame rate control to avoid high CPU usage"""
        # Calculate target frame interval (slightly shorter to ensure target frame rate is met)
        target_frame_interval = 1.0 / self.fps * 0.95

        try:
            while self.is_recording and not self.stop_recording_flag:
                frame_start = time.time()

                status = self.zed.grab()
                if status == sl.ERROR_CODE.SUCCESS:
                    self.frame_count += 1

                    # Get image for visualization
                    if self.enable_visualization:
                        self._update_visualization_images()

                    if self.frame_count % 100 == 0:
                        elapsed = time.time() - self.recording_start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        print(f"[ZED] Recorded {self.frame_count} frames, current frame rate: {fps:.2f} Hz")

                    # Frame rate control: ensure not exceeding target frame rate, reduce CPU usage
                    frame_elapsed = time.time() - frame_start
                    sleep_time = target_frame_interval - frame_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                elif status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    print("[ZED] Reached SVO file size limit")
                    break
                else:
                    # Brief sleep when grab fails
                    time.sleep(0.001)
        except Exception as e:
            print(f"[ZED] Recording loop error: {e}")
    
    def _update_visualization_images(self):
        """Get left eye image from ZED camera for visualization"""
        try:
            # Get left eye image
            self.zed.retrieve_image(self.zed_left_image, sl.VIEW.LEFT)
            left_np = self.zed_left_image.get_data()

            # Update shared image (thread-safe)
            with self.image_lock:
                self.left_image = left_np.copy() if left_np is not None else None
        except Exception as e:
            pass  # Silently handle visualization image retrieval errors
    
    def _visualization_loop(self):
        """Visualization loop: display live left eye feed"""
        if not CV2_AVAILABLE:
            print("[ZED] cv2 unavailable, skipping visualization")
            return

        window_name = "ZED Mini - Left Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Set window size
        display_width = self.image_width
        display_height = self.image_height
        # Limit maximum window size
        max_width = 1280
        if display_width > max_width:
            scale = max_width / display_width
            display_width = int(display_width * scale)
            display_height = int(display_height * scale)
        cv2.resizeWindow(window_name, display_width, display_height)

        print(f"[ZED] Visualization window created: {window_name}")
        print("[ZED] Press 'q' key to close visualization window")
        
        last_fps_update = time.time()
        frame_times = []
        display_fps = 0.0
        
        while not self.stop_visualization_flag:
            frame_start = time.time()

            # Get current image
            with self.image_lock:
                left = self.left_image.copy() if self.left_image is not None else None

            if left is None:
                # If no image, display waiting screen
                placeholder = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...", (50, self.image_height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow(window_name, placeholder)
            else:
                # Convert color format (BGRA -> BGR)
                if left.shape[2] == 4:
                    left = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)

                # Add OSD info
                display_image = self._add_osd_info(left, display_fps)

                cv2.imshow(window_name, display_image)

            # Calculate display frame rate
            frame_times.append(time.time() - frame_start)
            if time.time() - last_fps_update > 0.5:
                if frame_times:
                    display_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
                    frame_times = []
                last_fps_update = time.time()

            # Check keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[ZED] User requested to close visualization window")
                self.stop_visualization_flag = True
                break

            # Control display frame rate (approx 30fps)
            elapsed = time.time() - frame_start
            sleep_time = max(0, (1.0 / 30) - elapsed)
            time.sleep(sleep_time)

        cv2.destroyWindow(window_name)
        print("[ZED] Visualization window closed")
    
    def _add_osd_info(self, image, display_fps):
        """Add OSD (On-Screen Display) info to image"""
        # Create semi-transparent overlay
        overlay = image.copy()
        h, w = image.shape[:2]

        # OSD background area
        osd_height = 90
        cv2.rectangle(overlay, (0, 0), (w, osd_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25

        # Recording status
        if self.is_recording:
            status_text = f"REC Episode {self.episode}"
            status_color = (0, 0, 255)  # Red
            # Recording time and frame count
            elapsed = time.time() - self.recording_start_time if self.recording_start_time else 0
            rec_fps = self.frame_count / elapsed if elapsed > 0 else 0
            info_text = f"Frames: {self.frame_count} | {elapsed:.1f}s | {rec_fps:.1f} FPS"
        else:
            status_text = "IDLE"
            status_color = (0, 255, 0)  # Green
            info_text = "Press 'c' to start"

        # Draw status
        cv2.putText(image, status_text, (10, line_height), font, font_scale, status_color, thickness)
        cv2.putText(image, info_text, (10, line_height * 2), font, font_scale, (255, 255, 255), thickness)

        # Display frame rate
        fps_text = f"Display: {display_fps:.1f} FPS | Camera: {self.fps} FPS"
        cv2.putText(image, fps_text, (10, line_height * 3), font, font_scale, (200, 200, 200), thickness)

        # Recording indicator (blinking red dot when recording)
        if self.is_recording:
            if int(time.time() * 2) % 2 == 0:  # Blink every 0.5 seconds
                cv2.circle(image, (w - 25, 25), 12, (0, 0, 255), -1)

        return image
    
    def stop_recording(self):
        """Stop recording SVO"""
        if not self.is_recording:
            return

        self.stop_recording_flag = True
        self.is_recording = False

        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

        # Stop recording
        self.zed.disable_recording()
        self.recording_stop_time = time.time()

        # Calculate recording statistics
        duration_sec = self.recording_stop_time - self.recording_start_time if self.recording_start_time else 0
        fps = self.frame_count / duration_sec if duration_sec > 0 else 0

        print(f"[ZED] Recording stopped")
        print(f"[ZED] Recording duration: {duration_sec:.3f} seconds, total frames: {self.frame_count}, average frame rate: {fps:.2f} Hz")

        if os.path.exists(self.svo_path):
            file_size_mb = os.path.getsize(self.svo_path) / (1024 * 1024)
            print(f"[ZED] SVO file size: {file_size_mb:.2f} MB")
    
    def discard_recording(self):
        """Discard current recording: stop recording and delete SVO file"""
        if not self.is_recording:
            print("[ZED] No recording in progress")
            return

        self.stop_recording_flag = True
        self.is_recording = False

        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

        # Stop recording
        self.zed.disable_recording()
        self.recording_stop_time = time.time()

        print(f"[ZED] Discarding recording episode_{self.episode}")

        # Delete SVO file
        if self.svo_path and os.path.exists(self.svo_path):
            try:
                os.remove(self.svo_path)
                print(f"[ZED] Deleted SVO file: {self.svo_path}")
            except Exception as e:
                print(f"[ZED] Failed to delete SVO file: {e}")

        self.episode = None
        self.svo_path = None
    
    def close(self):
        """Close camera"""
        if self.is_recording:
            self.stop_recording()

        # Stop visualization thread
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.stop_visualization_flag = True
            self.visualization_thread.join(timeout=2.0)
            print("[ZED] Visualization thread stopped")

        if self.zed and self.is_camera_open:
            self.zed.close()
            self.is_camera_open = False
            print("[ZED] Camera closed")
    
    def grab_frame_for_preview(self):
        """
        Grab a frame for preview (used when not recording)
        Used to display live feed even when not recording
        """
        if not self.is_camera_open or self.is_recording:
            return

        status = self.zed.grab()
        if status == sl.ERROR_CODE.SUCCESS and self.enable_visualization:
            self._update_visualization_images()


class HDF5DataExporter:
    """
    HDF5 Data Exporter - Each episode saved as a separate file

    Data structure:
    dataset_dir/
    ├── episode_0.hdf5
    │   ├── timestamp (N,) - time.time() timestamps
    │   ├── observation_state (N, num_joints) - Joint angles
    │   ├── observation_eef_state (N, 14) - End-effector pose
    │   ├── action (N, num_joints) - Actions
    │   ├── action_eef (N, 14) - End-effector actions
    │   ├── teleop_navigate_command (N, 3) - Navigation commands
    │   ├── teleop_base_height_command (N, 1) - Base height commands
    │   ├── hand_status (N, 2) - Hand status (trigger state): [left hand status, right hand status], 1 when trigger pressed, 0 otherwise
    │   └── metadata (attributes)
    │       ├── task_prompt
    │       ├── robot_id
    │       ├── fps
    │       └── joint_names
    ├── episode_0.svo2  (ZED Mini recorded video)
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

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Episode buffer
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

        # Flag: whether interrupted by B button (needs to keep index)
        self.interrupted_by_b_button = False

        # Scan directory, find next episode index
        self._init_episode_index()
        
    def _init_episode_index(self):
        """Scan directory, find next available episode index"""
        existing_episodes = []
        for filename in os.listdir(self.save_dir):
            if filename.startswith("episode_") and filename.endswith(".hdf5"):
                try:
                    # Extract index from filename: episode_X.hdf5 -> X
                    index = int(filename.replace("episode_", "").replace(".hdf5", ""))
                    existing_episodes.append(index)
                except ValueError:
                    continue

        if existing_episodes:
            self.episode_buffer["episode_index"] = max(existing_episodes) + 1
            print(f"Continuing to add to existing dataset, current episode index: {self.episode_buffer['episode_index']}")
        else:
            print(f"Creating new dataset directory: {self.save_dir}")
    
    def add_frame(self, frame_data: dict):
        """Add a frame of data to buffer"""
        # Add timestamp
        self.episode_buffer["timestamp"].append(time.time())

        # Add each data field
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
        # Add hand status (trigger state): [left hand status, right hand status], 1 when trigger pressed, 0 otherwise
        hand_status = frame_data.get("hand_status", np.array([0, 0], dtype=np.int32))
        # Ensure hand_status is array with shape (2,)
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
        """Save current episode to a separate HDF5 file"""
        if self.episode_buffer["size"] == 0:
            print("Warning: Episode is empty, skipping save")
            return

        episode_index = self.episode_buffer['episode_index']
        episode_filename = f"episode_{episode_index}.hdf5"
        episode_path = os.path.join(self.save_dir, episode_filename)

        with h5py.File(episode_path, "w") as f:
            # Save timestamps
            f.create_dataset(
                "timestamp",
                data=np.array(self.episode_buffer["timestamp"], dtype=np.float64),
                compression="gzip"
            )

            # Save observation states
            f.create_dataset(
                "observation_state",
                data=np.stack(self.episode_buffer["observation_state"]),
                compression="gzip"
            )

            # Save end-effector states
            f.create_dataset(
                "observation_eef_state",
                data=np.stack(self.episode_buffer["observation_eef_state"]),
                compression="gzip"
            )

            # Save actions
            f.create_dataset(
                "action",
                data=np.stack(self.episode_buffer["action"]),
                compression="gzip"
            )

            # Save end-effector actions
            f.create_dataset(
                "action_eef",
                data=np.stack(self.episode_buffer["action_eef"]),
                compression="gzip"
            )

            # Save navigation commands
            f.create_dataset(
                "teleop_navigate_command",
                data=np.stack(self.episode_buffer["teleop_navigate_command"]),
                compression="gzip"
            )

            # Save base height commands
            f.create_dataset(
                "teleop_base_height_command",
                data=np.stack(self.episode_buffer["teleop_base_height_command"]),
                compression="gzip"
            )

            # Save hand status (trigger state): [left hand status, right hand status]
            hand_status_data = np.stack(self.episode_buffer["hand_status"])
            # Ensure shape is (N, 2)
            if len(hand_status_data.shape) == 1:
                # If only one dimension, reshape to (N, 1) then expand to (N, 2)
                hand_status_data = hand_status_data.reshape(-1, 1)
                hand_status_data = np.repeat(hand_status_data, 2, axis=1)
            elif hand_status_data.shape[1] == 1:
                # If (N, 1), expand to (N, 2)
                hand_status_data = np.repeat(hand_status_data, 2, axis=1)
            f.create_dataset(
                "hand_status",
                data=hand_status_data,
                compression="gzip"
            )

            # Add metadata attributes
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
        print(f"Saved {episode_filename}: {self.episode_buffer['size']} frames, duration {duration:.2f}s")

        # Reset buffer
        self._reset_buffer()
    
    def save_episode_as_discarded(self, keep_index=False):
        """Discard current episode

        Args:
            keep_index: If True, keep episode_index unchanged (for B button interrupt case)
        """
        print(f"Discarding episode_{self.episode_buffer['episode_index']}: {self.episode_buffer['size']} frames")
        self._reset_buffer(keep_index=keep_index)

    def _reset_buffer(self, keep_index=False):
        """Reset buffer

        Args:
            keep_index: If True, keep episode_index unchanged
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
    G1 Data Collector + ZED Mini camera recording
    Simultaneous Pico data collection and ZED Mini video on PC
    Supports real-time visualization (left eye image only)
    """
    
    def __init__(
        self,
        node,
        state_topic_name: str,
        data_exporter: HDF5DataExporter,
        text_to_speech=None,
        frequency=20,
        # ZED Mini camera parameters
        enable_zed: bool = True,
        zed_resolution: str = "HD720",
        zed_fps: int = 30,
        zed_record_depth: bool = True,
        # Visualization parameters
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
        # Subscribe to teleop commands to get finger data for trigger state inference
        self._teleop_cmd_subscriber = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)
        self.rate = self.node.create_rate(self.frequency)

        self.latest_proprio_msg = None
        self.latest_teleop_cmd = None

        self.telemetry = Telemetry(window_size=100)

        # Initialize ZED Mini camera
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
                    print("[WARN] ZED Mini camera initialization failed, disabling camera functionality")
                    self.zed_recorder = None
            except Exception as e:
                print(f"[WARN] ZED Mini camera initialization error: {e}")
                self.zed_recorder = None
        elif self.enable_zed and not ZED_AVAILABLE:
            print("[WARN] ZED SDK unavailable, disabling camera functionality")

        print(f"Recording to {self.data_exporter.save_dir}")
        print(f"ZED Mini camera: {'Enabled' if self.zed_recorder else 'Disabled'}")
        print(f"Real-time visualization: {'Enabled' if (self.zed_recorder and self.enable_visualization) else 'Disabled'}")

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
                # If previously interrupted by B button, keep index unchanged
                if self.data_exporter.interrupted_by_b_button:
                    self.data_exporter.interrupted_by_b_button = False
                    self._print_and_say(f"Resume recording episode {episode_index} (keeping index unchanged)")
                else:
                    self._print_and_say(f"Started recording episode {episode_index}")
                # Start ZED camera recording
                if self.zed_recorder:
                    self.zed_recorder.start_recording(episode_index)
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._print_and_say("Stopping recording, preparing to save")
                # Stop ZED camera recording
                if self.zed_recorder:
                    self.zed_recorder.stop_recording()
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state")
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                # B button interrupt: discard current episode, keep index unchanged, wait for next A button to resume
                self.data_exporter.save_episode_as_discarded(keep_index=True)
                self.data_exporter.interrupted_by_b_button = True
                self._episode_state.reset_state()
                self._print_and_say("B button interrupt: Discarded current episode, waiting for next A button to resume (keeping index unchanged)")
                # Discard ZED camera recording
                if self.zed_recorder:
                    self.zed_recorder.discard_recording()

    def _get_hand_status_from_fingers(self, left_fingers=None, right_fingers=None):
        """Infer trigger state from finger data

        Args:
            left_fingers: Left hand finger data, numpy array with shape (25, 4, 4)
            right_fingers: Right hand finger data, numpy array with shape (25, 4, 4)

        Returns:
            np.ndarray: Array with shape (2,), [left hand status, right hand status], each value is 0 or 1
                       1 indicates trigger pressed, 0 indicates not pressed
        """
        # According to pico_streamer logic:
        # - When only trigger pressed (trigger > 0.5 and grip <= 0.5), index finger closes: fingertips[4 + index, 0, 3] = 1.0
        # - When both trigger and grip pressed (trigger > 0.5 and grip > 0.5), middle finger closes: fingertips[4 + middle, 0, 3] = 1.0
        # So check if index or middle finger is closed to determine if trigger is pressed

        index = 5  # Index finger index
        middle = 10  # Middle finger index

        left_status = 0
        right_status = 0

        # Check if left hand trigger pressed
        if left_fingers is not None:
            try:
                left_index_closed = left_fingers[4 + index, 0, 3] > 0.5  # Index finger closed
                left_middle_closed = left_fingers[4 + middle, 0, 3] > 0.5  # Middle finger closed
                if left_index_closed or left_middle_closed:
                    left_status = 1
            except (IndexError, TypeError):
                left_status = 0

        # Check if right hand trigger pressed
        if right_fingers is not None:
            try:
                right_index_closed = right_fingers[4 + index, 0, 3] > 0.5  # Index finger closed
                right_middle_closed = right_fingers[4 + middle, 0, 3] > 0.5  # Middle finger closed
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

        # Get latest teleop command to get finger data
        teleop_cmd = self._teleop_cmd_subscriber.get_msg()
        if teleop_cmd is not None:
            self.latest_teleop_cmd = teleop_cmd

        if self._episode_state.get_state() == self._episode_state.RECORDING:
            # Get finger data from teleop command, infer trigger state
            hand_status = np.array([0, 0], dtype=np.int32)  # [left hand status, right hand status]
            if self.latest_teleop_cmd is not None:
                # Try to get finger data from teleop command
                # Note: teleop command may contain left_fingers and right_fingers, but format may vary
                # Need to adjust based on actual data structure
                left_fingers = None
                right_fingers = None

                # Check if finger data exists (might be in ik_data or other fields)
                if "left_fingers" in self.latest_teleop_cmd:
                    left_fingers = self.latest_teleop_cmd["left_fingers"]
                    if isinstance(left_fingers, dict) and "position" in left_fingers:
                        left_fingers = np.array(left_fingers["position"])

                if "right_fingers" in self.latest_teleop_cmd:
                    right_fingers = self.latest_teleop_cmd["right_fingers"]
                    if isinstance(right_fingers, dict) and "position" in right_fingers:
                        right_fingers = np.array(right_fingers["position"])

                # If state message has finger data, also try to use it
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
            # Save ongoing episode (if any)
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
            self._print_and_say(f"Recording complete: {self.data_exporter.save_dir}", say=False)
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}")

        # Close ZED camera
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

                    # 4. If not recording, still grab frames for preview
                    if self.zed_recorder and not self.zed_recorder.is_recording:
                        self.zed_recorder.grab_frame_for_preview()

                    end_time = time.monotonic()

                self.rate.sleep()

                # Check if visualization window was closed
                if self.zed_recorder and self.zed_recorder.stop_visualization_flag:
                    print("[INFO] Visualization window closed, continuing to run...")
                    # Visualization closure does not affect data collection

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            # User triggered keyboard interrupt, mark ongoing episode as discarded
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()
            # Discard ZED camera recording
            if self.zed_recorder and self.zed_recorder.is_recording:
                self.zed_recorder.discard_recording()

        finally:
            self.save_and_cleanup()


def main(config: DataExporterConfig, enable_visualization: bool = False):
    rclpy.init(args=None)
    node = rclpy.create_node("data_exporter")

    # Initialize robot model to get joint names
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    g1_rm = g1.instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    # Create HDF5 data exporter
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
        # ZED Mini camera parameters
        enable_zed=True,
        zed_resolution="HD720",  # ZED Mini supported resolutions: HD2K, HD1080, HD720, VGA
        zed_fps=30,             # Frame rate: 15, 30, 60 (depends on resolution)
        zed_record_depth=True,   # Whether to record depth information
        # Visualization parameters
        enable_visualization=enable_visualization,
    )

    print("\n" + "="*60)
    print("G1 Data Collector + ZED Mini Camera Recording Started")
    print("="*60)
    print(f"Save directory: {save_dir}")
    print(f"File format:")
    print(f"  - episode_X.hdf5 (robot state data)")
    print(f"  - episode_X.svo2 (ZED Mini video)")
    print(f"Task description: {config.task_prompt}")
    print(f"Collection frequency: {config.data_collection_frequency} Hz")
    print(f"ZED resolution: HD720, frame rate: 30 FPS")
    print(f"Real-time visualization: {'Enabled' if enable_visualization else 'Disabled'}")
    print("-"*60)
    print("Operation instructions:")
    print("  Press 'c' - Start/stop recording (controls both Pico collection and ZED recording)")
    print("  Press 'x' - Discard current episode (discards both h5 and svo2)")
    if enable_visualization:
        print("  Press 'q' - Close visualization window (does not affect recording)")
    print("  Ctrl+C - Exit program")
    print("="*60 + "\n")

    data_collector.run()


if __name__ == "__main__":
    import argparse

    # First parse visualization-related parameters
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--visualize", action="store_true",
                        help="Enable real-time visualization (left eye image)")

    # Parse known arguments, pass rest to tyro
    args, remaining = parser.parse_known_args()

    # Use tyro to parse DataExporterConfig
    import sys
    sys.argv = [sys.argv[0]] + remaining  # Remove already parsed arguments
    config = tyro.cli(DataExporterConfig)

    # If dataset_name not specified, use timestamp as default name
    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("zed_mini_dataset_%Y%m%d_%H%M%S")

    # If task_prompt not specified, set default value
    if config.task_prompt == "demo":
        config.task_prompt = "ZED Mini Data Collection"

    main(
        config,
        enable_visualization=args.visualize,
    )

