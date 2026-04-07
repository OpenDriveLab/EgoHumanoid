"""
G1 Data Collector - HDF5 format
Uses HDF5 format to save data, each frame with time.time() timestamp
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
        # Subscribe to teleop commands to get finger data for trigger state inference
        self._teleop_cmd_subscriber = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)
        self.rate = self.node.create_rate(self.frequency)

        self.latest_proprio_msg = None
        self.latest_teleop_cmd = None

        self.telemetry = Telemetry(window_size=100)

        # ZED camera TCP client
        self.zed_camera_host = zed_camera_host
        self.zed_camera_port = zed_camera_port
        self.zed_camera_socket = None
        if self.zed_camera_host:
            self._connect_zed_camera()

        print(f"Recording to {self.data_exporter.save_dir}")
    
    def _connect_zed_camera(self):
        """Connect to ZED camera server"""
        try:
            self.zed_camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.zed_camera_socket.connect((self.zed_camera_host, self.zed_camera_port))
            print(f"[INFO] Connected to ZED camera server: {self.zed_camera_host}:{self.zed_camera_port}")
        except Exception as e:
            print(f"[WARN] Unable to connect to ZED camera server {self.zed_camera_host}:{self.zed_camera_port}: {e}")
            self.zed_camera_socket = None

    def _send_zed_command(self, command: str):
        """Send command to ZED camera server"""
        if self.zed_camera_socket is None:
            return

        try:
            self.zed_camera_socket.send(command.encode('utf-8'))
            response = self.zed_camera_socket.recv(1024).decode('utf-8').strip()
            print(f"[INFO] ZED camera response: {response}")
            return response
        except Exception as e:
            print(f"[WARN] Failed to send ZED camera command: {e}")
            # Attempt to reconnect
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
                # If previously interrupted by B button, keep index unchanged
                if self.data_exporter.interrupted_by_b_button:
                    self.data_exporter.interrupted_by_b_button = False
                    self._print_and_say(f"Resume recording episode {episode_index} (keeping index unchanged)")
                else:
                    self._print_and_say(f"Started recording episode {episode_index}")
                # Send start collection command to ZED camera
                self._send_zed_command(f"start_{episode_index}")
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._print_and_say("Stopping recording, preparing to save")
                # Send stop collection command to ZED camera
                self._send_zed_command("STOP")
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state")
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                # B button interrupt: discard current episode, keep index unchanged, wait for next A button to resume
                self.data_exporter.save_episode_as_discarded(keep_index=True)
                self.data_exporter.interrupted_by_b_button = True
                self._episode_state.reset_state()
                self._print_and_say("B button interrupt: Discarded current episode, waiting for next A button to resume (keeping index unchanged)")
                # Send discard command to ZED camera
                self._send_zed_command("DISCARD")
                # Send stop collection command to ZED camera
                self._send_zed_command("STOP")

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
            # Ensure ZED camera has stopped (may have been sent during keyboard input, but ensure again here)
            self._send_zed_command("STOP")
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

        # Send exit command to ZED camera
        self._send_zed_command("QUIT")

        # Close ZED camera connection
        if self.zed_camera_socket:
            try:
                self.zed_camera_socket.close()
            except Exception as e:
                print(f"[WARN] Error closing ZED camera connection: {e}")

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
            # User triggered keyboard interrupt, mark ongoing episode as discarded
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()
            # Send stop command to ZED camera
            self._send_zed_command("STOP")

        finally:
            self.save_and_cleanup()


def main(config: DataExporterConfig):
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
    print("HDF5 Data Collector Started")
    print("="*50)
    print(f"Save directory: {save_dir}")
    print(f"File format: episode_X.hdf5 (each episode saved separately)")
    print(f"Task description: {config.task_prompt}")
    print(f"Collection frequency: {config.data_collection_frequency} Hz")
    print("-"*50)
    print("Operation instructions:")
    print("  Press 'c' - Start/stop recording")
    print("  Press 'x' - Discard current episode")
    print("  Ctrl+C - Exit program")
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
