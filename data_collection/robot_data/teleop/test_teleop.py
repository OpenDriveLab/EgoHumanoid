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
    """VR data debugger - for checking VR command reception"""

    def __init__(self, enable_debug=True, print_interval=1.0):
        """
        Args:
            enable_debug: Whether to enable debug output
            print_interval: Print interval (seconds)
        """
        self.enable_debug = enable_debug
        self.print_interval = print_interval
        self.last_print_time = 0.0

        # Data statistics
        self.data_received_count = 0
        self.last_data = None
        self.data_change_count = 0

        # Data validity check
        self.wrist_pos_history = {"left": deque(maxlen=10), "right": deque(maxlen=10)}
        self.nav_cmd_history = deque(maxlen=10)

    def check_data(self, data: dict, iteration: int):
        """Check and display VR data

        Args:
            data: data dictionary returned by teleop_policy
            iteration: current iteration count
        """
        if not self.enable_debug:
            return

        self.data_received_count += 1
        t_now = time.monotonic()

        # Check if data has changed
        data_changed = False
        if self.last_data is not None:
            data_changed = self._check_data_changed(data, self.last_data)
            if data_changed:
                self.data_change_count += 1

        self.last_data = data.copy() if data else None

        # Periodically print detailed information
        if t_now - self.last_print_time >= self.print_interval:
            self._print_debug_info(data, iteration, data_changed)
            self.last_print_time = t_now

    def _check_data_changed(self, current: dict, last: dict) -> bool:
        """Check if data has changed"""
        # Check key fields
        key_fields = ["wrist_pose", "navigate_cmd", "base_height_command", "target_upper_body_pose"]

        for key in key_fields:
            if key in current and key in last:
                curr_val = np.array(current[key])
                last_val = np.array(last[key])
                if not np.allclose(curr_val, last_val, atol=1e-4):
                    return True

        return False

    def _print_debug_info(self, data: dict, iteration: int, data_changed: bool):
        """Print debug information"""
        print("\n" + "="*60)
        print(f"[VR Debug] Iteration: {iteration} | Data received count: {self.data_received_count}")
        print(f"[VR Debug] Data change count: {self.data_change_count} | This data changed: {'Yes' if data_changed else 'No'}")
        print("-"*60)

        if data is None:
            print("[Warning] No data received!")
            return

        # Check teleop_policy activation status
        if hasattr(data, 'get'):
            is_active = getattr(data, 'is_active', None)
            if is_active is not None:
                print(f"[Status] Teleop Policy activation status: {'Activated' if is_active else 'Not activated'}")

        # Check wrist position
        if "wrist_pose" in data:
            wrist_pose = np.array(data["wrist_pose"])
            if len(wrist_pose) >= 14:
                left_pos = wrist_pose[0:3]
                right_pos = wrist_pose[7:10]
                print(f"[Wrist Position] Left: [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]")
                print(f"[Wrist Position] Right: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")

                # Check if position is valid (non-zero and not at origin)
                left_valid = not np.allclose(left_pos, 0, atol=0.01)
                right_valid = not np.allclose(right_pos, 0, atol=0.01)
                print(f"[Validity] Left hand position: {'Valid' if left_valid else 'Invalid (near origin)'}")
                print(f"[Validity] Right hand position: {'Valid' if right_valid else 'Invalid (near origin)'}")
            else:
                print(f"[Warning] wrist_pose length abnormal: {len(wrist_pose)} (expected 14)")
        else:
            print("[Warning] Missing wrist_pose field")

        # Check finger data
        left_fingers_present = "left_fingers" in data
        right_fingers_present = "right_fingers" in data
        print(f"[Finger Data] Left: {'Present' if left_fingers_present else 'Missing'} | Right: {'Present' if right_fingers_present else 'Missing'}")

        if left_fingers_present and isinstance(data["left_fingers"], dict):
            if "position" in data["left_fingers"]:
                left_fingers = np.array(data["left_fingers"]["position"])
                # Check if any finger is closed (trigger pressed)
                index_finger = left_fingers[4 + 5, 0, 3] if left_fingers.shape[0] > 9 else 0
                print(f"[Finger Status] Left index finger: {'Closed' if index_finger > 0.5 else 'Open'}")

        if right_fingers_present and isinstance(data["right_fingers"], dict):
            if "position" in data["right_fingers"]:
                right_fingers = np.array(data["right_fingers"]["position"])
                index_finger = right_fingers[4 + 5, 0, 3] if right_fingers.shape[0] > 9 else 0
                print(f"[Finger Status] Right index finger: {'Closed' if index_finger > 0.5 else 'Open'}")

        # Check navigation command
        if "navigate_cmd" in data:
            nav_cmd = np.array(data["navigate_cmd"])
            if len(nav_cmd) >= 3:
                print(f"[Navigation Command] Linear velocity: [{nav_cmd[0]:.3f}, {nav_cmd[1]:.3f}] | Angular velocity: {nav_cmd[2]:.3f}")
                nav_active = not np.allclose(nav_cmd, 0, atol=0.01)
                print(f"[Validity] Navigation command: {'Active' if nav_active else 'Stationary'}")
            else:
                print(f"[Warning] navigate_cmd length abnormal: {len(nav_cmd)} (expected 3)")
        else:
            print("[Warning] Missing navigate_cmd field")

        # Check base height command
        if "base_height_command" in data:
            height_cmd = data["base_height_command"]
            if isinstance(height_cmd, (list, np.ndarray)):
                height_val = height_cmd[0] if len(height_cmd) > 0 else height_cmd
            else:
                height_val = height_cmd
            print(f"[Base Height] Command value: {height_val:.3f}")
        else:
            print("[Warning] Missing base_height_command field")

        # Check target joint angles
        if "target_upper_body_pose" in data:
            target_pose = np.array(data["target_upper_body_pose"])
            print(f"[Target Joints] Joint count: {len(target_pose)}")
            if len(target_pose) > 0:
                print(f"[Target Joints] Range: [{target_pose.min():.3f}, {target_pose.max():.3f}]")
                print(f"[Target Joints] Mean: {target_pose.mean():.3f}")
        else:
            print("[Warning] Missing target_upper_body_pose field")

        # Check timestamp
        if "timestamp" in data:
            print(f"[Timestamp] {data['timestamp']:.6f}")

        # Check data source
        if hasattr(data, 'get'):
            source = data.get('source', 'unknown')
            print(f"[Data Source] {source}")

        print("="*60 + "\n")

    def print_summary(self):
        """Print statistics summary"""
        if not self.enable_debug:
            return

        print("\n" + "="*60)
        print("[VR Debug Summary]")
        print(f"  Total data received count: {self.data_received_count}")
        print(f"  Data change count: {self.data_change_count}")
        if self.data_received_count > 0:
            change_rate = self.data_change_count / self.data_received_count * 100
            print(f"  Data change rate: {change_rate:.2f}%")
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

    # Initialize VR data debugger
    # Can control debug output via environment variable or config
    import os
    enable_debug = os.getenv("VR_DEBUG", "true").lower() == "true"
    debug_print_interval = float(os.getenv("VR_DEBUG_INTERVAL", "1.0"))  # Default print once per second
    vr_debugger = VRDataDebugger(enable_debug=enable_debug, print_interval=debug_print_interval)

    print("\n" + "="*60)
    print("[VR Debug] Debug mode enabled")
    print(f"[VR Debug] Print interval: {debug_print_interval}s")
    print(f"[VR Debug] Control frequency: {config.teleop_frequency}Hz")
    print("="*60 + "\n")

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                # Get the current teleop action
                with telemetry.timer("get_action"):
                    data = teleop_policy.get_action()

                    # Check VR data
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
                    print("[Warning] teleop_policy.get_action() returned None or empty data!")

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
        # Print debug summary
        vr_debugger.print_summary()
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(TeleopConfig)
    main(config)
