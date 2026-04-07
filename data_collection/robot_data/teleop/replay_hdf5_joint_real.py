"""
HDF5 Data Replay - Real Robot Version

Based on run_g1_control_loop.py structure, replays HDF5 dataset on real robot

⚠️ Warning: This script controls a real robot, ensure:
   1. No obstacles around the robot
   2. Emergency stop within reach
   3. Dataset speed and amplitude within safe range

Usage:
1. Run script
2. Press ] to activate policy
3. Press p to start replaying dataset
4. Press o to deactivate policy anytime

Usage examples:
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
    """Real robot replay config, inherits from ControlLoopConfig"""
    
    dataset_dir: str = "outputs/data_1224_0"
    """Dataset directory path"""
    
    episode: Optional[int] = None
    """Episode index to replay, None means replay all"""
    
    speed: float = 1.0
    """Replay speed multiplier (default 0.5x safer for real robot)"""
    
    loop: bool = False
    """Whether to loop replay"""
    
    # Real robot default settings
    interface: str = "real"
    """Interface type, real robot uses 'real'"""
    
    require_confirmation: bool = True
    """Whether confirmation required to start"""
    
    preparation_duration: float = 3.0
    """Preparation phase duration (seconds), time for arm to slowly move to first frame position"""


class HDF5CommandProvider:
    """
    Provide command from HDF5 dataset, replacing ROS subscription
    
    Press 'p' key to enter preparation phase (slowly move to first frame position)
    Press 'l' key to start replay
    """
    
    def __init__(self, dataset_dir: str, episode: Optional[int] = None, speed: float = 1.0,
                 preparation_duration: float = 3.0):
        self.episodes = self._load_episodes(dataset_dir, episode)
        self.speed = speed
        self.preparation_duration = preparation_duration  # Preparation phase duration (seconds)

        # Replay state
        self.is_replaying = False
        self.is_preparing = False  # Preparation phase (slowly move to first frame)
        self.preparation_complete = False  # Preparation phase complete, waiting to start replay
        self.current_episode_idx = 0
        self.current_frame_idx = 0
        self.last_frame_time = None
        self.frame_interval = None

        # Preparation phase related
        self.preparation_start_time = None
        self.preparation_start_upper_body = None  # Upper arm position at preparation start
        self.preparation_target_upper_body = None  # Target upper arm position for first frame
        self.preparation_target_wrist = None  # Target wrist position for first frame
        
        print(f"[HDF5] Loaded {len(self.episodes)} episodes")
        print(f"[HDF5] Replay speed: {speed}x")
        print(f"[HDF5] Preparation time: {preparation_duration}s")
        print(f"[HDF5] Press 'p' key to enter preparation phase (slowly move to first frame)")
        print(f"[HDF5] Press 'l' key to start/pause replay")
    
    def _load_episodes(self, dataset_dir: str, episode: Optional[int]) -> list:
        """Load episode data"""
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Data directory does not exist: {dataset_dir}")
        
        hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
        if len(hdf5_files) == 0:
            single_file = os.path.join(dataset_dir, "data.hdf5")
            if os.path.exists(single_file):
                hdf5_files = [single_file]
            else:
                raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")
        
        episodes = []
        for hdf5_file in hdf5_files:
            try:
                with h5py.File(hdf5_file, "r") as f:
                    ep = {
                        "file": os.path.basename(hdf5_file),
                        "fps": f.attrs.get("fps", 20),
                        "num_frames": f.attrs.get("num_frames", len(f["observation_state"])),
                    }

                    # Load control commands
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
                    print(f"  Loaded: {ep['file']} - {ep['num_frames']} frames, {ep['fps']} FPS")
            except Exception as e:
                print(f"  [WARN] Loading {hdf5_file} failed: {e}")
        
        if episode is not None:
            if episode >= len(episodes):
                raise ValueError(f"Episode {episode} does not exist, total {len(episodes)} ")
            episodes = [episodes[episode]]
        
        return episodes
    
    def _interpolate_wrist_pose(self, start_pose: np.ndarray, target_pose: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate wrist pose (7 dims each for left/right: [x, y, z, qx, qy, qz, qw])
        
        Use SLERP to interpolate quaternions
        """
        result = np.zeros_like(start_pose)
        
        # Process left and right hands
        for hand_idx in range(2):
            start_idx = hand_idx * 7
            end_idx = start_idx + 7
            
            # Position (linear interpolation)
            start_pos = start_pose[start_idx:start_idx + 3]
            target_pos = target_pose[start_idx:start_idx + 3]
            result[start_idx:start_idx + 3] = start_pos * (1 - t) + target_pos * t
            
            # Quaternion (simple linear interpolation then normalize, sufficient for preparation phase)
            start_quat = start_pose[start_idx + 3:end_idx]  # (qx, qy, qz, qw)
            target_quat = target_pose[start_idx + 3:end_idx]
            
            # Simple linear interpolation (then normalize)
            interp_quat = start_quat * (1 - t) + target_quat * t
            # Normalize
            norm = np.linalg.norm(interp_quat)
            if norm > 1e-6:
                interp_quat = interp_quat / norm
            result[start_idx + 3:end_idx] = interp_quat
        
        return result
    
    def handle_keyboard_button(self, key):
        """Handle keyboard input"""
        if key == "p":
            # Enter preparation phase
            if not self.is_preparing and not self.is_replaying:
                self.is_preparing = True
                self.preparation_complete = False  # Reset preparation complete flag
                self.preparation_start_time = time.time()
                # Reset preparation state variables
                self.preparation_start_upper_body = None
                self.preparation_start_wrist = None
                self.preparation_target_upper_body = None
                self.preparation_target_wrist = None
                print("\n[HDF5] 🔄 Entering preparation phase: slowly moving to first frame position...")
                print(f"[HDF5] Preparation time: {self.preparation_duration}s, press 'l' to start replay after completion")
        elif key == "l":
            # Start/pause replay
            if self.is_preparing:
                print("\n[HDF5] ⚠️ Preparation phase incomplete, please wait for completion before pressing 'l'")
                return

            if not self.preparation_complete and not self.is_replaying:
                print("\n[HDF5] ⚠️ Please first press 'p' to complete preparation phase")
                return

            self.is_replaying = not self.is_replaying
            if self.is_replaying:
                self.preparation_complete = False  # Clear preparation complete flag after starting replay
                print("\n[HDF5] ▶ Starting replay")
                self.current_frame_idx = 0  # Reset to first frame
                self.last_frame_time = time.time()
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
            else:
                print("\n[HDF5] ⏸ Pausing replay")
    
    def get_command(self, robot_model, current_obs=None) -> Optional[dict]:
        """
        Get current frame command

        Args:
            robot_model: Robot model
            current_obs: Current observation (used to get current arm position during preparation phase)

        Returns None to use default command (not started replay or replay finished)
        Returns dict for command from dataset or interpolated command from preparation phase
        """
        if len(self.episodes) == 0:
            return None

        # Preparation phase: slowly move to first frame position
        if self.is_preparing:
            if current_obs is None:
                print("[HDF5] Warning: Preparation phase needs current_obs, but not provided")
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
                if "observation_state" in ep:
                    self.preparation_target_upper_body = ep["observation_state"][0][upper_body_indices].copy()
                else:
                    # If no observation data, use current position
                    self.preparation_target_upper_body = self.preparation_start_upper_body.copy()

                # Record first frame wrist position
                if "wrist_pose" in ep:
                    self.preparation_target_wrist = ep["wrist_pose"][0].copy()
                else:
                    self.preparation_target_wrist = np.array(DEFAULT_WRIST_POSE).copy()

                # Record starting wrist position (from current observation)
                self.preparation_start_wrist = current_obs.get("wrist_pose", np.array(DEFAULT_WRIST_POSE)).copy()

                print(f"[HDF5] Starting position recorded, target position set")

            # Calculate interpolation progress (using smooth curve)
            if elapsed >= self.preparation_duration:
                # Preparation complete
                progress = 1.0
                self.is_preparing = False
                self.preparation_complete = True  # Mark preparation complete
                print(f"\n[HDF5] ✓ Preparation complete! Press 'l' to start replay")
            else:
                # Use smooth interpolation (ease-in-out)
                t = elapsed / self.preparation_duration
                progress = t * t * (3.0 - 2.0 * t)  # smoothstep

            # Interpolate upper arm position
            upper_body_indices = robot_model.get_joint_group_indices("upper_body")
            if (self.preparation_start_upper_body is not None and
                self.preparation_target_upper_body is not None):
                interp_upper_body = (
                    self.preparation_start_upper_body * (1 - progress) +
                    self.preparation_target_upper_body * progress
                )
            else:
                # If not initialized yet, use target position
                interp_upper_body = (
                    self.preparation_target_upper_body
                    if self.preparation_target_upper_body is not None
                    else current_obs["q"][upper_body_indices]
                )

            # Interpolate wrist position (using SLERP for quaternions)
            if (self.preparation_start_wrist is not None and
                self.preparation_target_wrist is not None):
                interp_wrist = self._interpolate_wrist_pose(
                    self.preparation_start_wrist,
                    self.preparation_target_wrist,
                    progress
                )
            else:
                # If not initialized yet, use target position
                interp_wrist = self.preparation_target_wrist if self.preparation_target_wrist is not None else np.array(DEFAULT_WRIST_POSE)

            # Build command (only move upper arm and wrist, keep others default)
            cmd = {
                "target_upper_body_pose": interp_upper_body,
                "wrist_pose": interp_wrist,
                "navigate_cmd": DEFAULT_NAV_CMD,  # Do not move base during preparation phase
                "base_height_command": DEFAULT_BASE_HEIGHT,
            }

            # Print progress
            if int(elapsed * 10) % 10 == 0:  # Print every 0.1 seconds
                print(f"\r[HDF5] Preparing... {elapsed:.1f}/{self.preparation_duration:.1f}s ({progress*100:.1f}%)",
                      end="", flush=True)

            return cmd

        # Preparation complete, waiting to start replay: continuously send first frame command
        if self.preparation_complete and not self.is_replaying:
            # Use target position already set during preparation phase
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
                # If target position not set (should not happen), use default value
                upper_body_indices = robot_model.get_joint_group_indices("upper_body")
                cmd = {
                    "target_upper_body_pose": current_obs["q"][upper_body_indices] if current_obs else robot_model.get_default_q()[upper_body_indices],
                    "wrist_pose": DEFAULT_WRIST_POSE,
                    "navigate_cmd": DEFAULT_NAV_CMD,
                    "base_height_command": DEFAULT_BASE_HEIGHT,
                }
                return cmd

        # Replay phase
        if not self.is_replaying:
            return None

        # Check if need to advance to next frame
        now = time.time()
        if self.last_frame_time is not None and self.frame_interval is not None:
            if now - self.last_frame_time < self.frame_interval:
                # Not time for next frame yet, return current frame command
                pass
            else:
                # Advance to next frame
                self.current_frame_idx += 1
                self.last_frame_time = now

        ep = self.episodes[self.current_episode_idx]

        # Check if current episode finished
        if self.current_frame_idx >= ep["num_frames"]:
            self.current_frame_idx = 0
            self.current_episode_idx += 1

            if self.current_episode_idx >= len(self.episodes):
                # All episodes finished
                print("\n[HDF5] ✓ All episodes replay complete")
                self.is_replaying = False
                self.current_episode_idx = 0
                return None
            else:
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
                print(f"\n[HDF5] Switching to Episode {self.current_episode_idx + 1}/{len(self.episodes)}")

        # Build command
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
        
        # Print progress
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
    """Print safety warning"""
    print("\n" + "!" * 70)
    print("!" + " " * 68 + "!")
    print("!" + "  ⚠️  WARNING: About to replay data on real robot!".center(68) + "!")
    print("!" + " " * 68 + "!")
    print("!" * 70)
    print("""
Please ensure:
  1. No obstacles or personnel around the robot
  2. Emergency stop switch is within reach and functional
  3. Dataset motion amplitude is within safe range
  4. Replay speed is set appropriately (recommend starting with 0.5x)
  5. You are familiar with how to perform emergency stop

Emergency stop methods:
  - Press keyboard 'o' to deactivate policy
  - Press physical emergency stop button
  - Ctrl+C to terminate program
""")
    print("!" * 70 + "\n")


def main(config: ReplayRealConfig):
    # Safety confirmation
    print_safety_warning()

    if config.require_confirmation:
        confirm = input("Confirm running replay on real robot? Enter 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Cancelled")
            return

    ros_manager = ROSManager(node_name="ReplayRealControlLoop")
    node = ros_manager.node

    # Start configuration server
    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())

    wbc_config = config.load_wbc_yaml()

    # Status publishers (for data collection and monitoring)
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

    # HDF5 command provider
    hdf5_provider = HDF5CommandProvider(
        dataset_dir=config.dataset_dir,
        episode=config.episode,
        speed=config.speed,
        preparation_duration=config.preparation_duration,
    )

    # Keyboard dispatcher
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
    print("HDF5 Real Robot Replay Control Loop")
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
                # Get observation
                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)

                t_now = time.monotonic()

                # Get command (pass current observation for preparation phase)
                with telemetry.timer("get_command"):
                    replay_cmd = hdf5_provider.get_command(robot_model, current_obs=obs)

                # Build wbc_goal
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

                # Get action
                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)

                # Execute action
                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)

                # Publish status (same as run_g1_control_loop.py)
                with telemetry.timer("publish_status"):
                    # Policy status
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

                    # Joint safety status
                    joint_safety_ok = env.get_joint_safety_status()
                    joint_safety_status_msg = {
                        "joint_safety_ok": joint_safety_ok,
                        "timestamp": t_now,
                    }
                    joint_safety_status_pub.publish(joint_safety_status_msg)

                    # Safety warning
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

            # Timing log
            if config.verbose_timing:
                telemetry.log_timing_info(context="Replay Real Control Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.control_frequency):
                telemetry.log_timing_info(context="Replay Real Control Loop Missed", threshold=0.001)

    except ros_manager.exceptions() as e:
        print(f"\nROSManager interrupted: {e}")
    except KeyboardInterrupt:
        print("\nUser interrupted")
    finally:
        print("\nCleaning up...")
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()
        print("Exited safely")


if __name__ == "__main__":
    config = tyro.cli(ReplayRealConfig)
    main(config)

