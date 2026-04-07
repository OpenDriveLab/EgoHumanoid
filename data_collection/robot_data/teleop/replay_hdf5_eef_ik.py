"""
HDF5 EEF Replay Player (IK Version) - Real Robot Version

Based on replay_hdf5_real.py, but uses IK to calculate joint angles from EEF poses

Differences from replay_hdf5_real.py:
- Reads action_eef (end-effector poses) from dataset
- Uses TeleopRetargetingIK's IK solver to calculate joint angles
- Only controls upper body, base controlled by default values

Warning: This script controls a real robot, ensure safety measures are in place

Usage:
1. Run the script
2. Press ] to activate policy
3. Press p to enter preparation phase (slowly move to first frame position)
4. Wait for preparation to complete, then press l to start replay

Usage examples:
    python decoupled_wbc/control/main/teleop/replay_hdf5_eef_ik.py --dataset-dir outputs/data_1224_0
    python decoupled_wbc/control/main/teleop/replay_hdf5_eef_ik.py --dataset-dir outputs/data_1224_0 --episode 0 --no-with-hands
    python decoupled_wbc/control/main/teleop/replay_hdf5_eef_ik.py --dataset-dir outputs/data_1224_0 --speed 0.5 --no-with-hands
"""

from copy import deepcopy
from dataclasses import dataclass
import glob
import os
import time
from typing import Optional
import sys
sys.path.insert(0, '/root/Projects/GR00T-WholeBodyControl')
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
class ReplayEEFIKConfig(ControlLoopConfig):
    """EEF IK replay configuration, inherits from ControlLoopConfig"""

    dataset_dir: str = "outputs/data_1224_0"
    """Dataset directory path"""

    episode: Optional[int] = None
    """Episode index to replay, None means replay all"""

    speed: float = 1.0
    """Replay speed multiplier"""

    loop: bool = False
    """Whether to loop replay"""

    # Real robot default settings
    interface: str = "real"
    """Interface type, real robot uses 'real'"""

    require_confirmation: bool = True
    """Whether confirmation is required to start"""

    preparation_duration: float = 3.0
    """Preparation phase duration (seconds), time for arm to slowly move to first frame position"""


class HDF5EEFProvider:
    """
    Provides EEF poses from HDF5 dataset, calculates joint angles using IK

    Press 'p' key to enter preparation phase (slowly move to first frame position)
    Press 'l' key to start replay
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

        # Initialize IK solver
        left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
        self.retargeting_ik = TeleopRetargetingIK(
            robot_model=robot_model,
            left_hand_ik_solver=left_hand_ik_solver,
            right_hand_ik_solver=right_hand_ik_solver,
            enable_visualization=False,
            body_active_joint_groups=["upper_body"],
        )

        # Get wrist frame names
        self.left_wrist_frame = robot_model.supplemental_info.hand_frame_names["left"]
        self.right_wrist_frame = robot_model.supplemental_info.hand_frame_names["right"]

        # Replay state
        self.is_replaying = False
        self.is_preparing = False
        self.preparation_complete = False
        self.current_episode_idx = 0
        self.current_frame_idx = 0
        self.last_frame_time = None
        self.frame_interval = None

        # Preparation phase related
        self.preparation_start_time = None
        self.preparation_start_upper_body = None
        self.preparation_target_upper_body = None
        self.preparation_target_wrist = None

        print(f"[HDF5 EEF] Loaded {len(self.episodes)} episodes")
        print(f"[HDF5 EEF] Replay speed: {speed}x")
        print(f"[HDF5 EEF] Preparation time: {preparation_duration}s")
        print(f"[HDF5 EEF] Press 'p' key to enter preparation phase (slowly move to first frame)")
        print(f"[HDF5 EEF] Press 'l' key to start/pause replay")
    
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
                        "num_frames": f.attrs.get("num_frames", f["action_eef"].shape[0]),
                    }

                    # Load EEF poses
                    if "action_eef" in f:
                        ep["eef_pose"] = np.array(f["action_eef"])
                    elif "observation_eef_state" in f:
                        ep["eef_pose"] = np.array(f["observation_eef_state"])
                    else:
                        raise ValueError(f"action_eef or observation_eef_state not found in {hdf5_file}")

                    # Load navigation and height commands (optional)
                    if "teleop_navigate_command" in f:
                        ep["navigate_cmd"] = np.array(f["teleop_navigate_command"])

                    if "teleop_base_height_command" in f:
                        ep["base_height_command"] = np.array(f["teleop_base_height_command"])

                    episodes.append(ep)
                    print(f"  Loaded: {ep['file']} - {ep['num_frames']} frames, {ep['fps']} FPS")
            except Exception as e:
                print(f"  [WARN] Loading {hdf5_file} failed: {e}")

        if episode is not None:
            if episode >= len(episodes):
                raise ValueError(f"Episode {episode} does not exist, total {len(episodes)}")
            episodes = [episodes[episode]]

        return episodes
    
    def _eef_to_transformation_matrix(self, eef_data: np.ndarray) -> dict:
        """
        Convert EEF data to transformation matrix dictionary

        Args:
            eef_data: (14,) [left: x,y,z,qw,qx,qy,qz, right: x,y,z,qw,qx,qy,qz]
                     Note: quaternion format is scalar_first=True, i.e., (w, x, y, z)

        Returns:
            dict: {left_wrist_frame: T_left, right_wrist_frame: T_right}
        """
        # Extract left and right hand data
        left_pos = eef_data[:3]
        left_quat_scalar_first = eef_data[3:7]  # (qw, qx, qy, qz) - scalar_first format
        right_pos = eef_data[7:10]
        right_quat_scalar_first = eef_data[10:14]  # (qw, qx, qy, qz) - scalar_first format

        # Convert to scipy expected format (x, y, z, w)
        # scalar_first (w, x, y, z) -> vector_first (x, y, z, w)
        left_quat = np.array([left_quat_scalar_first[1], left_quat_scalar_first[2],
                             left_quat_scalar_first[3], left_quat_scalar_first[0]])
        right_quat = np.array([right_quat_scalar_first[1], right_quat_scalar_first[2],
                              right_quat_scalar_first[3], right_quat_scalar_first[0]])

        # Convert to transformation matrix (use directly, do not apply additional rotation)
        # action_eef data has already been processed by WristsPreProcessor, no additional rotation needed
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
    
    def _compute_upper_body_joints_from_eef(self, eef_data: np.ndarray) -> np.ndarray:
        """
        Calculate upper body joint angles from EEF pose

        Uses the same approach as TeleopPolicy: through set_goal() and get_action() to maintain state continuity
        This ensures BodyIKSolver's state continuity (continue from last result, not reset every time)

        Args:
            eef_data: (14,) EEF pose

        Returns:
            (N,) Upper body joint angles
        """
        # Convert to transformation matrix
        body_data = self._eef_to_transformation_matrix(eef_data)

        # Use the same approach as TeleopPolicy: set_goal() + get_action()
        # Note: left_hand_data and right_hand_data passed as None (do not use hand IK)
        # In _inverse_kinematics, check if is not None to decide whether to call hand IK solver
        ik_data = {
            "body_data": body_data,
            "left_hand_data": None,  # type: ignore  # Do not use hand IK
            "right_hand_data": None,  # type: ignore  # Do not use hand IK
        }
        self.retargeting_ik.set_goal(ik_data)
        # get_action() returns a numpy array (upper body joint angles)
        # Note: Although type annotation is dict[str, any], it actually returns a numpy array
        target_joints = np.array(self.retargeting_ik.get_action())  # type: ignore

        return target_joints
    
    def handle_keyboard_button(self, key):
        """Handle keyboard input"""
        if key == "p":
            if not self.is_preparing and not self.is_replaying:
                self.is_preparing = True
                self.preparation_complete = False
                self.preparation_start_time = time.time()
                self.preparation_start_upper_body = None
                self.preparation_target_upper_body = None
                self.preparation_target_wrist = None
                print("\n[HDF5 EEF] 🔄 Entering preparation phase: slowly moving to first frame position...")
                print(f"[HDF5 EEF] Preparation time: {self.preparation_duration}s, press 'l' to start replay after completion")
        elif key == "l":
            if self.is_preparing:
                print("\n[HDF5 EEF] ⚠️ Preparation phase incomplete, please wait for completion before pressing 'l'")
                return

            if not self.preparation_complete and not self.is_replaying:
                print("\n[HDF5 EEF] ⚠️ Please first press 'p' to complete preparation phase")
                return

            self.is_replaying = not self.is_replaying
            if self.is_replaying:
                self.preparation_complete = False
                print("\n[HDF5 EEF] ▶ Starting replay")
                self.current_frame_idx = 0
                self.last_frame_time = time.time()
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
            else:
                print("\n[HDF5 EEF] ⏸ Pausing replay")
    
    def get_command(self, robot_model, current_obs=None) -> Optional[dict]:
        """
        Get current frame command (calculated through IK)
        """
        if len(self.episodes) == 0:
            return None

        # Preparation phase: slowly move to first frame position
        if self.is_preparing:
            if current_obs is None:
                print("[HDF5 EEF] Warning: Preparation phase needs current_obs, but not provided")
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
                # Calculate target joint angles from first frame EEF
                first_frame_eef = ep["eef_pose"][0]
                self.preparation_target_upper_body = self._compute_upper_body_joints_from_eef(first_frame_eef)
                self.preparation_target_wrist = first_frame_eef.copy()

                print(f"[HDF5 EEF] Starting position recorded, target position set")

            # Calculate interpolation progress
            if elapsed >= self.preparation_duration:
                progress = 1.0
                self.is_preparing = False
                self.preparation_complete = True
                print(f"\n[HDF5 EEF] ✓ Preparation complete! Press 'l' to start replay")
            else:
                t = elapsed / self.preparation_duration
                progress = t * t * (3.0 - 2.0 * t)  # smoothstep

            # Interpolate upper arm position
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
                "wrist_pose": self.preparation_target_wrist,  # Use first frame wrist position
                "navigate_cmd": DEFAULT_NAV_CMD,
                "base_height_command": DEFAULT_BASE_HEIGHT,
            }

            if int(elapsed * 10) % 10 == 0:
                print(f"\r[HDF5 EEF] Preparing... {elapsed:.1f}/{self.preparation_duration:.1f}s ({progress*100:.1f}%)",
                      end="", flush=True)

            return cmd

        # Preparation complete, waiting to start replay: continuously send first frame command
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

        # Replay phase
        if not self.is_replaying:
            return None

        # Check if need to advance to next frame
        now = time.time()
        if self.last_frame_time is not None and self.frame_interval is not None:
            if now - self.last_frame_time >= self.frame_interval:
                self.current_frame_idx += 1
                self.last_frame_time = now

        ep = self.episodes[self.current_episode_idx]

        # Check if current episode finished
        if self.current_frame_idx >= ep["num_frames"]:
            self.current_frame_idx = 0
            self.current_episode_idx += 1

            if self.current_episode_idx >= len(self.episodes):
                print("\n[HDF5 EEF] ✓ All episodes replay complete")
                self.is_replaying = False
                self.current_episode_idx = 0
                return None
            else:
                ep = self.episodes[self.current_episode_idx]
                self.frame_interval = 1.0 / (ep["fps"] * self.speed)
                print(f"\n[HDF5 EEF] Switching to Episode {self.current_episode_idx + 1}/{len(self.episodes)}")

        # Get current frame EEF
        frame_idx = self.current_frame_idx
        eef_data = ep["eef_pose"][frame_idx]

        # Use IK to calculate upper body joint angles
        # Note: Using set_goal() + get_action() approach automatically maintains BodyIKSolver state continuity
        upper_body_joints = self._compute_upper_body_joints_from_eef(eef_data)

        # Build command
        cmd = {
            "target_upper_body_pose": upper_body_joints,
            "wrist_pose": eef_data,
            "navigate_cmd": ep.get("navigate_cmd", DEFAULT_NAV_CMD)[frame_idx] if "navigate_cmd" in ep else DEFAULT_NAV_CMD,
            "base_height_command": ep.get("base_height_command", DEFAULT_BASE_HEIGHT)[frame_idx] if "base_height_command" in ep else DEFAULT_BASE_HEIGHT,
        }

        if isinstance(cmd["base_height_command"], np.ndarray):
            cmd["base_height_command"] = cmd["base_height_command"].item() if cmd["base_height_command"].size == 1 else cmd["base_height_command"][0]

        # Print progress
        if frame_idx % 20 == 0:
            progress = (frame_idx + 1) / ep["num_frames"] * 100
            nav = cmd["navigate_cmd"]
            print(
                f"\r[HDF5 EEF] Ep {self.current_episode_idx + 1}/{len(self.episodes)} | "
                f"Frame {frame_idx + 1}/{ep['num_frames']} ({progress:.1f}%) | "
                f"Nav: [{nav[0]:+.2f}, {nav[1]:+.2f}, {nav[2]:+.2f}]",
                end="", flush=True
            )

        return cmd


def print_safety_warning():
    """Print safety warning"""
    print("\n" + "!" * 70)
    print("!" + " " * 68 + "!")
    print("!" + "  ⚠️  WARNING: About to replay EEF data on real robot (IK version)!".center(68) + "!")
    print("!" + " " * 68 + "!")
    print("!" * 70)
    print("""
Please ensure:
  1. No obstacles or personnel around the robot
  2. Emergency stop switch is within reach and functional
  3. EEF data motion amplitude is within safe range
  4. Replay speed is set appropriately
  5. You are familiar with how to perform emergency stop

Emergency stop methods:
  - Press keyboard 'o' to deactivate policy
  - Press physical emergency stop button
  - Ctrl+C to terminate program
""")
    print("!" * 70 + "\n")


def main(config: ReplayEEFIKConfig):
    # Safety confirmation
    print_safety_warning()

    if config.require_confirmation:
        confirm = input("Confirm running replay on real robot? Enter 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Cancelled")
            return

    ros_manager = ROSManager(node_name="ReplayEEFIKControlLoop")
    node = ros_manager.node

    # Start configuration server
    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())

    # Status publishers
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

    # EEF provider (includes IK solver)
    eef_provider = HDF5EEFProvider(
        dataset_dir=config.dataset_dir,
        robot_model=robot_model,
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
    dispatcher.register(eef_provider)
    dispatcher.start()

    rate = node.create_rate(config.control_frequency)

    print("\n" + "=" * 60)
    print("HDF5 EEF IK Replay Control Loop")
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

                # Get command (calculated through IK)
                with telemetry.timer("get_command"):
                    replay_cmd = eef_provider.get_command(robot_model, current_obs=obs)

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

                # Publish status
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

            if config.verbose_timing:
                telemetry.log_timing_info(context="Replay EEF IK Control Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.control_frequency):
                telemetry.log_timing_info(context="Replay EEF IK Control Loop Missed", threshold=0.001)

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
    config = tyro.cli(ReplayEEFIKConfig)
    main(config)

