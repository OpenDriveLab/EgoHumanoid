#!/usr/bin/env python3
"""
PICO + ZED Mini Data Collection Script (PC Integrated Version)

Features:
    1. Real-time full-body tracking data acquisition from PICO device (24 joints)
    2. Real-time full-body skeleton visualization in MeshCat
    3. Real-time left/right controller pose acquisition and visualization
    4. Real-time left/right hand tracking data acquisition and visualization (26 joints)
    5. Simultaneous ZED Mini camera SVO2 video recording
    6. Save two files: HDF5 (PICO data) and SVO2 (ZED video)

Usage:
    python pico_zed_collection.py [--name dataset_name] [--interval 0.05] [--episode 0]
                                  [--data-dir DIR] [--dataset-name NAME] [--visualize-zed]

    Arguments:
    --name: Dataset name (default: body_data)
    --interval: Update interval (seconds, default: 0.01, i.e., 100fps)
    --episode: Starting episode index (default: 0)
    --data-dir: Data save directory (default: data_collection)
    --dataset-name: Dataset directory name (default: uses --name value)
    --visualize-zed, -v: Enable ZED camera real-time visualization (default: disabled)
    --no-depth: Do not record depth information (default: record depth)

    Note: Data will be automatically saved to HDF5 and SVO2 files
    Note: During collection, press SPACE to finish current episode
    Note: Each episode's data will be saved to dataset directory:
          - HDF5: {data_dir}/{dataset_name}/episode_{index}.hdf5
          - SVO2: {data_dir}/{dataset_name}/episode_{index}.svo2

    Examples:
    python pico_zed_collection.py                              # Default parameters
    python pico_zed_collection.py --name my_data              # Specify dataset name
    python pico_zed_collection.py --data-dir my_data          # Specify data directory
    python pico_zed_collection.py --visualize-zed             # Enable ZED visualization

Data Format:

    HDF5 File (PICO data):
    - body_pose: (frames, 24, 7) - Full-body joint poses
    - local_timestamps_ns: (frames,) - Local PC timestamps (nanoseconds)
    - left_controller_pose: (frames, 7) - Left controller pose
    - right_controller_pose: (frames, 7) - Right controller pose
    - left_hand_pose: (frames, 26, 7) - Left hand tracking data
    - right_hand_pose: (frames, 26, 7) - Right hand tracking data
    - left_hand_active: (frames,) - Left hand tracking active status (0=inactive, 1=active)
    - right_hand_active: (frames,) - Right hand tracking active status (0=inactive, 1=active)

    SVO2 File (ZED video):
    - Contains left/right images and depth information
    - Can be played back using ZED SDK

Body Joint Index Definition (24 joints):
    0: Pelvis (Pelvis, root joint)
    1: LEFT_HIP (Left hip)
    2: RIGHT_HIP (Right hip)
    3: SPINE1 (Spine 1)
    4: LEFT_KNEE (Left knee)
    5: RIGHT_KNEE (Right knee)
    6: SPINE2 (Spine 2)
    7: LEFT_ANKLE (Left ankle)
    8: RIGHT_ANKLE (Right ankle)
    9: SPINE3 (Spine 3)
    10: LEFT_FOOT (Left foot)
    11: RIGHT_FOOT (Right foot)
    12: NECK (Neck)
    13: LEFT_COLLAR (Left collar)
    14: RIGHT_COLLAR (Right collar)
    15: HEAD (Head)
    16: LEFT_SHOULDER (Left shoulder)
    17: RIGHT_SHOULDER (Right shoulder)
    18: LEFT_ELBOW (Left elbow)
    19: RIGHT_ELBOW (Right elbow)
    20: LEFT_WRIST (Left wrist)
    21: RIGHT_WRIST (Right wrist)
    22: LEFT_HAND (Left hand)
    23: RIGHT_HAND (Right hand)
"""

import numpy as np
import xrobotoolkit_sdk as xrt
import pyzed.sl as sl
import meshcat
import meshcat.geometry as g
import time
import argparse
from scipy.spatial.transform import Rotation
import h5py
from datetime import datetime
import os
import threading
import sys
import select
import termios
import tty
import cv2


# ========== Full-body skeleton connection relationship definition ==========
def get_body_connections():
    """
    Define the joint connection relationships of the full-body skeleton
    """
    connections = []
    # Spine chain: Pelvis(0) -> Spine1(3) -> Spine2(6) -> Spine3(9) -> Neck(12) -> Head(15)
    connections.extend([(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)])
    # Left leg: Pelvis(0) -> LEFT_HIP(1) -> LEFT_KNEE(4) -> LEFT_ANKLE(7) -> LEFT_FOOT(10)
    connections.extend([(0, 1), (1, 4), (4, 7), (7, 10)])
    # Right leg: Pelvis(0) -> RIGHT_HIP(2) -> RIGHT_KNEE(5) -> RIGHT_ANKLE(8) -> RIGHT_FOOT(11)
    connections.extend([(0, 2), (2, 5), (5, 8), (8, 11)])
    # Left arm: Spine3(9) -> LEFT_COLLAR(13) -> LEFT_SHOULDER(16) -> LEFT_ELBOW(18) -> LEFT_WRIST(20) -> LEFT_HAND(22)
    connections.extend([(9, 13), (13, 16), (16, 18), (18, 20), (20, 22)])
    # Right arm: Spine3(9) -> RIGHT_COLLAR(14) -> RIGHT_SHOULDER(17) -> RIGHT_ELBOW(19) -> RIGHT_WRIST(21) -> RIGHT_HAND(23)
    connections.extend([(9, 14), (14, 17), (17, 19), (19, 21), (21, 23)])
    return connections


# ========== Hand skeleton connection relationship definition ==========
def get_finger_connections():
    """Define the connection relationships between finger joints"""
    connections = []
    # Thumb connections
    connections.extend([(1, 2), (2, 3), (3, 4), (4, 5)])
    # Index finger connections
    connections.extend([(1, 6), (6, 7), (7, 8), (8, 9), (9, 10)])
    # Middle finger connections
    connections.extend([(1, 11), (11, 12), (12, 13), (13, 14), (14, 15)])
    # Ring finger connections
    connections.extend([(1, 16), (16, 17), (17, 18), (18, 19), (19, 20)])
    # Little finger connections
    connections.extend([(1, 21), (21, 22), (22, 23), (23, 24), (24, 25)])
    return connections


# ========== Coordinate transformation matrix definition ==========
RX90 = np.array([[1, 0, 0, 0],
                 [0, 0, -1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])
RZ90 = np.array([[0, 1, 0, 0],
                 [-1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
RZS180 = np.array([[-1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

T_body = RZ90 @ RX90
T_left = RZ90 @ RX90


def transform_coordinates(positions):
    """Apply coordinate system transformation"""
    if len(positions.shape) == 1:
        pose = np.eye(4)
        pose[:3, 3] = positions
        transformed_pose = T_body @ pose
        transformed_pos = transformed_pose[:3, 3].copy()
        transformed_pos[2] += 0.7
        transformed_pos[0] += 0.55
        return transformed_pos
    else:
        transformed = np.zeros_like(positions)
        for i in range(len(positions)):
            pose = np.eye(4)
            pose[:3, 3] = positions[i]
            transformed_pose = T_body @ pose
            transformed_pos = transformed_pose[:3, 3].copy()
            transformed_pos[2] += 0.7
            transformed_pos[0] += 0.55
            transformed[i] = transformed_pos
        return transformed


def transform_coordinates_hand(data, is_right_hand=False):
    """Apply hand coordinate system transformation"""
    if len(data.shape) == 1:
        pose = np.eye(4)
        pose[:3, 3] = data
        if is_right_hand:
            transformed_pose = T_left @ pose @ RZS180
        else:
            transformed_pose = T_left @ pose
        transformed_pos = transformed_pose[:3, 3].copy()
        transformed_pos[2] += 0.7
        transformed_pos[0] += 0.55
        return transformed_pos
    else:
        transformed = np.zeros_like(data)
        for i in range(len(data)):
            pose = np.eye(4)
            pose[:3, 3] = data[i]
            if is_right_hand:
                transformed_pose = T_left @ pose @ RZS180
            else:
                transformed_pose = T_left @ pose
            transformed_pos = transformed_pose[:3, 3].copy()
            transformed_pos[2] += 0.7
            transformed_pos[0] += 0.55
            transformed[i] = transformed_pos
        return transformed


def create_body_visualizer(vis, sphere_radius=0.02, show_connections=True):
    """Create full body skeleton visualization objects"""
    joint_names = [
        "Pelvis", "LEFT_HIP", "RIGHT_HIP", "SPINE1", "LEFT_KNEE", "RIGHT_KNEE",
        "SPINE2", "LEFT_ANKLE", "RIGHT_ANKLE", "SPINE3", "LEFT_FOOT", "RIGHT_FOOT",
        "NECK", "LEFT_COLLAR", "RIGHT_COLLAR", "HEAD", "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HAND", "RIGHT_HAND"
    ]
    
    for joint_idx in range(24):
        sphere = g.Sphere(sphere_radius)
        if joint_idx == 0:
            color = 0x00ff00
        elif joint_idx == 15:
            color = 0xff0000
        elif 1 <= joint_idx <= 11:
            color = 0x0066ff + (joint_idx * 0x000011)
        elif 12 <= joint_idx <= 15:
            color = 0xffaa00 + ((joint_idx - 12) * 0x000011)
        else:
            color = 0xff6600 + ((joint_idx - 16) * 0x000011)
        
        try:
            vis[f"joint_{joint_idx}"].set_object(sphere, g.MeshLambertMaterial(color=color))
            initial_transform = np.eye(4, dtype=np.float64)
            vis[f"joint_{joint_idx}"].set_transform(initial_transform)
        except Exception as e:
            print(f"⚠ Failed to create joint {joint_idx} ({joint_names[joint_idx]}): {e}")
    
    connections = []
    if show_connections:
        all_connections = get_body_connections()
        line_color = 0xffffff
        line_idx = 0
        for parent_joint, child_joint in all_connections:
            if parent_joint < 24 and child_joint < 24:
                initial_points = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32).T
                line = g.Line(g.PointsGeometry(initial_points), g.LineBasicMaterial(color=line_color, linewidth=3))
                vis[f"line_{line_idx}"].set_object(line)
                connections.append((parent_joint, child_joint, line_idx))
                line_idx += 1
        if len(connections) > 0:
            print(f"✓ Created {len(connections)} skeleton connections")
    return connections


def update_body_visualization(vis, body_pose, connections):
    """Update full body skeleton visualization"""
    if body_pose is None:
        return
    try:
        if not isinstance(body_pose, np.ndarray):
            body_pose = np.array(body_pose)
        if len(body_pose) != 24:
            return
        positions = np.array([joint[:3] for joint in body_pose])
        transformed_positions = transform_coordinates(positions)
        for joint_idx in range(len(transformed_positions)):
            position = transformed_positions[joint_idx]
            transform = np.eye(4)
            transform[:3, 3] = position
            vis[f"joint_{joint_idx}"].set_transform(transform)
        for parent_joint, child_joint, line_idx in connections:
            if parent_joint < len(transformed_positions) and child_joint < len(transformed_positions):
                start_pos = transformed_positions[parent_joint]
                end_pos = transformed_positions[child_joint]
                line_points = np.array([start_pos, end_pos], dtype=np.float32).T
                line = g.Line(g.PointsGeometry(line_points), g.LineBasicMaterial(color=0xffffff, linewidth=3))
                vis[f"line_{line_idx}"].set_object(line)
    except Exception as e:
        pass


def create_hand_visualizer(vis, hand_prefix="", sphere_radius=0.01, show_connections=True):
    """Create hand visualization objects"""
    if hand_prefix == "left_hand_":
        base_colors = [0x0066ff, 0x0088ff, 0x00aaff, 0x00ccff, 0x00eeff, 0x00ffff]
        line_color = 0x66ccff
    elif hand_prefix == "right_hand_":
        base_colors = [0xff6600, 0xff8800, 0xffaa00, 0xffcc00, 0xffee00, 0xffff00]
        line_color = 0xff9966
    else:
        base_colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff]
        line_color = 0xffffff
    
    vis_index = list(range(26))
    for idx, keypoint_idx in enumerate(vis_index):
        sphere = g.Sphere(sphere_radius)
        color = base_colors[idx % len(base_colors)]
        try:
            vis[f"{hand_prefix}keypoint_{keypoint_idx}"].set_object(sphere, g.MeshLambertMaterial(color=color))
            initial_transform = np.eye(4, dtype=np.float64)
            vis[f"{hand_prefix}keypoint_{keypoint_idx}"].set_transform(initial_transform)
        except Exception as e:
            print(f"⚠ Failed to create {hand_prefix} joint {keypoint_idx}: {e}")
    
    connections = []
    if show_connections:
        all_connections = get_finger_connections()
        vis_index_set = set(vis_index)
        line_idx = 0
        for start_joint, end_joint in all_connections:
            if start_joint in vis_index_set and end_joint in vis_index_set:
                initial_points = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32).T
                line = g.Line(g.PointsGeometry(initial_points), g.LineBasicMaterial(color=line_color, linewidth=2))
                vis[f"{hand_prefix}line_{line_idx}"].set_object(line)
                connections.append((start_joint, end_joint, line_idx))
                line_idx += 1
        if len(connections) > 0:
            hand_name = "left hand" if hand_prefix == "left_hand_" else "right hand" if hand_prefix == "right_hand_" else "hand"
            print(f"✓ {hand_name} created {len(connections)} connections")
    return connections


def update_hand_visualization(vis, hand_state, connections, hand_prefix=""):
    """Update hand visualization"""
    if hand_state is None:
        return
    try:
        if not isinstance(hand_state, np.ndarray):
            hand_state = np.array(hand_state)
        if len(hand_state.shape) != 2 or hand_state.shape[0] != 26:
            return
        positions = hand_state[:, :3].copy()
        is_right_hand = (hand_prefix == "right_hand_")
        transformed_positions = transform_coordinates_hand(positions, is_right_hand=is_right_hand)
        for keypoint_idx in range(len(transformed_positions)):
            position = transformed_positions[keypoint_idx]
            transform = np.eye(4)
            transform[:3, 3] = position
            vis[f"{hand_prefix}keypoint_{keypoint_idx}"].set_transform(transform)
        for start_joint, end_joint, line_idx in connections:
            if start_joint < len(transformed_positions) and end_joint < len(transformed_positions):
                start_pos = transformed_positions[start_joint]
                end_pos = transformed_positions[end_joint]
                line_points = np.array([start_pos, end_pos], dtype=np.float32).T
                line_color = 0x66ccff if hand_prefix == "left_hand_" else 0xff9966 if hand_prefix == "right_hand_" else 0xffffff
                line = g.Line(g.PointsGeometry(line_points), g.LineBasicMaterial(color=line_color, linewidth=2))
                vis[f"{hand_prefix}line_{line_idx}"].set_object(line)
    except Exception as e:
        pass


def create_controller_visualizer(vis, sphere_radius=0.03):
    """Create controller visualization objects"""
    left_sphere = g.Sphere(sphere_radius)
    vis["controller_left"].set_object(left_sphere, g.MeshLambertMaterial(color=0x0066ff))
    initial_transform = np.eye(4, dtype=np.float64)
    vis["controller_left"].set_transform(initial_transform)
    
    right_sphere = g.Sphere(sphere_radius)
    vis["controller_right"].set_object(right_sphere, g.MeshLambertMaterial(color=0xff6600))
    vis["controller_right"].set_transform(initial_transform)
    print("✓ Controller visualization objects created (left controller: blue, right controller: orange)")


def update_controller_visualization(vis, left_controller_pose, right_controller_pose):
    """Update controller visualization"""
    try:
        if left_controller_pose is not None:
            left_pos = np.array(left_controller_pose[:3])
            if np.linalg.norm(left_pos) > 1e-6:
                transformed_left_pos = transform_coordinates(left_pos)
                left_transform = np.eye(4)
                left_transform[:3, 3] = transformed_left_pos
                vis["controller_left"].set_transform(left_transform)
        if right_controller_pose is not None:
            right_pos = np.array(right_controller_pose[:3])
            if np.linalg.norm(right_pos) > 1e-6:
                transformed_right_pos = transform_coordinates(right_pos)
                right_transform = np.eye(4)
                right_transform[:3, 3] = transformed_right_pos
                vis["controller_right"].set_transform(right_transform)
    except Exception as e:
        pass


def check_keyboard_input():
    """Non-blocking keyboard input detection (space key)"""
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
            if char == ' ':
                return True
    except:
        pass
    return False


# ========== ZED Mini Camera Class ==========
class ZEDMiniRecorder:
    """ZED Mini Camera Recorder"""
    
    def __init__(self, record_depth=True, visualize=False):
        self.record_depth = record_depth
        self.visualize = visualize
        self.zed = None
        self.is_recording = False
        self.running = False
        self.svo_path = None
        
        # Recording statistics
        self.recording_start_time = None
        self.recording_stop_time = None
        self.frame_count = 0
        
        # Mat objects for visualization
        self.image_left = sl.Mat()
        
        # Frame rate statistics
        self.grab_count = 0
        self.grab_start_time = None
        
        # Thread lock
        self.lock = threading.Lock()
        
        print(f"[ZED] Depth recording: {'enabled' if self.record_depth else 'disabled'}")
        print(f"[ZED] Real-time visualization: {'enabled' if self.visualize else 'disabled'}")
    
    def init_camera(self):
        """Initialize ZED Mini camera"""
        self.zed = sl.Camera()
        
        # Set initialization parameters (ZED Mini)
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Resolution supported by ZED Mini
        init_params.camera_fps = 60  # ZED Mini max 60fps
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL if self.record_depth else sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER
        
        # Open camera
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Cannot open ZED Mini camera: {status}")

        print("[ZED] ✓ ZED Mini camera opened successfully")

        # Get camera information
        camera_info = self.zed.get_camera_information()
        print(f"[ZED] Camera model: {camera_info.camera_model}")
        print(f"[ZED] Serial number: {camera_info.serial_number}")

        resolution = camera_info.camera_configuration.resolution
        print(f"[ZED] Resolution: {resolution.width}x{resolution.height}")
        self.resolution = (resolution.width, resolution.height)
    
    def start_recording(self, svo_path):
        """Start recording SVO"""
        with self.lock:
            if self.is_recording:
                print("[ZED] [WARN] Recording already in progress, stopping current recording")
                self._stop_recording_internal()

            self.svo_path = svo_path

            # Create directory
            os.makedirs(os.path.dirname(svo_path), exist_ok=True)

            if os.path.exists(self.svo_path):
                print(f"[ZED] [WARN] SVO file already exists: {self.svo_path}")
                print(f"[ZED] [WARN] Will overwrite existing file")

            # Set recording parameters
            recording_params = sl.RecordingParameters()
            recording_params.video_filename = self.svo_path
            recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264

            err = self.zed.enable_recording(recording_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Cannot enable recording: {err}")

            self.is_recording = True
            self.recording_start_time = time.time()
            self.frame_count = 0

            print(f"[ZED] ✓ Started recording SVO, saving to: {self.svo_path}")
    
    def _stop_recording_internal(self):
        """Stop recording SVO (internal method, caller must hold lock)"""
        if not self.is_recording:
            return
        
        self.zed.disable_recording()
        self.is_recording = False
        self.recording_stop_time = time.time()
        
        duration_sec = self.recording_stop_time - self.recording_start_time if self.recording_start_time else 0
        fps = self.frame_count / duration_sec if duration_sec > 0 else 0
        
        print(f"[ZED] ✓ Recording stopped")
        print(f"[ZED]   Duration: {duration_sec:.3f} seconds")
        print(f"[ZED]   Total frames: {self.frame_count}")
        print(f"[ZED]   Average frame rate: {fps:.2f} Hz (FPS)")
        print(f"[ZED]   SVO file: {self.svo_path}")

        if os.path.exists(self.svo_path):
            file_size_mb = os.path.getsize(self.svo_path) / (1024 * 1024)
            print(f"[ZED]   File size: {file_size_mb:.2f} MB")
    
    def stop_recording(self):
        """Stop recording SVO (thread-safe)"""
        with self.lock:
            self._stop_recording_internal()
    
    def camera_loop(self):
        """Camera loop: continuously grab frames"""
        print("[ZED] Camera loop started")
        if self.visualize:
            print("[ZED] Visualization window opened, press 'q' key to exit program")
        
        self.grab_start_time = time.time()
        self.grab_count = 0
        
        try:
            while self.running:
                status = self.zed.grab()
                if status == sl.ERROR_CODE.SUCCESS:
                    self.grab_count += 1
                    
                    with self.lock:
                        if self.is_recording:
                            self.frame_count += 1
                            if self.frame_count % 100 == 0:
                                elapsed = time.time() - self.recording_start_time
                                fps = self.frame_count / elapsed if elapsed > 0 else 0
                                print(f"[ZED] Recorded {self.frame_count} frames, current frame rate: {fps:.2f} Hz")

                    # Visualization display
                    if self.visualize:
                        self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
                        display_frame = self.image_left.get_data()[:, :, :3].copy()
                        
                        grab_elapsed = time.time() - self.grab_start_time
                        grab_fps = self.grab_count / grab_elapsed if grab_elapsed > 0 else 0
                        
                        with self.lock:
                            is_rec = self.is_recording
                            rec_frame = self.frame_count
                            rec_start = self.recording_start_time
                        
                        if is_rec:
                            rec_elapsed = time.time() - rec_start if rec_start else 0
                            rec_fps = rec_frame / rec_elapsed if rec_elapsed > 0 else 0
                            info_text = f"[REC] Frame: {rec_frame} | FPS: {rec_fps:.1f} | Time: {rec_elapsed:.1f}s"
                            text_color = (0, 0, 255)
                        else:
                            info_text = f"[IDLE] FPS: {grab_fps:.1f}"
                            text_color = (0, 255, 0)
                        
                        cv2.putText(display_frame, info_text, (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                        cv2.imshow("ZED Mini Camera", display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("[ZED] User pressed 'q' key, exiting...")
                            self.running = False
                            break
                            
                elif status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    print("[ZED] [WARN] Reached SVO file size limit")
                    with self.lock:
                        if self.is_recording:
                            self.stop_recording()
                else:
                    time.sleep(0.001)
                    
        except Exception as e:
            print(f"[ZED] [ERROR] Camera loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.visualize:
                cv2.destroyAllWindows()
            print("[ZED] Camera loop ended")
    
    def close(self):
        """Close camera"""
        self.running = False
        self.stop_recording()
        if self.zed:
            self.zed.close()
            print("[ZED] Camera closed")


def data_acquisition_loop(stop_event, data_lock, shared_data, interval=0.05):
    """Independent data acquisition loop thread"""
    print("[PICO] Data acquisition thread started")
    while not stop_event.is_set():
        try:
            body_pose = xrt.get_body_joints_pose()
            left_controller_pose = xrt.get_left_controller_pose()
            right_controller_pose = xrt.get_right_controller_pose()
            left_hand_state = xrt.get_left_hand_tracking_state()
            right_hand_state = xrt.get_right_hand_tracking_state()
            
            with data_lock:
                shared_data['body_pose'] = body_pose
                shared_data['left_controller_pose'] = left_controller_pose
                shared_data['right_controller_pose'] = right_controller_pose
                shared_data['left_hand_state'] = left_hand_state
                shared_data['right_hand_state'] = right_hand_state
            
            time.sleep(interval)
        except Exception as e:
            if not stop_event.is_set():
                print(f"[PICO] [WARN] Data acquisition error: {e}")
            time.sleep(interval)
    print("[PICO] Data acquisition thread stopped")


def visualization_loop(vis, connections, left_hand_connections, right_hand_connections,
                       stop_event, data_lock, shared_data):
    """Independent visualization loop thread"""
    print("[VIS] Visualization thread started")
    while not stop_event.is_set():
        try:
            with data_lock:
                body_pose = shared_data.get('body_pose', None)
                left_controller_pose = shared_data.get('left_controller_pose', None)
                right_controller_pose = shared_data.get('right_controller_pose', None)
                left_hand_state = shared_data.get('left_hand_state', None)
                right_hand_state = shared_data.get('right_hand_state', None)
            
            if body_pose is not None:
                try:
                    body_array = np.array(body_pose)
                    if len(body_array) == 24:
                        update_body_visualization(vis, body_array, connections)
                except:
                    pass
            
            update_controller_visualization(vis, left_controller_pose, right_controller_pose)
            
            if left_hand_state is not None:
                try:
                    left_hand_array = np.array(left_hand_state)
                    if left_hand_array.shape[0] == 26:
                        update_hand_visualization(vis, left_hand_array, left_hand_connections, hand_prefix="left_hand_")
                except:
                    pass
            
            if right_hand_state is not None:
                try:
                    right_hand_array = np.array(right_hand_state)
                    if right_hand_array.shape[0] == 26:
                        update_hand_visualization(vis, right_hand_array, right_hand_connections, hand_prefix="right_hand_")
                except:
                    pass
            
            time.sleep(0.05)
        except Exception as e:
            if not stop_event.is_set():
                print(f"[VIS] [WARN] Visualization update error: {e}")
            time.sleep(0.05)
    print("[VIS] Visualization thread stopped")


def save_data_to_hdf5(all_body_data, all_local_timestamps,
                      all_left_controller_data, all_right_controller_data,
                      all_left_hand_data, all_right_hand_data,
                      all_left_hand_active, all_right_hand_active,
                      args, episode_index, svo_path):
    """Save data to HDF5 file"""
    if len(all_body_data) == 0:
        print("[HDF5] [WARN] No data to save")
        return None
    
    try:
        body_data = np.array(all_body_data)
        local_timestamps = np.array(all_local_timestamps)
        left_controller_data = np.array(all_left_controller_data)
        right_controller_data = np.array(all_right_controller_data)
        left_hand_data = np.array(all_left_hand_data)
        right_hand_data = np.array(all_right_hand_data)
        left_hand_active = np.array(all_left_hand_active)
        right_hand_active = np.array(all_right_hand_active)
        
        dataset_name = args.dataset_name if args.dataset_name else args.name
        dataset_dir = os.path.join(args.data_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        hdf5_path = os.path.join(dataset_dir, f"episode_{episode_index}.hdf5")
        
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('body_pose', data=body_data, compression='gzip', compression_opts=4)
            f.create_dataset('local_timestamps_ns', data=local_timestamps, compression='gzip', compression_opts=4)
            f.create_dataset('left_controller_pose', data=left_controller_data, compression='gzip', compression_opts=4)
            f.create_dataset('right_controller_pose', data=right_controller_data, compression='gzip', compression_opts=4)
            f.create_dataset('left_hand_pose', data=left_hand_data, compression='gzip', compression_opts=4)
            f.create_dataset('right_hand_pose', data=right_hand_data, compression='gzip', compression_opts=4)
            f.create_dataset('left_hand_active', data=left_hand_active, compression='gzip', compression_opts=4)
            f.create_dataset('right_hand_active', data=right_hand_active, compression='gzip', compression_opts=4)
            
            # Metadata
            f.attrs['dataset_name'] = args.name
            f.attrs['creation_time'] = datetime.now().isoformat()
            f.attrs['total_frames'] = len(all_body_data)
            f.attrs['collection_interval_s'] = args.interval
            f.attrs['episode_index'] = episode_index
            f.attrs['data_dir'] = args.data_dir
            f.attrs['zed_svo_path'] = os.path.basename(svo_path) if svo_path else ""
            f.attrs['zed_synced'] = True
            
            f.attrs['body_pose_format'] = '(frames, 24, 7) - 24 joints, each with [x, y, z, qx, qy, qz, qw]'
            f.attrs['controller_pose_format'] = '(frames, 7) - [x, y, z, qx, qy, qz, qw]'
            f.attrs['hand_pose_format'] = '(frames, 26, 7) - 26 joints, each with [x, y, z, qx, qy, qz, qw]'
            f.attrs['local_timestamp_format'] = 'nanoseconds (from local PC)'
            f.attrs['hand_active_format'] = '0=inactive, 1=active'
            
            body_joint_names = [
                "Pelvis", "LEFT_HIP", "RIGHT_HIP", "SPINE1", "LEFT_KNEE", "RIGHT_KNEE",
                "SPINE2", "LEFT_ANKLE", "RIGHT_ANKLE", "SPINE3", "LEFT_FOOT", "RIGHT_FOOT",
                "NECK", "LEFT_COLLAR", "RIGHT_COLLAR", "HEAD", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HAND", "RIGHT_HAND"
            ]
            f.attrs['body_joint_names'] = np.array([name.encode('utf-8') for name in body_joint_names])
            
            hand_joint_names = [
                "Palm", "Wrist",
                "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_Tip",
                "Index_MCP", "Index_PIP", "Index_Middle", "Index_DIP", "Index_Tip",
                "Middle_MCP", "Middle_PIP", "Middle_Middle", "Middle_DIP", "Middle_Tip",
                "Ring_MCP", "Ring_PIP", "Ring_Middle", "Ring_DIP", "Ring_Tip",
                "Little_MCP", "Little_PIP", "Little_Middle", "Little_DIP", "Little_Tip"
            ]
            f.attrs['hand_joint_names'] = np.array([name.encode('utf-8') for name in hand_joint_names])
        
        print(f"[HDF5] ✓ Data saved to: {hdf5_path}")
        print(f"[HDF5]   File size: {os.path.getsize(hdf5_path) / (1024*1024):.2f} MB")
        print(f"[HDF5]   Total frames: {len(all_body_data)}")
        return hdf5_path
    except Exception as e:
        print(f"[HDF5] ✗ Failed to save data: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='PICO + ZED Mini Data Collection (PC Integrated Version)')
    parser.add_argument('--name', default='body_data', type=str,
                       help='Dataset name (default: body_data)')
    parser.add_argument('--dataset-name', type=str, default=None,
                       help='Dataset directory name (default: uses --name value)')
    parser.add_argument('--data-dir', default='data_collection', type=str,
                       help='Data save directory (default: data_collection)')
    parser.add_argument('--interval', default=0.01, type=float,
                       help='Update interval (seconds, default: 0.01, i.e., 100fps)')
    parser.add_argument('--episode', type=int, default=0,
                       help='Starting episode index (default: 0)')
    parser.add_argument('--visualize-zed', '-v', action='store_true',
                       help='Enable ZED camera real-time visualization (default: disabled)')
    parser.add_argument('--no-depth', action='store_true',
                       help='Do not record depth information (default: record depth)')
    args = parser.parse_args()
    
    # ========== Initialize PICO SDK ==========
    print("\n" + "="*60)
    print("PICO + ZED Mini Data Collection System Starting")
    print("="*60)

    print("\n[PICO] Initializing PICO SDK...")
    try:
        xrt.init()
        print("[PICO] ✓ SDK initialization successful!")
    except Exception as e:
        print(f"[PICO] ✗ SDK initialization failed: {e}")
        return

    body_available = xrt.is_body_data_available()
    if not body_available:
        print("[PICO] ⚠ Warning: No body data currently available")
        print("[PICO]    Please ensure PICO device is connected and full-body tracking is activated")

    # ========== Initialize ZED Mini Camera ==========
    print("\n[ZED] Initializing ZED Mini camera...")
    zed_recorder = ZEDMiniRecorder(
        record_depth=not args.no_depth,
        visualize=args.visualize_zed
    )
    try:
        zed_recorder.init_camera()
    except Exception as e:
        print(f"[ZED] ✗ ZED Mini initialization failed: {e}")
        xrt.close()
        return

    # ========== Initialize MeshCat Visualizer ==========
    print("\n[VIS] Initializing MeshCat visualizer...")
    try:
        vis = meshcat.Visualizer()
        vis.open()
        vis.delete()
        print("[VIS] ✓ MeshCat visualizer initialization successful!")
        print("[VIS]   Open browser at http://localhost:7000/static/ to view visualization")
    except Exception as e:
        print(f"[VIS] ✗ MeshCat initialization failed: {e}")
        zed_recorder.close()
        xrt.close()
        return
    
    # Create visualization objects
    print("\n[VIS] Creating visualization objects...")
    connections = create_body_visualizer(vis, sphere_radius=0.02, show_connections=True)
    print("[VIS] ✓ Full body skeleton visualization objects created")
    create_controller_visualizer(vis, sphere_radius=0.03)
    left_hand_connections = create_hand_visualizer(vis, hand_prefix="left_hand_", sphere_radius=0.01, show_connections=True)
    print("[VIS] ✓ Left hand visualization objects created")
    right_hand_connections = create_hand_visualizer(vis, hand_prefix="right_hand_", sphere_radius=0.01, show_connections=True)
    print("[VIS] ✓ Right hand visualization objects created")

    # Add test sphere
    test_sphere = g.Sphere(0.03)
    vis["test_origin"].set_object(test_sphere, g.MeshLambertMaterial(color=0xff0000))
    test_transform = np.eye(4)
    vis["test_origin"].set_transform(test_transform)
    print("[VIS] ✓ Red test sphere added at origin\n")

    # Set stdin to non-blocking mode
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    # Create shared data and lock
    data_lock = threading.Lock()
    shared_data = {
        'body_pose': None,
        'left_controller_pose': None,
        'right_controller_pose': None,
        'left_hand_state': None,
        'right_hand_state': None
    }
    stop_threads = threading.Event()
    
    # Start data acquisition thread
    print("[PICO] Starting data acquisition thread...")
    data_thread = threading.Thread(
        target=data_acquisition_loop,
        args=(stop_threads, data_lock, shared_data, args.interval),
        daemon=True
    )
    data_thread.start()
    print("[PICO] ✓ Data acquisition thread started\n")

    # Start visualization thread
    print("[VIS] Starting visualization thread...")
    vis_thread = threading.Thread(
        target=visualization_loop,
        args=(vis, connections, left_hand_connections, right_hand_connections,
              stop_threads, data_lock, shared_data),
        daemon=True
    )
    vis_thread.start()
    print("[VIS] ✓ Visualization thread started\n")

    # Start ZED camera loop thread
    print("[ZED] Starting ZED camera loop thread...")
    zed_recorder.running = True
    zed_thread = threading.Thread(target=zed_recorder.camera_loop, daemon=True)
    zed_thread.start()
    print("[ZED] ✓ ZED camera loop thread started\n")

    print("="*60)
    print("System ready!")
    print("="*60)
    print("Press SPACE key to complete current episode collection, press Ctrl+C to exit program completely\n")

    # Main loop
    try:
        while True:
            # Data storage
            all_body_data = []
            all_local_timestamps = []
            all_left_controller_data = []
            all_right_controller_data = []
            all_left_hand_data = []
            all_right_hand_data = []
            all_left_hand_active = []
            all_right_hand_active = []
            
            time.sleep(0.5)
            
            # User input episode index
            episode_index = None
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                print("\n" + "="*50)
                print("Preparing to collect new episode")
                print("="*50)
                user_input = input("Enter episode index: ")
                if not user_input.strip():
                    print("[WARN] Empty input, skipping this collection")
                    tty.setcbreak(sys.stdin.fileno())
                    continue
                try:
                    episode_index = int(user_input.strip())
                except ValueError:
                    print(f"[WARN] Invalid input, skipping this collection")
                    tty.setcbreak(sys.stdin.fileno())
                    continue

                tty.setcbreak(sys.stdin.fileno())

            except Exception as e:
                print(f"[ERROR] Failed to process episode index input: {e}")
                tty.setcbreak(sys.stdin.fileno())
                continue

            # Set SVO path
            dataset_name = args.dataset_name if args.dataset_name else args.name
            dataset_dir = os.path.join(args.data_dir, dataset_name)
            svo_path = os.path.join(dataset_dir, f"episode_{episode_index}.svo2")
            
            # Start ZED recording
            try:
                zed_recorder.start_recording(svo_path)
            except Exception as e:
                print(f"[ZED] [ERROR] Cannot start recording: {e}")
                continue

            print(f"\n[INFO] ✓ Started collecting Episode {episode_index}")
            print("[INFO] Press SPACE key to complete current episode collection\n")
            
            frame_count = 0
            episode_finished = False
            episode_start_time = time.time()
            
            # Collection loop
            while True:
                if check_keyboard_input():
                    print("\n[INFO] SPACE key detected, completing current episode collection...")
                    episode_finished = True
                    break

                # Check if ZED recorder is still running
                if not zed_recorder.running:
                    print("\n[INFO] ZED camera stopped, ending collection...")
                    episode_finished = True
                    break
                
                with data_lock:
                    body_pose = shared_data.get('body_pose', None)
                    left_controller_pose = shared_data.get('left_controller_pose', None)
                    right_controller_pose = shared_data.get('right_controller_pose', None)
                    left_hand_state = shared_data.get('left_hand_state', None)
                    right_hand_state = shared_data.get('right_hand_state', None)
                
                body_available = xrt.is_body_data_available()
                left_hand_active = xrt.get_left_hand_is_active()
                right_hand_active = xrt.get_right_hand_is_active()
                
                local_timestamp = time.time_ns()
                
                all_body_data.append(body_pose)
                all_local_timestamps.append(local_timestamp)
                all_left_controller_data.append(left_controller_pose)
                all_right_controller_data.append(right_controller_pose)
                all_left_hand_data.append(left_hand_state)
                all_right_hand_data.append(right_hand_state)
                all_left_hand_active.append(left_hand_active)
                all_right_hand_active.append(right_hand_active)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - episode_start_time
                    current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    print(f"[PICO] Collected {frame_count} frames, current frequency: {current_fps:.2f} Hz")

                time.sleep(args.interval)

            # Post-episode processing
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            average_fps = frame_count / episode_duration if episode_duration > 0 else 0

            print(f"\n[INFO] Episode {episode_index} collection statistics:")
            print(f"  PICO total frames: {frame_count}")
            print(f"  Collection duration: {episode_duration:.2f} seconds")
            print(f"  PICO average frequency: {average_fps:.2f} Hz")

            # Stop ZED recording
            zed_recorder.stop_recording()

            # Save PICO data to HDF5
            if len(all_body_data) > 0:
                print(f"\n[HDF5] Saving Episode {episode_index} PICO data...")
                save_data_to_hdf5(
                    all_body_data, all_local_timestamps,
                    all_left_controller_data, all_right_controller_data,
                    all_left_hand_data, all_right_hand_data,
                    all_left_hand_active, all_right_hand_active,
                    args, episode_index=episode_index, svo_path=svo_path
                )
                print(f"[INFO] ✓ Episode {episode_index} collection complete!\n")
            else:
                print(f"[WARN] Episode {episode_index} has no PICO data to save\n")
            
    except KeyboardInterrupt:
        print("\n\n[INFO] User interrupted (Ctrl+C)")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Stop ZED recording
        zed_recorder.stop_recording()

        # Save current episode data
        if 'all_body_data' in locals() and len(all_body_data) > 0 and episode_index is not None:
            print(f"\n[HDF5] Saving data...")
            save_data_to_hdf5(
                all_body_data, all_local_timestamps,
                all_left_controller_data, all_right_controller_data,
                all_left_hand_data, all_right_hand_data,
                all_left_hand_active, all_right_hand_active,
                args, episode_index=episode_index, svo_path=svo_path if 'svo_path' in locals() else ""
            )
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    finally:
        print("\n[INFO] Stopping all threads...")
        stop_threads.set()
        zed_recorder.running = False

        data_thread.join(timeout=2.0)
        print("[INFO] ✓ PICO data acquisition thread stopped")

        vis_thread.join(timeout=2.0)
        print("[INFO] ✓ Visualization thread stopped")

        zed_thread.join(timeout=2.0)
        print("[INFO] ✓ ZED camera thread stopped")
        
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except:
            pass
        
        zed_recorder.close()

        print("\n[PICO] Closing SDK...")
        xrt.close()
        print("[INFO] ✓ Program ended")


if __name__ == '__main__':
    main()

