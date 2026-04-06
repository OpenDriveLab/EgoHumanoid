#!/usr/bin/env python3
"""
Camera and Robot Data Merge Script

Function: Merge camera image data into robot HDF5 files, ensuring data synchronization through timestamp alignment

Configuration:
- Image storage format: JPEG compression
- Output location: merged_data/
- Timestamp difference handling: Use nearest image, mark in report if exceeds threshold
- Data format: Read camera data from svo2 files
- Delta EEF: Automatically calculate and add delta EEF data (incremental transformation)
- Delta Height: Automatically calculate and add delta height data (height increment from teleop_base_height_command)

Directory structure:
- Recursively find all episode_*.hdf5 files and corresponding svo2 files under data_dir
- Output to output_dir, uniformly renumbered as episode_0.hdf5, episode_1.hdf5, ...

Performance optimizations:
- Use binary search to optimize timestamp matching (O(n*log(m)) instead of O(n*m))
- Support multithreading for parallel processing of multiple episodes
- Use OpenCV for JPEG encoding (faster than PIL if available)
- Batch processing of timestamp matching and warning detection
"""

import os
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import io
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
import pyzed.sl as sl
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation as R


def _process_single_episode_wrapper(merger, hdf5_file, svo2_file, output_file, episode_name):
    """Wrapper function for processing a single episode (for parallel processing)"""
    return merger._merge_episode(hdf5_file, svo2_file, output_file, episode_name)


def pose_to_transform_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Convert position and quaternion to 4x4 homogeneous transformation matrix

    Args:
        position: (3,) xyz position
        quaternion: (4,) quaternion (qx, qy, qz, qw) - scipy format

    Returns:
        (4, 4) homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quaternion).as_matrix()
    T[:3, 3] = position
    return T


def transform_matrix_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    """
    Extract xyz translation and rpy rotation from transformation matrix

    Args:
        T: (4, 4) transformation matrix

    Returns:
        (6,) [x, y, z, roll, pitch, yaw] - rotation in radians
    """
    # Extract translation
    xyz = T[:3, 3]

    # Extract rotation and convert to Euler angles (roll, pitch, yaw)
    rot_matrix = T[:3, :3]
    rpy = R.from_matrix(rot_matrix).as_euler('xyz')  # roll, pitch, yaw

    return np.concatenate([xyz, rpy])


def compute_delta_transform(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Compute incremental transformation matrix from T1 to T2

    delta_T = T1^{-1} @ T2
    i.e.: T2 = T1 @ delta_T

    Args:
        T1: (4, 4) initial transformation matrix
        T2: (4, 4) target transformation matrix

    Returns:
        (4, 4) incremental transformation matrix
    """
    T1_inv = np.linalg.inv(T1)
    return T1_inv @ T2


def compute_delta_eef_for_hand(eef_data: np.ndarray) -> np.ndarray:
    """
    Compute all delta EEF for a single hand

    Args:
        eef_data: (N, 7) each row is [x, y, z, qw, qx, qy, qz] - scalar_first format

    Returns:
        (N, 6) each row is [dx, dy, dz, roll, pitch, yaw]
        First frame delta is set to 0
    """
    n_frames = len(eef_data)
    delta_eef = np.zeros((n_frames, 6))

    for i in range(1, n_frames):
        # Extract position and quaternion (scalar_first format: w, x, y, z)
        pos_prev = eef_data[i-1, :3]
        quat_scalar_first_prev = eef_data[i-1, 3:7]  # (qw, qx, qy, qz)

        pos_curr = eef_data[i, :3]
        quat_scalar_first_curr = eef_data[i, 3:7]  # (qw, qx, qy, qz)

        # Convert to scipy expected format (x, y, z, w)
        quat_vector_first_prev = np.array([
            quat_scalar_first_prev[1],  # x
            quat_scalar_first_prev[2],  # y
            quat_scalar_first_prev[3],  # z
            quat_scalar_first_prev[0]   # w
        ])
        quat_vector_first_curr = np.array([
            quat_scalar_first_curr[1],  # x
            quat_scalar_first_curr[2],  # y
            quat_scalar_first_curr[3],  # z
            quat_scalar_first_curr[0]   # w
        ])

        # Transformation matrices for previous and current frames
        T_prev = pose_to_transform_matrix(pos_prev, quat_vector_first_prev)
        T_curr = pose_to_transform_matrix(pos_curr, quat_vector_first_curr)

        # Compute incremental transformation
        delta_T = compute_delta_transform(T_prev, T_curr)

        # Convert to xyzrpy
        delta_eef[i] = transform_matrix_to_xyzrpy(delta_T)

    return delta_eef


def compute_delta_eef(eef_data: np.ndarray) -> np.ndarray:
    """
    Compute delta EEF for both hands

    Args:
        eef_data: (N, 14) each row is [left hand: x,y,z,qw,qx,qy,qz, right hand: x,y,z,qw,qx,qy,qz] - scalar_first format

    Returns:
        (N, 12) each row is [left hand: dx,dy,dz,roll,pitch,yaw, right hand: dx,dy,dz,roll,pitch,yaw]
    """
    # Separate left and right hand data
    left_eef = eef_data[:, :7]
    right_eef = eef_data[:, 7:14]

    # Compute delta for each hand separately
    left_delta = compute_delta_eef_for_hand(left_eef)
    right_delta = compute_delta_eef_for_hand(right_eef)

    # Concatenate
    return np.concatenate([left_delta, right_delta], axis=1)


def compute_delta_height(height_data: np.ndarray) -> np.ndarray:
    """
    Compute delta height (height increment)

    Args:
        height_data: (N,) or (N, 1) height data

    Returns:
        (N,) delta height, first frame delta is set to 0
    """
    # Ensure it's a 1D array
    if height_data.ndim > 1:
        height_data = height_data.squeeze()

    n_frames = len(height_data)
    delta_height = np.zeros(n_frames, dtype=np.float32)

    # Use vectorized operation to compute differences between adjacent frames (faster than loop)
    if n_frames > 1:
        delta_height[1:] = height_data[1:] - height_data[:-1]

    return delta_height


class CameraRobotMerger:
    """Camera and Robot Data Merger"""

    # Timestamp difference thresholds (milliseconds)
    WARNING_THRESHOLD_MS = 20.0   # Warning threshold
    SEVERE_THRESHOLD_MS = 50.0    # Severe warning threshold

    def __init__(self, data_dir: str, output_dir: str, num_workers: int = 1, compute_delta_eef: bool = True):
        """
        Initialize the merger

        Args:
            data_dir: Data directory path, will recursively search for all episode_*.hdf5 files
            output_dir: Output directory path
            num_workers: Number of parallel threads
            compute_delta_eef: Whether to compute delta EEF data (default True)
        """
        self.data_path = Path(data_dir).resolve()
        self.output_path = Path(output_dir).resolve()
        self.num_workers = num_workers
        self.compute_delta_eef = compute_delta_eef

        # Merge report
        self.report = {
            "merge_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_dir": str(self.data_path),
            "output_dir": str(self.output_path),
            "episodes": {},
            "total_frames": 0,
            "total_warnings": 0,
            "total_severe_warnings": 0,
            "avg_time_diff_ms": 0.0,
            "max_time_diff_ms": 0.0
        }
    
    def merge_all(self):
        """Merge dataset: recursively find all episode files under data_dir and process them"""
        print("=" * 80)
        print("🔄 Start merging camera data with robot data")
        print("=" * 80)

        print(f"📁 Data directory: {self.data_path}")
        print(f"📁 Output directory: {self.output_path}")

        # Recursively find all episode_*.hdf5 files
        hdf5_files = sorted(
            self.data_path.rglob("episode_*.hdf5"),
            key=lambda x: (x.parent, int(x.stem.split('_')[1]) if '_' in x.stem else 0)
        )

        # Recursively find all episode_*.svo2 files
        svo2_files = sorted(
            self.data_path.rglob("episode_*.svo2"),
            key=lambda x: (x.parent, int(x.stem.split('_')[1]) if '_' in x.stem else 0)
        )

        print(f"📊 Found {len(hdf5_files)} HDF5 files")
        print(f"📊 Found {len(svo2_files)} SVO2 files")

        # Check if file counts match
        if len(hdf5_files) == 0:
            print(f"❌ Error: No episode_*.hdf5 files found")
            return

        if len(svo2_files) == 0:
            print(f"❌ Error: No episode_*.svo2 files found")
            return

        if len(hdf5_files) != len(svo2_files):
            print(f"\n❌ Error: HDF5 file count ({len(hdf5_files)}) does not match SVO2 file count ({len(svo2_files)})!")
            print(f"\n   HDF5 file list:")
            for f in hdf5_files[:10]:  # Show only first 10
                print(f"     - {f}")
            if len(hdf5_files) > 10:
                print(f"     ... and {len(hdf5_files) - 10} more files")
            print(f"\n   SVO2 file list:")
            for f in svo2_files[:10]:  # Show only first 10
                print(f"     - {f}")
            if len(svo2_files) > 10:
                print(f"     ... and {len(svo2_files) - 10} more files")
            print(f"\n   Please check data integrity and try again!")
            return

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Prepare processing parameters for all episodes and check pairing
        episode_tasks = []
        all_time_diffs = []
        missing_pairs = []

        output_index = 0  # Sequential numbering, only increments for valid episodes
        for hdf5_file in hdf5_files:
            episode_name = hdf5_file.stem  # episode_0

            # Find corresponding svo2 file (in same directory, or hdf5 -> svo2 directory mapping)
            svo2_file = self._find_svo2_file(hdf5_file)

            if svo2_file is None or not svo2_file.exists():
                missing_pairs.append(hdf5_file)
                continue

            # Output file with unified numbering (save directly under output_dir)
            output_filename = f"episode_{output_index}.hdf5"
            output_file = self.output_path / output_filename

            episode_tasks.append((hdf5_file, svo2_file, output_file, episode_name, output_index))
            output_index += 1  # Only increment for valid episodes

        # If there are unpaired files, report error and exit
        if missing_pairs:
            print(f"\n❌ Error: Cannot find corresponding SVO2 files for the following {len(missing_pairs)} HDF5 files:")
            for f in missing_pairs[:10]:
                print(f"     - {f}")
            if len(missing_pairs) > 10:
                print(f"     ... and {len(missing_pairs) - 10} more files")
            print(f"\n   Please check data integrity and try again!")
            return

        print(f"✅ All files paired successfully, total {len(episode_tasks)} episodes")

        # Process episodes in parallel or serial
        if self.num_workers > 1 and len(episode_tasks) > 1:
            # Parallel processing
            print(f"  🚀 Using {self.num_workers} threads for parallel processing")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for hdf5_file, svo2_file, output_file, episode_name, output_index in episode_tasks:
                    future = executor.submit(
                        _process_single_episode_wrapper,
                        self, hdf5_file, svo2_file, output_file, episode_name
                    )
                    futures.append((future, episode_name, output_index, str(hdf5_file)))

                # Collect results
                for future, episode_name, output_index, source_path in tqdm(futures, desc="  Processing episodes"):
                    try:
                        episode_report = future.result()
                        if episode_report:
                            output_name = f"episode_{output_index}"
                            episode_report["source_file"] = source_path
                            self.report["episodes"][output_name] = episode_report
                            self.report["total_frames"] += episode_report["frame_count"]
                            self.report["total_warnings"] += episode_report["warning_count"]
                            self.report["total_severe_warnings"] += episode_report["severe_warning_count"]
                            all_time_diffs.extend(episode_report["time_diffs_ms"])
                    except Exception as e:
                        print(f"    ❌ Error processing {episode_name}: {e}")
        else:
            # Serial processing
            for hdf5_file, svo2_file, output_file, episode_name, output_index in tqdm(episode_tasks, desc="  Processing episodes"):
                episode_report = self._merge_episode(hdf5_file, svo2_file, output_file, episode_name)

                if episode_report:
                    output_name = f"episode_{output_index}"
                    episode_report["source_file"] = str(hdf5_file)
                    self.report["episodes"][output_name] = episode_report
                    self.report["total_frames"] += episode_report["frame_count"]
                    self.report["total_warnings"] += episode_report["warning_count"]
                    self.report["total_severe_warnings"] += episode_report["severe_warning_count"]
                    all_time_diffs.extend(episode_report["time_diffs_ms"])

        # Compute statistics
        if all_time_diffs:
            self.report["avg_time_diff_ms"] = float(np.mean(all_time_diffs))
            self.report["max_time_diff_ms"] = float(np.max(all_time_diffs))

        # Simplify report (don't save per-frame time differences)
        for ep_name in self.report["episodes"]:
            if "time_diffs_ms" in self.report["episodes"][ep_name]:
                del self.report["episodes"][ep_name]["time_diffs_ms"]

        # Save report
        self._save_report()

        print("\n" + "=" * 80)
        print("✅ Merge completed!")
        print("=" * 80)
    
    def _find_svo2_file(self, hdf5_file: Path) -> Optional[Path]:
        """
        Find corresponding svo2 file based on hdf5 file path

        Supported directory structures:
        1. Same directory: .../episode_0.hdf5 -> .../episode_0.svo2
        2. hdf5/svo2 directory structure: .../hdf5/episode_0.hdf5 -> .../svo2/episode_0.svo2
        3. Nested directories: .../hdf5/subdir/episode_0.hdf5 -> .../svo2/subdir/episode_0.svo2
        """
        episode_name = hdf5_file.stem

        # 1. First try same directory
        svo2_same_dir = hdf5_file.parent / f"{episode_name}.svo2"
        if svo2_same_dir.exists():
            return svo2_same_dir

        # 2. Try hdf5 -> svo2 directory mapping
        # Check if path contains 'hdf5'
        parts = hdf5_file.parts
        if 'hdf5' in parts:
            hdf5_idx = parts.index('hdf5')
            # Build svo2 path
            svo2_parts = list(parts[:hdf5_idx]) + ['svo2'] + list(parts[hdf5_idx + 1:-1]) + [f"{episode_name}.svo2"]
            svo2_file = Path(*svo2_parts)
            if svo2_file.exists():
                return svo2_file

        return None
    
    
    def _merge_episode(self, robot_hdf5_path: Path,
                       svo2_path: Path,
                       output_hdf5_path: Path,
                       episode_name: str) -> Optional[Dict]:
        """Merge a single episode"""

        # Read svo2 file, get all images and timestamps
        camera_data = self._read_svo2_file(svo2_path)

        if camera_data is None or len(camera_data["timestamps_ns"]) == 0:
            return None

        # Open robot HDF5 file
        with h5py.File(robot_hdf5_path, 'r') as robot_f:
            if "timestamp" not in robot_f:
                return None

            robot_timestamps = robot_f["timestamp"][:]
            num_frames = len(robot_timestamps)

            # Create output file
            with h5py.File(output_hdf5_path, 'w') as output_f:
                # Copy original data
                for key in robot_f.keys():
                    data = robot_f[key][:]
                    output_f.create_dataset(key, data=data, compression="gzip")

                # Compute and add delta EEF data (if enabled)
                if self.compute_delta_eef:
                    if 'observation_eef_state' in robot_f and 'action_eef' in robot_f:
                        obs_eef = robot_f['observation_eef_state'][:]
                        action_eef = robot_f['action_eef'][:]

                        # Compute delta EEF
                        obs_delta = compute_delta_eef(obs_eef)
                        action_delta = compute_delta_eef(action_eef)

                        # Write delta EEF data
                        output_f.create_dataset('obs_delta_eef', data=obs_delta, compression='gzip')
                        output_f.create_dataset('action_delta_eef', data=action_delta, compression='gzip')

                        # Add metadata description
                        output_f['obs_delta_eef'].attrs['description'] = 'Delta EEF from observation: [left: dx,dy,dz,roll,pitch,yaw, right: dx,dy,dz,roll,pitch,yaw]'
                        output_f['obs_delta_eef'].attrs['units'] = 'meters for xyz, radians for rpy'
                        output_f['action_delta_eef'].attrs['description'] = 'Delta EEF from action: [left: dx,dy,dz,roll,pitch,yaw, right: dx,dy,dz,roll,pitch,yaw]'
                        output_f['action_delta_eef'].attrs['units'] = 'meters for xyz, radians for rpy'

                # Compute and add delta height data (if height data exists)
                height_cmd = robot_f['teleop_base_height_command'][:]
                delta_height = compute_delta_height(height_cmd)
                output_f.create_dataset('delta_height', data=delta_height, compression='gzip')
                output_f['delta_height'].attrs['description'] = 'Delta height from teleop_base_height_command: dh = h[t] - h[t-1]'
                output_f['delta_height'].attrs['units'] = 'meters'
                output_f['delta_height'].attrs['source'] = 'teleop_base_height_command'

                # Create variable-length dtype for image data (to store JPEG compressed data)
                dt = h5py.special_dtype(vlen=np.uint8)
                left_images_dataset = output_f.create_dataset(
                    "observation_image_left",
                    shape=(num_frames,),
                    dtype=dt
                )
                right_images_dataset = output_f.create_dataset(
                    "observation_image_right",
                    shape=(num_frames,),
                    dtype=dt
                )

                # Store matched camera timestamps and time differences
                matched_camera_timestamps = np.zeros(num_frames, dtype=np.int64)
                time_diffs_ms = np.zeros(num_frames, dtype=np.float32)

                # Statistics
                warning_count = 0
                severe_warning_count = 0
                warning_frames = []

                camera_timestamps_ns = camera_data["timestamps_ns"]
                camera_images_left = camera_data["images_left"]
                camera_images_right = camera_data["images_right"]

                # Batch match timestamps (using binary search optimization)
                robot_timestamps_ns = (robot_timestamps * 1e9).astype(np.int64)

                # Use searchsorted for binary search, find insertion positions
                indices = np.searchsorted(camera_timestamps_ns, robot_timestamps_ns)

                # Handle boundary case: if index equals array length, use last element
                indices = np.clip(indices, 0, len(camera_timestamps_ns) - 1)

                # For each position, check if previous element is closer
                # Create previous index (but not less than 0)
                prev_indices = np.maximum(indices - 1, 0)

                # Compute time differences for current index and previous index
                curr_diffs = np.abs(camera_timestamps_ns[indices] - robot_timestamps_ns)
                prev_diffs = np.abs(camera_timestamps_ns[prev_indices] - robot_timestamps_ns)

                # Select closer index
                better_indices = np.where(curr_diffs < prev_diffs, indices, prev_indices)

                # Batch compute time differences
                matched_ts_ns_array = camera_timestamps_ns[better_indices]
                time_diffs_ms = np.abs(matched_ts_ns_array - robot_timestamps_ns) / 1e6
                matched_camera_timestamps[:] = matched_ts_ns_array

                # Batch check warning thresholds
                severe_mask = time_diffs_ms > self.SEVERE_THRESHOLD_MS
                warning_mask = (time_diffs_ms > self.WARNING_THRESHOLD_MS) & (~severe_mask)

                severe_warning_count = int(np.sum(severe_mask))
                warning_count = int(np.sum(warning_mask))

                # Collect warning frames (only save first 10)
                severe_indices = np.where(severe_mask)[0][:10]
                warning_indices = np.where(warning_mask)[0][:10]

                for i in severe_indices:
                    warning_frames.append({
                        "frame_idx": int(i),
                        "time_diff_ms": float(time_diffs_ms[i]),
                        "level": "severe"
                    })

                for i in warning_indices:
                    if len(warning_frames) < 10:
                        warning_frames.append({
                            "frame_idx": int(i),
                            "time_diff_ms": float(time_diffs_ms[i]),
                            "level": "warning"
                        })

                # Batch write image data (with progress bar)
                progress_bar = tqdm(range(num_frames),
                                    total=num_frames,
                                    desc=f"      Writing {episode_name}",
                                    leave=False,
                                    unit="frame")
                for i in progress_bar:
                    idx = better_indices[i]
                    left_images_dataset[i] = camera_images_left[idx]
                    right_images_dataset[i] = camera_images_right[idx]

                # Save timestamp-related data
                output_f.create_dataset(
                    "camera_timestamp",
                    data=matched_camera_timestamps,
                    compression="gzip"
                )
                output_f.create_dataset(
                    "timestamp_diff_ms",
                    data=time_diffs_ms,
                    compression="gzip"
                )

        return {
            "frame_count": num_frames,
            "warning_count": warning_count,
            "severe_warning_count": severe_warning_count,
            "warning_frames": warning_frames,
            "avg_time_diff_ms": float(np.mean(time_diffs_ms)),
            "max_time_diff_ms": float(np.max(time_diffs_ms)),
            "time_diffs_ms": time_diffs_ms.tolist()  # Temporary storage, will be deleted later
        }
    
    def _read_svo2_file(self, svo2_path: Path) -> Optional[Dict]:
        """Read all images and timestamps from svo2 file"""

        # Create ZED camera object
        zed = sl.Camera()

        # Set initialization parameters
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(svo2_path))
        init_params.svo_real_time_mode = False  # Non-real-time mode for reading
        init_params.coordinate_units = sl.UNIT.METER

        # Open camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"      ❌ Unable to open SVO2 file: {svo2_path}, error: {err}")
            return None

        # Get frame count
        nb_frames = zed.get_svo_number_of_frames()

        # Store data
        timestamps_ns = []
        images_left = []
        images_right = []

        # Create image containers
        left_image = sl.Mat()
        right_image = sl.Mat()

        # Set SVO position to start
        zed.set_svo_position(0)

        # Read frame by frame
        frame_count = 0
        while frame_count < nb_frames:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Get timestamp (nanoseconds)
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                timestamps_ns.append(timestamp.get_nanoseconds())

                # Get left eye image
                zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU)
                # Get right eye image
                zed.retrieve_image(right_image, sl.VIEW.RIGHT, sl.MEM.CPU)

                # Convert images to JPEG bytes
                left_jpeg_bytes = self._mat_to_jpeg_bytes(left_image)
                right_jpeg_bytes = self._mat_to_jpeg_bytes(right_image)

                images_left.append(left_jpeg_bytes)
                images_right.append(right_jpeg_bytes)

                frame_count += 1
            else:
                break

        # Close camera
        zed.close()

        if len(timestamps_ns) == 0:
            return None

        return {
            "timestamps_ns": np.array(timestamps_ns, dtype=np.int64),
            "images_left": images_left,
            "images_right": images_right
        }
    
    def _mat_to_jpeg_bytes(self, mat: sl.Mat) -> np.ndarray:
        """Convert ZED Mat image to JPEG byte array"""
        # Convert Mat to numpy array
        # Use numpy property for simplicity, if not available use get_data
        try:
            image_array = mat.numpy()
        except:
            image_array = mat.get_data()

        # Get number of channels
        channels = mat.get_channels()

        # Try to use cv2 (OpenCV) for JPEG encoding, usually faster than PIL
        try:
            import cv2
            # ZED image format is usually BGRA or BGR, need to convert to RGB
            if channels == 4:  # BGRA
                # Extract RGB channels, remove Alpha channel, and convert BGR to RGB
                image_rgb = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_BGR2RGB)
            elif channels == 3:  # BGR
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            elif channels == 1:  # Grayscale
                # Convert grayscale to RGB
                image_rgb = cv2.cvtColor(image_array.squeeze(), cv2.COLOR_GRAY2RGB)
            else:
                # Default try to use directly
                image_rgb = image_array

            # Use cv2.imencode for JPEG encoding (usually faster than PIL)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, jpeg_bytes = cv2.imencode('.jpg', image_rgb, encode_param)
            return np.frombuffer(jpeg_bytes, dtype=np.uint8)
        except ImportError:
            # If cv2 is not available, use PIL
            # Create PIL image
            # ZED image format is usually BGRA or BGR, need to convert to RGB
            if channels == 4:  # BGRA
                # Extract RGB channels, remove Alpha channel, and convert BGR to RGB
                image_rgb = Image.fromarray(image_array[:, :, [2, 1, 0]], 'RGB')
            elif channels == 3:  # BGR
                # Convert BGR to RGB
                image_rgb = Image.fromarray(image_array[:, :, [2, 1, 0]], 'RGB')
            elif channels == 1:  # Grayscale
                # Convert grayscale to RGB
                image_rgb = Image.fromarray(image_array.squeeze(), 'L').convert('RGB')
            else:
                # Default try to use directly
                image_rgb = Image.fromarray(image_array)

            # Convert to JPEG bytes
            jpeg_buffer = io.BytesIO()
            image_rgb.save(jpeg_buffer, format='JPEG', quality=95)
            jpeg_bytes = np.frombuffer(jpeg_buffer.getvalue(), dtype=np.uint8)

            return jpeg_bytes
    
    def _save_report(self):
        """Save merge report"""
        report_path = self.output_path / "merge_report.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)

        print(f"\n📄 Merge report saved: {report_path}")

        # Print overall statistics
        print("\n" + "=" * 60)
        print("📊 Overall Statistics")
        print("=" * 60)

        print(f"  Total episodes: {len(self.report['episodes'])}")
        print(f"  Total frames: {self.report['total_frames']}")
        print(f"  Average time difference: {self.report['avg_time_diff_ms']:.2f} ms")
        print(f"  Maximum time difference: {self.report['max_time_diff_ms']:.2f} ms")
        print(f"  Warnings (>{self.WARNING_THRESHOLD_MS}ms): {self.report['total_warnings']}")
        print(f"  Severe warnings (>{self.SEVERE_THRESHOLD_MS}ms): {self.report['total_severe_warnings']}")


def verify_merged_data(merged_hdf5_path: str):
    """Verify merged data"""
    print("\n" + "=" * 60)
    print(f"🔍 Verifying merged data: {merged_hdf5_path}")
    print("=" * 60)

    with h5py.File(merged_hdf5_path, 'r') as f:
        print(f"  📁 Dataset keys: {list(f.keys())[:10]}")

        # Check first dataset
        if len(f.keys()) > 0:
            first_key = list(f.keys())[0]
            print(f"\n  Example dataset '{first_key}':")

            for key in list(f.keys())[:5]:  # Only check first 5
                data = f[key]
                if isinstance(data, h5py.Dataset):
                    print(f"    {key}: shape={data.shape}, dtype={data.dtype}")

            # Try to decode an image
            if "observation_image_left" in f:
                jpeg_data = f["observation_image_left"][0]
                try:
                    img = Image.open(io.BytesIO(jpeg_data.tobytes()))
                    print(f"    ✅ Image decode successful: {img.size}, mode={img.mode}")
                except Exception as e:
                    print(f"    ❌ Image decode failed: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Merge ZED camera (svo2 format) data into robot HDF5 data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all episodes in default directory
  python merge_data.py

  # Specify data directory and output directory
  python merge_data.py --data_dir ./robot_data --output_dir ./merged_data

  # Use 4 threads for parallel processing (accelerated)
  python merge_data.py --num_workers 4 --data_dir ./robot_data --output_dir ./merged_data

  # Do not compute delta EEF data
  python merge_data.py --no-delta-eef --data_dir ./robot_data --output_dir ./merged_data

Description:
  The script recursively searches for all episode_*.hdf5 files and their corresponding svo2 files under data_dir,
  then merges them and outputs to output_dir, with file names renumbered as episode_0.hdf5, episode_1.hdf5, ...
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Data directory path, will recursively search for all episode_*.hdf5 files. Defaults to robot_data under script directory'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory path. Defaults to merged_data under script directory'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel threads (default 1, serial processing). Recommended to set to number of CPU cores'
    )

    parser.add_argument(
        '--compute-delta-eef',
        action='store_true',
        default=True,
        dest='compute_delta_eef',
        help='Compute and add delta EEF data (enabled by default)'
    )

    parser.add_argument(
        '--no-delta-eef',
        action='store_false',
        dest='compute_delta_eef',
        help='Do not compute delta EEF data'
    )

    args = parser.parse_args()

    # Determine data directory
    if args.data_dir is None:
        base_path = Path(__file__).parent.resolve()
        data_dir = base_path / "robot_data"
    else:
        data_dir = Path(args.data_dir).resolve()

    # Determine output directory
    if args.output_dir is None:
        base_path = Path(__file__).parent.resolve()
        output_dir = base_path / "merged_data"
    else:
        output_dir = Path(args.output_dir).resolve()

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel threads: {args.num_workers}")
    print(f"Compute delta EEF: {args.compute_delta_eef}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if data directory exists
    if not data_dir.exists():
        print(f"❌ Error: Data directory does not exist: {data_dir}")
        return

    # Execute merge
    merger = CameraRobotMerger(
        str(data_dir),
        str(output_dir),
        num_workers=args.num_workers,
        compute_delta_eef=args.compute_delta_eef
    )
    merger.merge_all()

    # Verify merge results (only verify first few files)
    if output_dir.exists():
        hdf5_files = sorted(output_dir.glob("episode_*.hdf5"))[:3]  # Only verify first 3
        for hdf5_file in hdf5_files:
            verify_merged_data(str(hdf5_file))

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
