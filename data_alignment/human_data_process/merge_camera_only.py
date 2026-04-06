#!/usr/bin/env python3
"""
Camera Data Merge Script (Images Only)

Function: Merge ZED camera image data into robot HDF5 file, ensuring data synchronization through timestamp alignment

Configuration:
- Image storage format: JPEG compression (quality 95)
- output location: merged_data/
- Timestamp difference handling: Use nearest image, mark in report if exceeds threshold
- Data format: Read camera data from svo2 file

Performance optimizations:
- Use binary search to optimize timestamp matching (O(n*log(m)) instead of O(n*m))
- Support multithreading for parallel processing of multiple episodes
- Use OpenCV for JPEG encoding (faster than PIL if available)
- batch processing of timestamp matching and warning detection
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


def _process_single_episode_wrapper(merger, hdf5_file, svo2_file, output_file, episode_name):
    """Wrapper function for processing a single episode (for parallel processing)"""
    return merger._merge_episode(hdf5_file, svo2_file, output_file, episode_name)


def extract_episode_index(filename: str) -> str:
    """
    Extract episode index from filename.
    Only supports format: downsample_episode_{index}.hdf5 -> {index}

    Args:
        filename: Filename (e.g.: downsample_episode_26.hdf5)

    Returns:
        Index string (e.g.: "26" or "0")
    """
    import re
    # Only match downsample_episode_{number}
    match = re.search(r'downsample_episode_(\d+)', filename)
    if match:
        return match.group(1)
    # if no match, return "0" as default
    return "0"


class CameraRobotMerger:
    """Camera and Robot Data Merger (Images Only)"""
    
    # Timestamp difference thresholds (milliseconds)
    WARNING_THRESHOLD_MS = 20.0   # Warning threshold
    SEVERE_THRESHOLD_MS = 50.0    # Severe warning threshold
    
    def __init__(self, data_dir: str, output_dir: str, num_workers: int = 1):
        """
        Initialize the merger
        
        Args:
            data_dir: Data directory path (directory containing robot_data, or robot_data directory itself)
            output_dir: output directory path
            num_workers: Number of parallel threads
        """
        data_dir = Path(data_dir).resolve()
        output_dir = Path(output_dir).resolve()
        
        # Determine data_dir structure:
        # 1. ifdata_dir/robot_dataexists,thendata_diris working directory, robot_data is at data_dir/robot_data
        # 2. ifdata_dir/task_name/hdf5exists（e.g. data_dir/toy/hdf5）,thendata_diris robot_data directory
        # 3. ifdata_dir/hdf5exists,thendata_diris robot_data directory（single task case,uncommon)
        if (data_dir / "robot_data").exists():
            # data_diris working directory, robot_data is in subdirectory
            self.robot_data_path = data_dir / "robot_data"
            self.base_path = data_dir
        elif any((data_dir / d).is_dir() and (data_dir / d / "hdf5").exists() 
                 for d in data_dir.iterdir() if d.is_dir()):
            # data_diritself isrobot_datadirectory（contains multiple task_name subdirectories）
            self.robot_data_path = data_dir
            self.base_path = data_dir.parent
        else:
            # Assume data_diris robot_data directory
            self.robot_data_path = data_dir
            self.base_path = data_dir.parent
        
        self.output_path = output_dir
        self.num_workers = num_workers
        
        # Merge report
        self.report = {
            "merge_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": {}
        }
    
    def merge_all(self, task_name: Optional[str] = None):
        """
        Merge dataset
        
        Args:
            task_name: Optional, specify task name to process. if None, process all tasks
        """
        print("=" * 80)
        print("🔄 Start merging camera image data into robot HDF5 file")
        print("=" * 80)
        
        # Createoutput directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Data directory: {self.robot_data_path}")
        print(f"📁 output directory: {self.output_path}")
        
        # Get task list（task_name）
        # Supports multiple structures（Only supports downsample_episode_{N}.hdf5 format）:
        # 1. task_name/hdf5/downsample_episode_{N}.hdf5 (new structure, directly under hdf5 directory)
        # 2. task_name/hdf5/batch_folder/downsample_episode_{N}.hdf5 (old structure, with batch folders)
        # 3. task_name/downsample_episode_{N}.hdf5 (directly under task_name directory)
        # 4. task_name/{batch folders}/downsample_episode_{N}.hdf5 (batch folders directly under task_name)
        task_names = []
        for d in self.robot_data_path.iterdir():
            if d.is_dir():
                # check if hdf5 subdirectory exists
                hdf5_subdir = d / "hdf5"
                if hdf5_subdir.exists():
                    # check if downsample_episode file exist in hdf5 directory（directly in hdf5 directory)
                    has_direct_episodes = any(
                        f.name.startswith("downsample_episode_") and f.suffix == ".hdf5" 
                        for f in hdf5_subdir.iterdir() if f.is_file()
                    )
                    # or batch folders with downsample_episode file
                    has_batch_episodes = any(
                        (batch_dir / f).name.startswith("downsample_episode_")
                        and (batch_dir / f).suffix == ".hdf5"
                        for batch_dir in hdf5_subdir.iterdir() if batch_dir.is_dir()
                        for f in batch_dir.iterdir() if f.is_file()
                    )
                    if has_direct_episodes or has_batch_episodes:
                        task_names.append(d.name)
                else:
                    # check if downsample_episode file exist directly in directory
                    has_direct_episodes = any(
                        f.name.startswith("downsample_episode_") and f.suffix == ".hdf5" 
                        for f in d.iterdir() if f.is_file()
                    )
                    # or batch folders with downsample_episode file (batch folders directly under task_name)
                    has_batch_episodes = any(
                        (batch_dir / f).name.startswith("downsample_episode_")
                        and (batch_dir / f).suffix == ".hdf5"
                        for batch_dir in d.iterdir() if batch_dir.is_dir()
                        for f in batch_dir.iterdir() if f.is_file()
                    )
                    if has_direct_episodes or has_batch_episodes:
                        task_names.append(d.name)
        task_names = sorted(task_names)
        
        # if task_name is specified, only process that task
        if task_name is not None:
            if task_name not in task_names:
                print(f"❌ Error: tasks '{task_name}' does not exist!")
                print(f"   Available tasks: {task_names}")
                return
            task_names = [task_name]
        
        print(f"📊 Found {len(task_names)} tasks: {task_names}")
        
        for task_name in task_names:
            self._merge_task(task_name)
        
        # Save report
        self._save_report()
        
        print("\n" + "=" * 80)
        print("✅ Merge completed!")
        print("=" * 80)
    
    def _merge_task(self, task_name: str):
        """Merge all episodes for a single task"""
        print(f"\n{'='*60}")
        print(f"📦 processing task: {task_name}")
        print("=" * 60)
        
        task_dir = self.robot_data_path / task_name
        
        # Supports multiple structures:
        # 1. task_name/hdf5/episode_{N}.hdf5 (new structure, directly under hdf5 directory)
        # 2. task_name/hdf5/batch_folder/episode_{N}.hdf5 (old structure, has batch folders, need recursive search)
        # 3. task_name/episode_{N}.hdf5 (directly under task_name directory)
        # 4. task_name/{batch folders}/episode_{N}.hdf5 (batch folders directly under task_name, hdf5 and svo2 in same batch folder)
        if (task_dir / "hdf5").exists():
            hdf5_base = task_dir / "hdf5"
            svo2_base = task_dir / "svo2"
        else:
            # directly under task_name directory (may be batch folders directly under task_name)
            hdf5_base = task_dir
            svo2_base = task_dir
        
        if not hdf5_base.exists():
            print(f"  ❌ HDF5Data directory does not exist: {hdf5_base}")
            return
        
        # When batch folders are directly under task_name, svo2 file are also in the same batch folder, no need for separate svo2_base check
        # We will check if svo2 file exist when processing each episode
        
        # Get all hdf5 file（recursive search, including those in batch folders）
        # Only supports downsample_episode_*.hdf5 format
        hdf5_file = sorted(
            hdf5_base.glob("**/downsample_episode_*.hdf5"),
            key=lambda x: int(extract_episode_index(x.name))
        )
        
        if len(hdf5_file) == 0:
            print(f"  ⚠️  No episode file found")
            return
        
        print(f"  📋 Found {len(hdf5_file)} episodes")
        
        # Createoutput directory
        output_task_dir = self.output_path / task_name
        output_task_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize task report
        task_report = {
            "episodes": {},
            "total_frame": 0,
            "total_warnings": 0,
            "total_severe_warnings": 0,
            "avg_time_diff_ms": 0.0,
            "max_time_diff_ms": 0.0
        }
        
        all_time_diffs = []
        
        # Prepare processing parameters for all episodes
        episode_tasks = []
        for hdf5_file in hdf5_file:
            # Extract index from hdf5 file name（Only supports downsample_episode_ format）
            episode_index = extract_episode_index(hdf5_file.name)
            # svo2 file always use episode_{index}.svo2 format
            svo2_episode_name = f"episode_{episode_index}"
            # output file maintain original hdf5 file names（downsample_episode_{index}）
            episode_name = hdf5_file.stem  # downsample_episode_0
            
            # Find corresponding svo2 file
            # Calculate relative path from hdf5_base
            relative_path = hdf5_file.relative_to(hdf5_base)
            
            # Determine if batch folders exist
            if len(relative_path.parts) > 1:
                # Has batch folders
                batch_folder = relative_path.parts[0]
                # check if separate hdf5 and svo2 directory structure exists
                if (task_dir / "hdf5").exists():
                    # Structure:task_name/hdf5/batch_folder/downsample_episode_0.hdf5 -> task_name/svo2/batch_folder/episode_0.svo2
                    svo2_file = svo2_base / relative_path.parent / f"{svo2_episode_name}.svo2"
                else:
                    # Structure:task_name/batch_folder/downsample_episode_0.hdf5 -> task_name/batch_folder/episode_0.svo2
                    # hdf5 and svo2 in same batch folder
                    svo2_file = hdf5_file.parent / f"{svo2_episode_name}.svo2"
            else:
                # directly under hdf5 directory
                batch_folder = None
                if (task_dir / "hdf5").exists():
                    # Structure:task_name/hdf5/downsample_episode_0.hdf5 -> task_name/svo2/episode_0.svo2
                    svo2_file = svo2_base / f"{svo2_episode_name}.svo2"
                else:
                    # Structure:task_name/downsample_episode_0.hdf5 -> task_name/episode_0.svo2
                    svo2_file = hdf5_file.parent / f"{svo2_episode_name}.svo2"
            
            if not svo2_file.exists():
                print(f"    ⚠️  {episode_name} No corresponding SVO2 file: {svo2_file}")
                continue
            
            # Maintain original batch folder structure and episode file names
            if batch_folder:
                # Has batch folders: output to output_dir/task_name/batch_folder/downsample_episode_{index}.hdf5
                output_batch_dir = output_task_dir / batch_folder
                output_batch_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_batch_dir / f"{episode_name}.hdf5"
            else:
                # No batch folders: output to output_dir/task_name/downsample_episode_{index}.hdf5
                output_file = output_task_dir / f"{episode_name}.hdf5"
            
            episode_tasks.append((hdf5_file, svo2_file, output_file, episode_name))
        
        # process episodes in parallel or serial
        if self.num_workers > 1 and len(episode_tasks) > 1:
            # Parallel processing（use thread pool, because ZED SDK may not support multiprocessing）
            print(f"  🚀 use {self.num_workers} threads for parallel processing")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Prepare parameters
                futures = []
                for hdf5_file, svo2_file, output_file, episode_name in episode_tasks:
                    future = executor.submit(
                        _process_single_episode_wrapper,
                        self, hdf5_file, svo2_file, output_file, episode_name
                    )
                    futures.append((future, episode_name))
                
                # Collect results
                for future, episode_name in tqdm(futures, desc=f"  process {task_name}"):
                    try:
                        episode_report = future.result()
                        if episode_report:
                            task_report["episodes"][episode_name] = episode_report
                            task_report["total_frame"] += episode_report["frame_count"]
                            task_report["total_warnings"] += episode_report["warning_count"]
                            task_report["total_severe_warnings"] += episode_report["severe_warning_count"]
                            all_time_diffs.extend(episode_report["time_diffs_ms"])
                    except Exception as e:
                        print(f"    ❌ process {episode_name} error occurred: {e}")
        else:
            # Serial processing（Maintain original logic）
            for hdf5_file, svo2_file, output_file, episode_name in tqdm(episode_tasks, desc=f"  process {task_name}"):
                episode_report = self._merge_episode(hdf5_file, svo2_file, output_file, episode_name)
                
                if episode_report:
                    task_report["episodes"][episode_name] = episode_report
                    task_report["total_frame"] += episode_report["frame_count"]
                    task_report["total_warnings"] += episode_report["warning_count"]
                    task_report["total_severe_warnings"] += episode_report["severe_warning_count"]
                    all_time_diffs.extend(episode_report["time_diffs_ms"])
        
        # Compute statistics
        if all_time_diffs:
            task_report["avg_time_diff_ms"] = float(np.mean(all_time_diffs))
            task_report["max_time_diff_ms"] = float(np.max(all_time_diffs))
        
        # Simplify report (don't save per-frame time differences)
        for ep_name in task_report["episodes"]:
            if "time_diffs_ms" in task_report["episodes"][ep_name]:
                del task_report["episodes"][ep_name]["time_diffs_ms"]
        
        self.report["datasets"][task_name] = task_report
        
        # Print task statistics
        print(f"\n  📊 {task_name} Statistics:")
        print(f"     Total episodes: {len(task_report['episodes'])}")
        print(f"     Total frame: {task_report['total_frame']}")
        print(f"     Average time difference: {task_report['avg_time_diff_ms']:.2f} ms")
        print(f"     Maximum time difference: {task_report['max_time_diff_ms']:.2f} ms")
        print(f"     Warnings (>{self.WARNING_THRESHOLD_MS}ms): {task_report['total_warnings']}")
        print(f"     Severe warnings (>{self.SEVERE_THRESHOLD_MS}ms): {task_report['total_severe_warnings']}")
    
    def _merge_episode(self, robot_hdf5_path: Path, 
                       svo2_path: Path,
                       output_hdf5_path: Path,
                       episode_name: str) -> Optional[Dict]:
        """Merge a single episode (image data only)"""
        
        # Read svo2 file, get all images and timestamps
        camera_data = self._read_svo2_file(svo2_path)
        
        if camera_data is None or len(camera_data["timestamps_ns"]) == 0:
            return None
        
        # Open robot HDF5 file
        with h5py.File(robot_hdf5_path, 'r') as robot_f:
            # use local_timestamps_ns（in nanoseconds）
            if "local_timestamps_ns" not in robot_f:
                print(f"      ⚠️  {episode_name} Cannot find local_timestamps_ns field")
                return None
            
            robot_timestamps_ns = robot_f["local_timestamps_ns"][:].astype(np.int64)
            num_frame = len(robot_timestamps_ns)
            
            camera_timestamps_ns = camera_data["timestamps_ns"]
            print(f"      📹 Camera: [{camera_timestamps_ns[0]}~{camera_timestamps_ns[-1]}]ns, {len(camera_timestamps_ns)}frame")
            print(f"      🤖 Robot: [{robot_timestamps_ns[0]}~{robot_timestamps_ns[-1]}]ns, {num_frame}frame")
            
            # Create output file
            with h5py.File(output_hdf5_path, 'w') as output_f:
                # Copy original data
                for key in robot_f.keys():
                    data = robot_f[key][:]
                    output_f.create_dataset(key, data=data, compression="gzip")
                
                # Create variable-length dtype for image data (to store JPEG compressed data)
                dt = h5py.special_dtype(vlen=np.uint8)
                left_images_dataset = output_f.create_dataset(
                    "observation_image_left", 
                    shape=(num_frame,), 
                    dtype=dt
                )
                right_images_dataset = output_f.create_dataset(
                    "observation_image_right", 
                    shape=(num_frame,), 
                    dtype=dt
                )
                
                # Store matched camera timestampsand time difference
                matched_camera_timestamps = np.zeros(num_frame, dtype=np.int64)
                time_diffs_ms = np.zeros(num_frame, dtype=np.float32)
                
                # Statistics
                warning_count = 0
                severe_warning_count = 0
                warning_frame = []
                
                camera_images_left = camera_data["images_left"]
                camera_images_right = camera_data["images_right"]
                
                # robot_timestamps_ns and camera_timestamps_nsalready processed above, use directly
                
                # Use searchsorted for binary search, find insertion positions
                indices = np.searchsorted(camera_timestamps_ns, robot_timestamps_ns)
                
                # Handle boundary case:if index equals array length, use last element
                indices = np.clip(indices, 0, len(camera_timestamps_ns) - 1)
                
                # For each position, check if previous element is closer
                # Create previous index (but not less than 0)
                prev_indices = np.maximum(indices - 1, 0)
                
                # Compute time differences for current index and previous index
                curr_diffs = np.abs(camera_timestamps_ns[indices] - robot_timestamps_ns)
                prev_diffs = np.abs(camera_timestamps_ns[prev_indices] - robot_timestamps_ns)
                
                # Select closer index
                better_indices = np.where(curr_diffs < prev_diffs, indices, prev_indices)
                
                # batch compute time differences
                matched_ts_ns_array = camera_timestamps_ns[better_indices]
                time_diffs_ms = np.abs(matched_ts_ns_array - robot_timestamps_ns) / 1e6
                matched_camera_timestamps[:] = matched_ts_ns_array
                
                # batch check warning thresholds
                severe_mask = time_diffs_ms > self.SEVERE_THRESHOLD_MS
                warning_mask = (time_diffs_ms > self.WARNING_THRESHOLD_MS) & (~severe_mask)
                
                severe_warning_count = int(np.sum(severe_mask))
                warning_count = int(np.sum(warning_mask))
                
                # CollectWarning frame（Only save first10）
                severe_indices = np.where(severe_mask)[0][:10]
                warning_indices = np.where(warning_mask)[0][:10]
                
                for i in severe_indices:
                    warning_frame.append({
                        "frame_idx": int(i),
                        "time_diff_ms": float(time_diffs_ms[i]),
                        "level": "severe"
                    })
                
                for i in warning_indices:
                    if len(warning_frame) < 10:
                        warning_frame.append({
                            "frame_idx": int(i),
                            "time_diff_ms": float(time_diffs_ms[i]),
                            "level": "warning"
                        })
                
                # batch write image data (with progress bar)
                progress_bar = tqdm(range(num_frame),
                                    total=num_frame,
                                    desc=f"      Write {episode_name}",
                                    leave=False,
                                    unit="frame")
                for i in progress_bar:
                    idx = better_indices[i]
                    left_images_dataset[i] = camera_images_left[idx]
                    right_images_dataset[i] = camera_images_right[idx]
                
                # SaveTimestamp-related data
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
            "frame_count": num_frame,
            "warning_count": warning_count,
            "severe_warning_count": severe_warning_count,
            "warning_frame": warning_frame,
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
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        
        # Open camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            # Some error codes are warnings, not fatal errors, can continue execution
            non_fatal_errors = [
                sl.ERROR_CODE.CALIBRATION_FILE_NOT_AVAILABLE,  # calibration file issue
            ]
            # check if SENSORS_NOT_AVAILABLE or other non-fatal errors exist
            try:
                non_fatal_errors.append(sl.ERROR_CODE.SENSORS_NOT_AVAILABLE)
            except AttributeError:
                pass
            
            if err in non_fatal_errors or "CALIBRATION" in str(err):
                print(f"      ⚠️  SVO2 file calibration warning（can be ignored）: {svo2_path}, Warning: {err}")
                # Continue execution, don't return None
            else:
                print(f"      ❌ Unable to open SVO2 file: {svo2_path}, Error: {err}")
                return None
        
        # Getframe count
        nb_frame = zed.get_svo_number_of_frame()
        
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
        while frame_count < nb_frame:
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
        """Convert ZED Mat image to JPEG byte array (quality 95)"""
        # Convert Mat to numpy array
        # use numpy property is simpler, if not available then use get_data
        try:
            image_array = mat.numpy()
        except:
            image_array = mat.get_data()
        
        # Getnumber of channels
        channels = mat.get_channels()
        
        # Try using cv2（OpenCV) for JPEG encoding, usually faster than PIL
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
            
            # use cv2.imencode for JPEG encoding（quality 95）
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, jpeg_bytes = cv2.imencode('.jpg', image_rgb, encode_param)
            return np.frombuffer(jpeg_bytes, dtype=np.uint8)
        except ImportError:
            # if cv2 not available, use PIL
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
            
            # Convert to JPEG bytes（quality 95）
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
        
        total_frame = 0
        total_warnings = 0
        total_severe = 0
        
        for task_name, task_report in self.report["datasets"].items():
            total_frame += task_report["total_frame"]
            total_warnings += task_report["total_warnings"]
            total_severe += task_report["total_severe_warnings"]
            print(f"  {task_name}:")
            print(f"    frame count: {task_report['total_frame']}")
            print(f"    Average time difference: {task_report['avg_time_diff_ms']:.2f} ms")
            print(f"    Warning/Severe warning: {task_report['total_warnings']}/{task_report['total_severe_warnings']}")
        
        print(f"\n  Total:")
        print(f"    Total frame: {total_frame}")
        print(f"    Total warnings: {total_warnings}")
        print(f"    Total severe warnings: {total_severe}")


def merge_simple_mode(hdf5_dir: Path, svo2_dir: Path, output_dir: Path, num_workers: int = 1):
    """
    Simplified mode merge: directly find and merge file in specified directory
    
    Args:
        hdf5_dir: hdf5 file directory,  contains downsample_episode_*.hdf5
        svo2_dir: svo2 file directory,  contains episode_*.svo2
        output_dir: output directory
        num_workers: Number of parallel threads
    """
    print("=" * 80)
    print("🔄 Start merging camera image data into robot HDF5 file (simplified mode)")
    print("=" * 80)
    
    # Createoutput directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all downsample_episode_*.hdf5 file
    hdf5_file = sorted(
        hdf5_dir.glob("downsample_episode_*.hdf5"),
        key=lambda x: int(extract_episode_index(x.name))
    )
    
    if len(hdf5_file) == 0:
        print(f"⚠️  No downsample_episode_*.hdf5 file found")
        return
    
    print(f"📋 Found {len(hdf5_file)}  downsample_episode file")
    
    # Create merger instance（to reuse its methods）
    merger = CameraRobotMerger.__new__(CameraRobotMerger)
    merger.num_workers = num_workers
    merger.report = {
        "merge_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {}
    }
    merger.output_path = output_dir
    
    # Prepare task list
    episode_tasks = []
    for hdf5_file in hdf5_file:
        # Extract index
        episode_index = extract_episode_index(hdf5_file.name)
        
        # Find corresponding svo2 file
        svo2_file = svo2_dir / f"episode_{episode_index}.svo2"
        
        if not svo2_file.exists():
            print(f"  ⚠️  {hdf5_file.name} No corresponding SVO2 file: {svo2_file.name}")
            continue
        
        # output file
        output_file = output_dir / f"episode_{episode_index}.hdf5"
        episode_name = f"episode_{episode_index}"
        
        episode_tasks.append((hdf5_file, svo2_file, output_file, episode_name))
    
    if len(episode_tasks) == 0:
        print("⚠️  No episodes to process")
        return
    
    print(f"📋 Prepare to process {len(episode_tasks)}  episodes")
    
    # Statistics data
    all_time_diffs = []
    task_report = {
        "episodes": {},
        "total_frame": 0,
        "total_warnings": 0,
        "total_severe_warnings": 0,
        "avg_time_diff_ms": 0.0,
        "max_time_diff_ms": 0.0
    }
    
    # processing episodes
    if num_workers > 1 and len(episode_tasks) > 1:
        print(f"🚀 use {num_workers} threads for parallel processing")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for hdf5_file, svo2_file, output_file, episode_name in episode_tasks:
                future = executor.submit(
                    _process_single_episode_wrapper,
                    merger, hdf5_file, svo2_file, output_file, episode_name
                )
                futures.append((future, episode_name))
            
            for future, episode_name in tqdm(futures, desc="processing progress"):
                try:
                    episode_report = future.result()
                    if episode_report:
                        task_report["episodes"][episode_name] = episode_report
                        task_report["total_frame"] += episode_report["frame_count"]
                        task_report["total_warnings"] += episode_report["warning_count"]
                        task_report["total_severe_warnings"] += episode_report["severe_warning_count"]
                        all_time_diffs.extend(episode_report["time_diffs_ms"])
                except Exception as e:
                    print(f"  ❌ process {episode_name} error occurred: {e}")
    else:
        for hdf5_file, svo2_file, output_file, episode_name in tqdm(episode_tasks, desc="processing progress"):
            episode_report = merger._merge_episode(hdf5_file, svo2_file, output_file, episode_name)
            
            if episode_report:
                task_report["episodes"][episode_name] = episode_report
                task_report["total_frame"] += episode_report["frame_count"]
                task_report["total_warnings"] += episode_report["warning_count"]
                task_report["total_severe_warnings"] += episode_report["severe_warning_count"]
                all_time_diffs.extend(episode_report["time_diffs_ms"])
    
    # Compute statistics
    if all_time_diffs:
        task_report["avg_time_diff_ms"] = float(np.mean(all_time_diffs))
        task_report["max_time_diff_ms"] = float(np.max(all_time_diffs))
    
    # Simplify report
    for ep_name in task_report["episodes"]:
        if "time_diffs_ms" in task_report["episodes"][ep_name]:
            del task_report["episodes"][ep_name]["time_diffs_ms"]
    
    merger.report["datasets"]["simple_mode"] = task_report
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"   Total episodes: {len(task_report['episodes'])}")
    print(f"   Total frame: {task_report['total_frame']}")
    print(f"   Average time difference: {task_report['avg_time_diff_ms']:.2f} ms")
    print(f"   Maximum time difference: {task_report['max_time_diff_ms']:.2f} ms")
    print(f"   Warnings (>{CameraRobotMerger.WARNING_THRESHOLD_MS}ms): {task_report['total_warnings']}")
    print(f"   Severe warnings (>{CameraRobotMerger.SEVERE_THRESHOLD_MS}ms): {task_report['total_severe_warnings']}")
    
    # Save report
    report_path = output_dir / "merge_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(merger.report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Merge report saved: {report_path}")
    
    # Verify merge results (only verify first few file)
    hdf5_file_out = list(output_dir.glob("episode_*.hdf5"))[:3]
    for hdf5_file in hdf5_file_out:
        verify_merged_data(str(hdf5_file))
    
    print("\n" + "=" * 80)
    print("✅ Merge completed!")
    print(f"   output directory: {output_dir}")
    print("=" * 80)


def verify_merged_data(merged_hdf5_path: str):
    """Verify merged data"""
    print("\n" + "=" * 60)
    print(f"🔍 Verifying merged data: {merged_hdf5_path}")
    print("=" * 60)
    
    with h5py.File(merged_hdf5_path, 'r') as f:
        print(f"  📁 Dataset keys: {list(f.keys())[:10]}")
        
        # check first dataset
        if len(f.keys()) > 0:
            first_key = list(f.keys())[0]
            print(f"\n  Example dataset '{first_key}':")
            
            for key in list(f.keys())[:5]:  # Only check first5
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
        description="Merge ZED camera (svo2 format) image data into robot HDF5 data file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplified mode: specify dataset-dir, automatically find hdf5/ and svo2/ under it
  python merge_camera_only.py --dataset-dir /path/to/data --output-dir /path/to/output
  
  # Use 8 threads for parallel processing
  python merge_camera_only.py --dataset-dir /path/to/data --output-dir /path/to/output --num-workers 8
  
  # Old mode: Specify data directory and output directory（Compatible with old usage）
  python merge_camera_only.py --data_dir ./robot_data --output_dir ./merged_data
        """
    )
    
    # New simplified parameters
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        help='Data directory path, will find downsample_episode_*.hdf5 under {dataset-dir}/hdf5/, and episode_*.svo2 under {dataset-dir}/svo2/'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='output directory path（New parameter, use with --dataset-dir）'
    )
    
    # Old parameter（Maintain compatibility）
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='[Old parameter] Data directory path (directory containing robot_data, or robot_data directory itself)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='[Old parameter] output directory path'
    )
    
    parser.add_argument(
        '--task_name',
        type=str,
        default=None,
        help='[Old parameter] Specify task name to process（Optional）'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel threads (default 1, serial processing)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='[Old parameter] Number of parallel threads'
    )
    
    args = parser.parse_args()
    
    # Determine number of threads（Prioritize new parameters）
    num_workers = args.num_workers if args.num_workers is not None else args.num_workers
    if num_workers is None:
        num_workers = 1
    
    # Determine whether to use new or old mode
    if args.dataset_dir is not None:
        # New simplified mode
        dataset_dir = Path(args.dataset_dir).resolve()
        
        # Determine output directory
        if args.output_dir is not None:
            output_dir = Path(args.output_dir).resolve()
        else:
            # Default output to dataset_dir/merged/
            output_dir = dataset_dir / "merged"
        
        print(f"📁 Data directory: {dataset_dir}")
        print(f"📁 output directory: {output_dir}")
        print(f"🔧 Number of threads: {num_workers}")
        print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # check data directory
        hdf5_dir = dataset_dir / "hdf5"
        svo2_dir = dataset_dir / "svo2"
        
        if not hdf5_dir.exists():
            print(f"❌ Error: hdf5 directory does not exist: {hdf5_dir}")
            return
        
        if not svo2_dir.exists():
            print(f"❌ Error: svo2 directory does not exist: {svo2_dir}")
            return
        
        # Execute simplified mode merge
        merge_simple_mode(hdf5_dir, svo2_dir, output_dir, num_workers)
    else:
        # Old mode（Maintain compatibility）
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
        print(f"output directory: {output_dir}")
        if args.task_name:
            print(f"Task name: {args.task_name}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # check if data directory exists
        if not data_dir.exists():
            print(f"❌ Error: Data directory does not exist: {data_dir}")
            return
        
        # Execute merge
        merger = CameraRobotMerger(
            str(data_dir), 
            str(output_dir), 
            num_workers=num_workers
        )
        merger.merge_all(task_name=args.task_name)
        
        # Verify merge results (only verify first few file)
        if output_dir.exists():
            hdf5_file = list(output_dir.rglob("*.hdf5"))[:3]  # Only verify first3
            for hdf5_file in hdf5_file:
                verify_merged_data(str(hdf5_file))
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

