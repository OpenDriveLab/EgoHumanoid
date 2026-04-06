"""
Script for downsampling processed episode data

Function: 
  1) Read processed episode_{index}.hdf5 file
  2) Read navigation_command, calculate average for every 5 frame
  3) Convert downsampled navigation_command to teleop_navigate_command（discretization processing）
  4) Read processed/positions_xyz, calculate average z for every 5 frame
  5) Calculate delta_height（change in z, first frame is 0）
  6) Inherit timestamps from original file（local_timestamps_ns）and downsample
  7) output downsampled episode file to specified path

Usage:
  Single file mode:
    python downsample_episode.py <input_episode> [--output <output_path>] [--downsample-rate 5]
    if --output not specified, output file will be saved in the same directory as input file, named downsample_episode_{index}.hdf5
  
  batch processing mode:
    python downsample_episode.py --dataset-dir <dataset_dir> [--downsample-rate 5]
    
  batch processing mode will save processed file in the same directory as original file:
    Input: dataset_dir/hdf5/episode_x.hdf5
    output: dataset_dir/hdf5/downsample_episode_x.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def downsample_navigation_command(nav_cmd: np.ndarray, downsample_rate: int) -> np.ndarray:
    """
    Downsample navigation_command, calculate average for each downsample_rate frame.
    
    Args:
        nav_cmd: (N, 3) array, contains [vx, vy, yaw_rate]
        downsample_rate: Downsample rate (take average every how many frame)
    
    Returns:
        (M, 3) array, M = ceil(N / downsample_rate)
    """
    nav_cmd = np.asarray(nav_cmd, dtype=np.float64)
    if nav_cmd.ndim != 2 or nav_cmd.shape[1] != 3:
        raise ValueError(f"navigation_command expected shape (N,3), got {nav_cmd.shape}")
    
    n = len(nav_cmd)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    
    # Calculate how many windows needed
    num_windows = (n + downsample_rate - 1) // downsample_rate
    
    # Initialize output array
    downsampled = np.zeros((num_windows, 3), dtype=np.float64)
    
    for i in range(num_windows):
        start_idx = i * downsample_rate
        end_idx = min(start_idx + downsample_rate, n)
        # Calculate average within this window
        downsampled[i] = np.mean(nav_cmd[start_idx:end_idx], axis=0)
    
    return downsampled


def downsample_z_positions(positions_xyz: np.ndarray, downsample_rate: int) -> np.ndarray:
    """
    Downsample z coordinate of positions_xyz, calculate average for each downsample_rate frame.
    
    Args:
        positions_xyz: (N, 3) array, contains [x, y, z]
        downsample_rate: Downsample rate (take average every how many frame)
    
    Returns:
        (M,) array, M = ceil(N / downsample_rate), contains downsampled z values
    """
    positions_xyz = np.asarray(positions_xyz, dtype=np.float64)
    if positions_xyz.ndim != 2 or positions_xyz.shape[1] != 3:
        raise ValueError(f"positions_xyz expected shape (N,3), got {positions_xyz.shape}")
    
    n = len(positions_xyz)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    
    z_values = positions_xyz[:, 2]
    
    # Calculate how many windows needed
    num_windows = (n + downsample_rate - 1) // downsample_rate
    
    # Initialize output array
    downsampled_z = np.zeros((num_windows,), dtype=np.float64)
    
    for i in range(num_windows):
        start_idx = i * downsample_rate
        end_idx = min(start_idx + downsample_rate, n)
        # Calculate average within this window
        downsampled_z[i] = np.mean(z_values[start_idx:end_idx])
    
    return downsampled_z


def downsample_timestamps(timestamps: np.ndarray, downsample_rate: int) -> np.ndarray:
    """
    Downsample timestamps, take first timestamp (or average) for each downsample_rate frame.
    
    Args:
        timestamps: (N,) array, timestamp data
        downsample_rate: Downsample rate (take one every how many frame)
    
    Returns:
        (M,) array, M = ceil(N / downsample_rate), contains downsampled timestamps
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1:
        raise ValueError(f"timestamps expected shape (N,), got {timestamps.shape}")
    
    n = len(timestamps)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    
    # Calculate how many windows needed
    num_windows = (n + downsample_rate - 1) // downsample_rate
    
    # Initialize output array
    downsampled_ts = np.zeros((num_windows,), dtype=np.float64)
    
    for i in range(num_windows):
        start_idx = i * downsample_rate
        end_idx = min(start_idx + downsample_rate, n)
        # Take first timestamp in window (preserve original meaning of timestamps)
        downsampled_ts[i] = timestamps[start_idx]
    
    return downsampled_ts


def compute_delta_height(z_values: np.ndarray) -> np.ndarray:
    """
    Calculate change in z values (delta_height).
    First frame's change is 0, second frame's delta is change relative to first frame.
    
    Args:
        z_values: (M,) array, downsampled z values
    
    Returns:
        (M,) array, delta_height, first frame is 0
    """
    z_values = np.asarray(z_values, dtype=np.float64)
    if z_values.ndim != 1:
        raise ValueError(f"z_values expected shape (M,), got {z_values.shape}")
    
    if len(z_values) == 0:
        return np.zeros((0,), dtype=np.float64)
    
    delta_height = np.zeros_like(z_values)
    if len(z_values) > 1:
        # First frame is 0, subsequent frame are changes relative to previous frame
        delta_height[1:] = z_values[1:] - z_values[:-1]
    
    return delta_height


def convert_to_teleop_navigate_command(nav_cmd: np.ndarray) -> np.ndarray:
    """
    Convert navigation_command to teleop_navigate_command, perform discretization processing.
    
    Args:
        nav_cmd: (N, 3) array, contains [vx, vy, yaw_rate]
    
    Returns:
        (N, 3) array, teleop_navigate_command
    
    Rules:
        x: > 0.04 -> 0.2, < -0.04 -> -0.1, [-0.04, 0.04] -> 0
        y: > 0.04 -> 0.2, < -0.04 -> -0.2, [-0.04, 0.04] -> 0
        yaw: > 0.2 -> 0.3, < -0.2 -> -0.3, [-0.2, 0.2] -> 0
    """
    nav_cmd = np.asarray(nav_cmd, dtype=np.float64)
    if nav_cmd.ndim != 2 or nav_cmd.shape[1] != 3:
        raise ValueError(f"navigation_command expected shape (N,3), got {nav_cmd.shape}")
    
    n = len(nav_cmd)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    
    teleop_cmd = np.zeros_like(nav_cmd)
    
    # process x (vx)
    x = nav_cmd[:, 0]
    teleop_cmd[:, 0] = np.where(
        x > 0.04, 0.2,
        np.where(x < -0.04, -0.1, 0.0)
    )
    
    # process y (vy)
    y = nav_cmd[:, 1]
    teleop_cmd[:, 1] = np.where(
        y > 0.04, 0.2,
        np.where(y < -0.04, -0.2, 0.0)
    )
    
    # process yaw (yaw_rate)
    yaw = nav_cmd[:, 2]
    teleop_cmd[:, 2] = np.where(
        yaw > 0.2, 0.3,
        np.where(yaw < -0.2, -0.3, 0.0)
    )
    
    return teleop_cmd


def extract_episode_index(filename: str) -> str:
    """
    Extract episode index from filename.
    
    Args:
        filename: Filename (e.g.: episode_26.hdf5 or episode_0.hdf5）
    
    Returns:
        Index string (e.g.: "26" or "0"）
    """
    import re
    # Match episode_{number}.hdf5 or episode_{number}.h5 etc
    match = re.search(r'episode_(\d+)', filename)
    if match:
        return match.group(1)
    # if no match, return "0" as default
    return "0"


def write_h5_dataset(h5_path: str, key: str, data: np.ndarray, attrs: dict | None = None, overwrite: bool = False):
    """Write HDF5 (supports hierarchical keys with '/')."""
    if key is None or len(str(key).strip()) == 0:
        raise ValueError("HDF5 key is empty")

    key = str(key)
    with h5py.File(h5_path, "a") as f:  # use "a" mode to support creating new files
        if "/" in key:
            grp_path, ds_name = key.rsplit("/", 1)
            grp = f.require_group(grp_path)
        else:
            grp = f
            ds_name = key

        if ds_name in grp:
            if not overwrite:
                raise RuntimeError(f"HDF5 dataset already exists: {key} (use --overwrite to replace)")
            del grp[ds_name]

        dset = grp.create_dataset(ds_name, data=np.asarray(data), compression="gzip", compression_opts=4)
        if attrs:
            for k, v in attrs.items():
                try:
                    dset.attrs[k] = v
                except Exception:
                    dset.attrs[k] = str(v)


def process_episode(input_path: Path, output_path: Path, downsample_rate: int = 5, overwrite: bool = False):
    """
    process single episode file, perform downsampling and output.
    
    Args:
        input_path: Input episode file path
        output_path: output episode file path
        downsample_rate: Downsample rate (default 5)
        overwrite: Whether to overwrite existing output file
    """
    # check input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # check if output file already exists
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"output file already exists: {output_path} (Use --overwrite to overwrite)")
    
    # Read input file
    print(f"Read input file: {input_path}")
    timestamps = None
    timestamp_key = None
    
    with h5py.File(str(input_path), "r") as f:
        # check if necessary keys exist
        if "navigation_command" not in f:
            raise KeyError(f"Input file missing navigation_command key: {input_path}")
        
        if "processed/positions_xyz" not in f:
            raise KeyError(f"Input file missing processed/positions_xyz key: {input_path}")
        
        # Readdata
        nav_cmd = f["navigation_command"][:]
        positions_xyz = f["processed/positions_xyz"][:]
        
        # Read original attributes (if available)
        dt = f.attrs.get("collection_interval_s", None)
        if dt is None:
            dt = f["navigation_command"].attrs.get("dt", None)
        if dt is None:
            dt = f["navigation_command"].attrs.get("collection_interval_s", 0.01)
        
        print(f"  originalframe count: {len(nav_cmd)}")
        print(f"  navigation_command shape: {nav_cmd.shape}")
        print(f"  positions_xyz shape: {positions_xyz.shape}")
        print(f"  Time step dt: {dt:.4f} s")
        
        # Readtimestamps
        timestamp_key = "local_timestamps_ns"
        if timestamp_key in f:
            timestamps_raw = f[timestamp_key][:]
            print(f"  Found timestamp key: {timestamp_key}, shape: {timestamps_raw.shape}")
            
            # check if timestamp length matches
            if len(timestamps_raw) == len(nav_cmd):
                timestamps = timestamps_raw
            elif len(timestamps_raw) > len(nav_cmd):
                # Timestamps longer, truncate to nav_cmd length
                print(f"    Warning: timestamp length ({len(timestamps_raw)}) > navigation_command length ({len(nav_cmd)}),will truncate timestamps")
                timestamps = timestamps_raw[:len(nav_cmd)]
            else:
                # Timestamps shorter, only use available portion
                print(f"    Warning: timestamp length ({len(timestamps_raw)}) < navigation_command length ({len(nav_cmd)}),will only use first {len(timestamps_raw)} frame")
                timestamps = timestamps_raw
                # Also truncate nav_cmd and positions_xyz to match timestamp length
                nav_cmd = nav_cmd[:len(timestamps_raw)]
                positions_xyz = positions_xyz[:len(timestamps_raw)]
        else:
            print(f"  Timestamp key not found: {timestamp_key},will skip timestamp processing")
    
    # check if data lengths are consistent
    if len(nav_cmd) != len(positions_xyz):
        raise ValueError(
            f"Data lengths inconsistent: navigation_command={len(nav_cmd)}, "
            f"positions_xyz={len(positions_xyz)}"
        )
    
    # downsamplenavigation_command
    print(f"\nDownsample navigation_command (downsample rate: {downsample_rate})...")
    downsampled_nav_cmd = downsample_navigation_command(nav_cmd, downsample_rate)
    print(f"  Shape after downsampling: {downsampled_nav_cmd.shape}")
    
    # convert toteleop_navigate_command
    print(f"\nconvert to teleop_navigate_command...")
    teleop_nav_cmd = convert_to_teleop_navigate_command(downsampled_nav_cmd)
    print(f"  teleop_navigate_command shape: {teleop_nav_cmd.shape}")
    
    # Downsample z coordinate
    print(f"\nDownsample z coordinate of positions_xyz (downsample rate: {downsample_rate})...")
    downsampled_z = downsample_z_positions(positions_xyz, downsample_rate)
    print(f"  Number of z values after downsampling: {len(downsampled_z)}")
    
    # Calculate delta_height
    print(f"\ncalculate delta_height...")
    delta_height = compute_delta_height(downsampled_z)
    print(f"  delta_height shape: {delta_height.shape}")
    print(f"  first frame delta_height: {delta_height[0]:.6f} (should be 0)")
    if len(delta_height) > 1:
        print(f"  Second frame delta_height: {delta_height[1]:.6f}")
    
    # downsampletimestamps（if exists）
    downsampled_timestamps = None
    if timestamps is not None:
        print(f"\nDownsample timestamps (downsample rate: {downsample_rate})...")
        downsampled_timestamps = downsample_timestamps(timestamps, downsample_rate)
        print(f"  Number of timestamps after downsampling: {len(downsampled_timestamps)}")
    
    # Writeoutput file
    print(f"\nWriteoutput file: {output_path}")
    
    # if output file already exists, delete it (if overwrite allowed)
    if output_path.exists() and overwrite:
        output_path.unlink()
    
    # Writedata
    write_h5_dataset(
        str(output_path),
        "navigation_command",
        downsampled_nav_cmd,
        attrs={
            "description": f"Downsampled navigation_command (downsample rate: {downsample_rate})",
            "original_frame": int(len(nav_cmd)),
            "downsampled_frame": int(len(downsampled_nav_cmd)),
            "downsample_rate": int(downsample_rate),
            "dt": float(dt * downsample_rate),  # Downsampled time step
            "units": "[m/s, m/s, rad/s]",
            "shape": str(downsampled_nav_cmd.shape),
        },
        overwrite=overwrite,
    )
    
    write_h5_dataset(
        str(output_path),
        "teleop_navigate_command",
        teleop_nav_cmd,
        attrs={
            "description": "Discretized command converted based on downsampled navigation_command",
            "original_frame": int(len(nav_cmd)),
            "downsampled_frame": int(len(teleop_nav_cmd)),
            "downsample_rate": int(downsample_rate),
            "dt": float(dt * downsample_rate),
            "units": "[m/s, m/s, rad/s]",
            "shape": str(teleop_nav_cmd.shape),
            "discretization_rules": "x: >0.04->0.2, <-0.04->-0.1, else->0; y: >0.04->0.2, <-0.04->-0.2, else->0; yaw: >0.2->0.3, <-0.2->-0.3, else->0",
        },
        overwrite=overwrite,
    )
    
    write_h5_dataset(
        str(output_path),
        "delta_height",
        delta_height,
        attrs={
            "description": "Downsampled z coordinate change, first frame is 0, subsequent frame are changes relative to previous frame",
            "original_frame": int(len(positions_xyz)),
            "downsampled_frame": int(len(delta_height)),
            "downsample_rate": int(downsample_rate),
            "dt": float(dt * downsample_rate),
            "units": "m",
            "shape": str(delta_height.shape),
        },
        overwrite=overwrite,
    )
    
    # Writetimestamps（if exists）
    if downsampled_timestamps is not None:
        # Try to read timestamp attributes from original file
        original_timestamp_attrs = {}
        if timestamp_key:
            with h5py.File(str(input_path), "r") as f:
                if timestamp_key in f:
                    for attr_name in f[timestamp_key].attrs.keys():
                        original_timestamp_attrs[attr_name] = f[timestamp_key].attrs[attr_name]
        
        write_h5_dataset(
            str(output_path),
            timestamp_key or "timestamps",
            downsampled_timestamps,
            attrs={
                "description": f"Downsampled timestamps (downsample rate: {downsample_rate})",
                "original_frame": int(len(timestamps)),
                "downsampled_frame": int(len(downsampled_timestamps)),
                "downsample_rate": int(downsample_rate),
                "source_key": str(timestamp_key),
                **original_timestamp_attrs,
            },
            overwrite=overwrite,
        )
    
    # Write some global attributes
    with h5py.File(str(output_path), "a") as f:
        f.attrs["source_file"] = str(input_path)
        f.attrs["downsample_rate"] = int(downsample_rate)
        f.attrs["collection_interval_s"] = float(dt * downsample_rate)
        f.attrs["original_collection_interval_s"] = float(dt)
    
    print("✓ processing completed")


def main():
    parser = argparse.ArgumentParser(
        description="Downsampled and processed episode data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file mode:
    python downsample_episode.py episode_26.hdf5
    python downsample_episode.py episode_26.hdf5 --output custom_output.hdf5
    python downsample_episode.py episode_26.hdf5 --downsample-rate 5
  
  batch processing mode:
    python downsample_episode.py --dataset-dir /path/to/data
    python downsample_episode.py --dataset-dir /path/to/data --downsample-rate 5
        """,
    )
    
    parser.add_argument(
        "input_episode",
        nargs="?",
        default=None,
        help="Single file mode: input episode file path (e.g.: episode_26.hdf5)",
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="batch processing mode: data root directory, will process all hdf5 files under {dataset-dir}/hdf5/",
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="episode_*.hdf5",
        help="batch processing mode: episode file matching pattern (default: episode_*.hdf5)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Single file mode: output episode file path (Optional, if not specified then use default naming: downsample_episode_{index}.hdf5)",
    )
    
    parser.add_argument(
        "--downsample-rate",
        type=int,
        default=5,
        help="Downsample rate, i.e. take average every how many frame (default: 5)",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="if output file already exists, overwrite it",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print file list to be processed, don't do any write/output",
    )
    
    parser.add_argument(
        "--max-file",
        type=int,
        default=0,
        help="Maximum number of episodes to process (0 means no limit)",
    )
    
    args = parser.parse_args()
    
    if args.downsample_rate <= 0:
        parser.error("Downsample rate must be greater than 0")
    
    # ========== single file mode / batch processing mode ==========
    if args.input_episode:
        # single file mode
        input_path = Path(args.input_episode)
        
        # if output path not specified, use default naming convention
        if args.output:
            output_path = Path(args.output)
        else:
            # Extract index from input file name, generate default output file name
            episode_index = extract_episode_index(input_path.name)
            output_path = input_path.parent / f"downsample_episode_{episode_index}.hdf5"
        
        if args.dry_run:
            print(f"[dry-run] would process: {input_path}")
            print(f"[dry-run] would write to: {output_path}")
            return
        
        try:
            process_episode(
                input_path=input_path,
                output_path=output_path,
                downsample_rate=args.downsample_rate,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        return
    
    # batch processing mode: must provide dataset-dir
    if not args.dataset_dir:
        parser.error("Please provide single file input_episode, or provide --dataset-dir to enable batch processing mode.")
    
    root_dir = Path(args.dataset_dir) / "hdf5"
    if not root_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root_dir}")
    
    # Search for all hdf5 files directly under hdf5 directory
    all_eps = sorted(root_dir.glob(args.pattern))
    
    if args.max_file and args.max_file > 0:
        all_eps = all_eps[: args.max_file]
    
    print(f"batch processing:root={root_dir}")
    print(f"  episodes={len(all_eps)}, pattern={args.pattern}")
    print(f"  output location: same directory as input file, named downsample_episode_{{index}}.hdf5")
    print(f"  downsample_rate={args.downsample_rate}")
    
    if args.dry_run:
        for p in all_eps[:50]:
            episode_index = extract_episode_index(p.name)
            output_path = p.parent / f"downsample_episode_{episode_index}.hdf5"
            print(f"[dry-run] {p} -> {output_path}")
        if len(all_eps) > 50:
            print(f"[dry-run] ... and {len(all_eps) - 50} more")
        return
    
    num_ok = 0
    num_fail = 0
    for ep_path in all_eps:
        try:
            # output file saved in same directory as input file
            # e.g.: dataset_dir/task_name/batch_name/episode_x.hdf5
            # outputto:dataset_dir/task_name/batch_name/downsample_episode_x.hdf5
            episode_index = extract_episode_index(ep_path.name)
            output_path = ep_path.parent / f"downsample_episode_{episode_index}.hdf5"
            
            print(f"-> {ep_path}  |  output={output_path}")
            process_episode(
                input_path=ep_path,
                output_path=output_path,
                downsample_rate=args.downsample_rate,
                overwrite=args.overwrite,
            )
            num_ok += 1
        except Exception as e:
            num_fail += 1
            error_type = type(e).__name__
            print(f"[FAIL] {ep_path}: [{error_type}] {e}")
    
    print("✓ batch processing completed")
    print(f"  ok={num_ok}, fail={num_fail}, overwrite={args.overwrite}")


if __name__ == "__main__":
    main()
