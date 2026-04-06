#!/usr/bin/env python3
"""
Export left camera images from HDF5 file to video
- Frame rate: 20Hz
- Display frame count info in video: {current frame}/{total frame}
- Support multiprocess parallel processing
"""

import os
import glob
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import processPoolExecutor, as_completed
import multiprocessing


def export_episode_to_video(args):
    """
    Export left camera images from single episode to video
    
    Args:
        args: (hdf5_path, output_path, fps) tuple
    Returns:
        (success, message) tuple
    """
    hdf5_path, output_path, fps = args
    return _export_episode_to_video_impl(hdf5_path, output_path, fps)


def _export_episode_to_video_impl(hdf5_path, output_path, fps=20):
    """
    Export left camera images from single episode to video
    
    Args:
        hdf5_path: HDF5filepath
        output_path: output video path
        fps: Frame rate, default 20Hz
    Returns:
        (success, message) tuple
    """
    try:
        basename = os.path.basename(hdf5_path)
        with h5py.File(hdf5_path, 'r') as f:
            img_data = f['observation_image_left']
            num_frame = len(img_data)
            
            # First decode first frame to get image dimensions
            first_frame = cv2.imdecode(np.frombuffer(img_data[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width = first_frame.shape[:2]
            
            # Compress to half resolution
            new_width = width // 2
            new_height = height // 2
            
            # use mp4v encoder
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
            
            for i in range(num_frame):
                # Decode image
                img_array = np.frombuffer(img_data[i], dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Scale to half resolution
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Add frame count text
                text = f"{i + 1}/{num_frame}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5  # Reduce font size for half resolution
                thickness = 1
                color = (255, 255, 255)  # White
                outline_color = (0, 0, 0)  # Black outline
                
                # Place in upper left corner, leave margin
                x = 10
                y = 20
                
                # First draw black outline (increase readability)
                cv2.putText(frame, text, (x, y), font, font_scale, outline_color, thickness + 2, cv2.LINE_AA)
                # Then draw white text
                cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                
                out.write(frame)
            
            out.release()
            return (True, f"Exported: {basename} ({num_frame} frame)")
    except Exception as e:
        return (False, f"Error processing {os.path.basename(hdf5_path)}: {e}")


def main():
    # Set path
    input_dir = "/cpfs01/shared/shimodi/data/human/source/garbage/version_6/final/"
    output_dir = "human_garbage_version_6_videos/"
    # cp -r human_garbage_version_6_videos /mnt/data/pengshijia/data/rss_data/videos/
    # Set parallel process count (default use half of CPU cores, avoid using all resources)
    # num_workers = max(1, multiprocessing.cpu_count() // 2)
    num_workers = multiprocessing.cpu_count()-1
    
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"output directory: {output_dir}")
    print(f"use {num_workers} parallel workers")
    
    # Get all HDF5 files
    hdf5_file = sorted(glob.glob(os.path.join(input_dir, "episode_*.hdf5")))
    print(f"Found {len(hdf5_file)} episode file")
    
    # Prepare task list
    tasks = []
    for hdf5_path in hdf5_file:
        basename = os.path.basename(hdf5_path)
        episode_idx = basename.replace("episode_", "").replace(".hdf5", "")
        output_path = os.path.join(output_dir, f"episode_{episode_idx}.mp4")
        
        # if video already exists, skip
        if os.path.exists(output_path):
            print(f"Skipping {basename} (video already exists)")
            continue
        
        tasks.append((hdf5_path, output_path, 20))
    
    if not tasks:
        print("No new videos to export.")
        return
    
    print(f"processing {len(tasks)} videos...")
    
    # use multiprocess parallel processing
    with processPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(export_episode_to_video, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Exporting videos") as pbar:
            for future in as_completed(futures):
                success, message = future.result()
                if success:
                    tqdm.write(message)
                else:
                    tqdm.write(f"[ERROR] {message}")
                pbar.update(1)
    
    print("Done!")


if __name__ == "__main__":
    main()
