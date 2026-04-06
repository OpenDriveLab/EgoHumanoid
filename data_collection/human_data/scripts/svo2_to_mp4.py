#!/usr/bin/env python3
"""
SVO2 to MP4 Conversion Tool

Usage:
    python svo2_to_mp4.py <svo2_file_path> [--output <output_file_path>] [--view left|right|side_by_side]

Examples:
    python svo2_to_mp4.py data_collection/body_data/episode_1.svo2
    python svo2_to_mp4.py data_collection/body_data/episode_1.svo2 --output output.mp4
    python svo2_to_mp4.py data_collection/body_data/episode_1.svo2 --view side_by_side
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pyzed.sl as sl


def svo2_to_mp4(svo_path: str, output_path: str = None, view: str = "left"):
    """
    Convert SVO2 file to MP4 format

    Args:
        svo_path: SVO2 file path
        output_path: Output MP4 file path (defaults to same name as input with .mp4 extension)
        view: View type - "left" (left camera), "right" (right camera), "side_by_side" (left and right side by side)
    """
    if not os.path.exists(svo_path):
        print(f"[ERROR] SVO file does not exist: {svo_path}")
        return False

    # Set output path
    if output_path is None:
        output_path = os.path.splitext(svo_path)[0] + ".mp4"
    
    print(f"[INFO] Input file: {svo_path}")
    print(f"[INFO] Output file: {output_path}")
    print(f"[INFO] View mode: {view}")

    # Initialize ZED camera (SVO mode)
    zed = sl.Camera()
    
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False  # Don't use real-time mode, process as fast as possible
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # No depth needed, only extract RGB images
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Cannot open SVO file: {err}")
        return False

    # Get SVO information
    total_frames = zed.get_svo_number_of_frames()
    fps = zed.get_camera_information().camera_configuration.fps
    resolution = zed.get_camera_information().camera_configuration.resolution
    width = resolution.width
    height = resolution.height
    
    print(f"[INFO] Resolution: {width}x{height}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total frames: {total_frames}")

    # Determine output size based on view mode
    if view == "side_by_side":
        output_width = width * 2
        output_height = height
    else:
        output_width = width
        output_height = height

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not video_writer.isOpened():
        print(f"[ERROR] Cannot create video writer")
        zed.close()
        return False

    # Create image containers
    image_left = sl.Mat()
    image_right = sl.Mat()
    
    frame_count = 0

    print(f"\n[INFO] Starting conversion...")

    while True:
        # Read frame
        err = zed.grab()

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print(f"\n[INFO] Reached end of SVO file")
            break
        elif err != sl.ERROR_CODE.SUCCESS:
            print(f"\n[WARN] Failed to read frame: {err}")
            continue

        # Get images
        if view == "left" or view == "side_by_side":
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
        if view == "right" or view == "side_by_side":
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

        # Convert to OpenCV format
        # ZED SDK returns BGRA format, OpenCV VideoWriter needs BGR format
        if view == "left":
            frame = image_left.get_data()[:, :, :3].copy()  # BGRA -> BGR (remove alpha channel)
        elif view == "right":
            frame = image_right.get_data()[:, :, :3].copy()
        else:  # side_by_side
            left_frame = image_left.get_data()[:, :, :3].copy()
            right_frame = image_right.get_data()[:, :, :3].copy()
            frame = np.hstack([left_frame, right_frame])

        # Write frame
        video_writer.write(frame)
        frame_count += 1

        # Show progress
        if frame_count % 100 == 0 or frame_count == total_frames:
            progress = frame_count / total_frames * 100
            print(f"\r[INFO] Progress: {frame_count}/{total_frames} ({progress:.1f}%)", end="", flush=True)

    # Clean up resources
    video_writer.release()
    zed.close()

    print(f"\n\n[INFO] ✓ Conversion complete!")
    print(f"[INFO] Output file: {output_path}")
    print(f"[INFO] Converted frames: {frame_count}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert SVO2 file to MP4 format')
    parser.add_argument('svo_path', type=str, help='SVO2 file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output MP4 file path (defaults to same name as input with .mp4 extension)')
    parser.add_argument('--view', '-v', type=str, default='left',
                        choices=['left', 'right', 'side_by_side'],
                        help='View mode: left (left camera), right (right camera), side_by_side (left and right side by side)')
    
    args = parser.parse_args()
    
    success = svo2_to_mp4(args.svo_path, args.output, args.view)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

