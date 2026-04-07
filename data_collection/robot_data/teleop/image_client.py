#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MJPEG video stream client
Get video stream from jetson_get_image.py server and display
"""

import cv2
import argparse
import sys


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MJPEG video stream client')
    parser.add_argument('--host', default='192.168.1.100', help='Server IP address (default: 192.168.1.100)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--window-name', default='Video Stream', help='Window name')

    args = parser.parse_args()

    # Build video stream URL
    stream_url = f"http://{args.host}:{args.port}/video_feed"

    print(f"{'='*50}")
    print("MJPEG video stream client")
    print(f"{'='*50}")
    print(f"Connection address: {stream_url}")
    print("Press 'q' or ESC to exit")
    print(f"{'='*50}\n")

    # Use OpenCV to open MJPEG stream
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Unable to connect to video stream {stream_url}")
        print("Please check:")
        print("  1. Server is running")
        print("  2. IP address and port are correct")
        print("  3. Network connection is normal")
        sys.exit(1)

    print("Successfully connected to video stream")

    # Create window
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Warning: Unable to read frame, attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue

            # Display frame
            cv2.imshow(args.window_name, frame)
            breakpoint()
            # Detect key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nUser exited")
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up")


if __name__ == '__main__':
    main()

