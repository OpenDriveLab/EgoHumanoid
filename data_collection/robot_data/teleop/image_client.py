#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MJPEG视频流客户端
从jetson_get_image.py服务端获取视频流并显示
"""

import cv2
import argparse
import sys


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MJPEG视频流客户端')
    parser.add_argument('--host', default='192.168.1.100', help='服务端IP地址 (默认: 192.168.1.100)')
    parser.add_argument('--port', type=int, default=8080, help='服务端端口 (默认: 8080)')
    parser.add_argument('--window-name', default='Video Stream', help='窗口名称')
    
    args = parser.parse_args()
    
    # 构建视频流URL
    stream_url = f"http://{args.host}:{args.port}/video_feed"
    
    print(f"{'='*50}")
    print("MJPEG视频流客户端")
    print(f"{'='*50}")
    print(f"连接地址: {stream_url}")
    print("按 'q' 或 ESC 退出")
    print(f"{'='*50}\n")
    
    # 使用OpenCV打开MJPEG流
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"错误: 无法连接到视频流 {stream_url}")
        print("请检查:")
        print("  1. 服务端是否已启动")
        print("  2. IP地址和端口是否正确")
        print("  3. 网络连接是否正常")
        sys.exit(1)
    
    print("✓ 成功连接到视频流")
    
    # 创建窗口
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("警告: 无法读取帧，尝试重新连接...")
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue
            
            # 显示帧
            cv2.imshow(args.window_name, frame)
            breakpoint()
            # 检测按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                print("\n用户退出")
                break
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("资源已清理")


if __name__ == '__main__':
    main()

