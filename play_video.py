#!/usr/bin/env python3
"""
轨迹视频播放脚本 - 查看生成的轨迹视频
"""

import os
import cv2
import argparse
from pathlib import Path

def play_video(video_path, speed=1.0):
    """播放视频文件"""
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"🎬 播放视频: {os.path.basename(video_path)}")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        print(f"   时长: {duration:.1f} 秒")
        print(f"   播放速度: {speed}x")
        print()
        print("控制说明:")
        print("  SPACE: 暂停/继续")
        print("  ESC: 退出")
        print("  Q: 退出")
        print("  LEFT/RIGHT: 快退/快进")
        print()
        
        # 计算帧延迟
        frame_delay = int(1000 / (fps * speed))  # 毫秒
        
        paused = False
        current_frame = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("视频播放完毕")
                    break
                
                current_frame += 1
                
                # 在帧上添加信息
                info_frame = frame.copy()
                cv2.putText(info_frame, f"Frame: {current_frame}/{frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_frame, f"Time: {current_frame/fps:.1f}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_frame, f"Speed: {speed}x", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示帧
                cv2.imshow('Trajectory Video', info_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == ord('q') or key == 27:  # Q 或 ESC
                break
            elif key == ord(' '):  # 空格键
                paused = not paused
                print("⏸️  暂停" if paused else "▶️  继续")
            elif key == 81:  # 左箭头
                current_frame = max(0, current_frame - 30)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                print(f"⏪ 快退到帧 {current_frame}")
            elif key == 83:  # 右箭头
                current_frame = min(frame_count - 1, current_frame + 30)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                print(f"⏩ 快进到帧 {current_frame}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ 播放视频失败: {e}")

def list_videos():
    """列出所有可用的视频文件"""
    videos_dir = "collected_trajectories/videos"
    if not os.path.exists(videos_dir):
        print(f"❌ 视频目录不存在: {videos_dir}")
        return
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    if not video_files:
        print("❌ 没有找到视频文件")
        return
    
    print("📹 可用的轨迹视频文件:")
    print("=" * 50)
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(videos_dir, video_file)
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"{i+1:2d}. {video_file}")
            print(f"    大小: {file_size:.1f} MB")
            print(f"    时长: {duration:.1f} 秒")
            print(f"    帧数: {frame_count}")
            print()
    
    print("使用方法:")
    print(f"  python {__file__} --video <视频文件名>")
    print(f"  例如: python {__file__} --video 000000.mp4")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="播放轨迹视频文件")
    parser.add_argument("--video", type=str, help="要播放的视频文件名")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍数")
    parser.add_argument("--list", action="store_true", help="列出所有可用的视频文件")
    
    args = parser.parse_args()
    
    if args.list:
        list_videos()
        return
    
    if not args.video:
        print("请指定要播放的视频文件")
        print(f"使用方法: python {__file__} --video <视频文件名>")
        print(f"或使用: python {__file__} --list 查看所有视频")
        return
    
    # 构建完整的视频路径
    video_path = os.path.join("collected_trajectories/videos", args.video)
    
    # 播放视频
    play_video(video_path, args.speed)

if __name__ == "__main__":
    main()
