#!/usr/bin/env python3
"""
视频帧提取脚本 - 从轨迹视频中提取关键帧并保存为图像
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, num_frames=10):
    """从视频中提取指定数量的帧"""
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return False
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"🎬 提取视频帧: {os.path.basename(video_path)}")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        print(f"   总帧数: {frame_count}")
        print(f"   时长: {duration:.1f} 秒")
        print(f"   提取帧数: {num_frames}")
        print(f"   输出目录: {output_dir}")
        print()
        
        # 计算要提取的帧的索引
        if num_frames >= frame_count:
            frame_indices = list(range(frame_count))
        else:
            # 均匀分布提取帧
            frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
        
        extracted_frames = []
        
        for i, frame_idx in enumerate(frame_indices):
            # 设置视频位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # 读取帧
            ret, frame = cap.read()
            if ret:
                # 计算时间戳
                timestamp = frame_idx / fps
                
                # 在帧上添加信息
                info_frame = frame.copy()
                cv2.putText(info_frame, f"Frame: {frame_idx}/{frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_frame, f"Time: {timestamp:.1f}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_frame, f"FPS: {fps:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 保存帧
                frame_filename = f"frame_{frame_idx:04d}_t{timestamp:.1f}s.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, info_frame)
                
                extracted_frames.append({
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'filename': frame_filename,
                    'path': frame_path
                })
                
                print(f"  ✅ 提取帧 {i+1}/{len(frame_indices)}: {frame_filename}")
        
        cap.release()
        
        print(f"\n🎉 成功提取 {len(extracted_frames)} 帧")
        print(f"   输出目录: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 提取视频帧失败: {e}")
        return False

def analyze_video_content(video_path):
    """分析视频内容，检测是否有变化"""
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"🔍 分析视频内容: {os.path.basename(video_path)}")
        print(f"   总帧数: {frame_count}")
        print(f"   帧率: {fps:.1f} FPS")
        print()
        
        # 提取几个关键帧进行分析
        sample_frames = [0, frame_count//4, frame_count//2, 3*frame_count//4, frame_count-1]
        
        frames_data = []
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 计算帧的统计信息
                mean_color = np.mean(frame, axis=(0, 1))
                std_color = np.std(frame, axis=(0, 1))
                
                frames_data.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'mean_color': mean_color,
                    'std_color': std_color
                })
                
                print(f"  帧 {frame_idx:4d} (t={frame_idx/fps:.1f}s):")
                print(f"    平均颜色: B={mean_color[0]:.1f}, G={mean_color[1]:.1f}, R={mean_color[2]:.1f}")
                print(f"    颜色标准差: B={std_color[0]:.1f}, G={std_color[1]:.1f}, R={std_color[2]:.1f}")
        
        cap.release()
        
        # 分析帧间差异
        if len(frames_data) > 1:
            print(f"\n📊 帧间差异分析:")
            for i in range(1, len(frames_data)):
                prev_frame = frames_data[i-1]
                curr_frame = frames_data[i]
                
                color_diff = np.abs(curr_frame['mean_color'] - prev_frame['mean_color'])
                total_diff = np.sum(color_diff)
                
                print(f"  帧 {prev_frame['frame_idx']} -> {curr_frame['frame_idx']}:")
                print(f"    颜色差异: B={color_diff[0]:.1f}, G={color_diff[1]:.1f}, R={color_diff[2]:.1f}")
                print(f"    总差异: {total_diff:.1f}")
                
                if total_diff > 50:
                    print(f"    🔴 显著变化")
                elif total_diff > 20:
                    print(f"    🟡 中等变化")
                else:
                    print(f"    🟢 轻微变化")
        
    except Exception as e:
        print(f"❌ 分析视频内容失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从轨迹视频中提取帧")
    parser.add_argument("--video", type=str, help="要处理的视频文件名")
    parser.add_argument("--output", type=str, default="extracted_frames", help="输出目录")
    parser.add_argument("--frames", type=int, default=10, help="要提取的帧数")
    parser.add_argument("--analyze", action="store_true", help="分析视频内容")
    parser.add_argument("--list", action="store_true", help="列出所有可用的视频文件")
    
    args = parser.parse_args()
    
    if args.list:
        videos_dir = "collected_trajectories/videos"
        if os.path.exists(videos_dir):
            video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
            video_files.sort()
            
            print("📹 可用的轨迹视频文件:")
            print("=" * 50)
            
            for i, video_file in enumerate(video_files):
                video_path = os.path.join(videos_dir, video_file)
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                
                print(f"{i+1:2d}. {video_file} ({file_size:.1f} MB)")
            
            print(f"\n使用方法:")
            print(f"  python {__file__} --video <视频文件名> --frames <帧数>")
            print(f"  例如: python {__file__} --video 000000.mp4 --frames 20")
        return
    
    if not args.video:
        print("请指定要处理的视频文件")
        print(f"使用方法: python {__file__} --video <视频文件名>")
        print(f"或使用: python {__file__} --list 查看所有视频")
        return
    
    # 构建完整的视频路径
    video_path = os.path.join("collected_trajectories/videos", args.video)
    
    if args.analyze:
        # 分析视频内容
        analyze_video_content(video_path)
    else:
        # 提取帧
        output_dir = os.path.join(args.output, os.path.splitext(args.video)[0])
        extract_frames(video_path, output_dir, args.frames)

if __name__ == "__main__":
    main()

