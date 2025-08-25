#!/usr/bin/env python3
"""
视频可视化脚本 - 提供轨迹视频的统计分析和内容预览
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_video_summary(video_path, output_dir):
    """创建视频摘要报告"""
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
        
        print(f"🎬 创建视频摘要: {os.path.basename(video_path)}")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        print(f"   总帧数: {frame_count}")
        print(f"   时长: {duration:.1f} 秒")
        print(f"   输出目录: {output_dir}")
        print()
        
        # 分析视频内容
        print("📊 分析视频内容...")
        
        # 提取关键帧
        key_frames = [0, frame_count//4, frame_count//2, 3*frame_count//4, frame_count-1]
        frames_data = []
        
        for frame_idx in key_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 计算帧的统计信息
                mean_color = np.mean(frame, axis=(0, 1))
                std_color = np.std(frame, axis=(0, 1))
                
                # 计算亮度
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                frames_data.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'mean_color': mean_color,
                    'std_color': std_color,
                    'brightness': brightness,
                    'frame': frame
                })
        
        # 创建可视化图表
        print("📈 创建可视化图表...")
        
        # 1. 颜色变化图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'视频分析: {os.path.basename(video_path)}', fontsize=16)
        
        # 颜色变化趋势
        timestamps = [f['timestamp'] for f in frames_data]
        b_values = [f['mean_color'][0] for f in frames_data]
        g_values = [f['mean_color'][1] for f in frames_data]
        r_values = [f['mean_color'][2] for f in frames_data]
        
        axes[0, 0].plot(timestamps, b_values, 'b-', label='Blue', linewidth=2)
        axes[0, 0].plot(timestamps, g_values, 'g-', label='Green', linewidth=2)
        axes[0, 0].plot(timestamps, r_values, 'r-', label='Red', linewidth=2)
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel('平均颜色值')
        axes[0, 0].set_title('颜色变化趋势')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 亮度变化
        brightness_values = [f['brightness'] for f in frames_data]
        axes[0, 1].plot(timestamps, brightness_values, 'k-', linewidth=2)
        axes[0, 1].set_xlabel('时间 (秒)')
        axes[0, 1].set_ylabel('平均亮度')
        axes[0, 1].set_title('亮度变化趋势')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 颜色标准差
        b_std = [f['std_color'][0] for f in frames_data]
        g_std = [f['std_color'][1] for f in frames_data]
        r_std = [f['std_color'][2] for f in frames_data]
        
        axes[1, 0].plot(timestamps, b_std, 'b-', label='Blue', linewidth=2)
        axes[1, 0].plot(timestamps, g_std, 'g-', label='Green', linewidth=2)
        axes[1, 0].plot(timestamps, r_std, 'r-', label='Red', linewidth=2)
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel('颜色标准差')
        axes[1, 0].set_title('颜色变化程度')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 关键帧预览
        if len(frames_data) >= 3:
            # 显示前3个关键帧
            for i in range(min(3, len(frames_data))):
                frame = frames_data[i]
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame['frame'], cv2.COLOR_BGR2RGB)
                axes[1, 1].imshow(frame_rgb)
                axes[1, 1].set_title(f'关键帧 {i+1} (t={frame["timestamp"]:.1f}s)')
                axes[1, 1].axis('off')
                break
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(output_dir, 'video_analysis.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 保存分析图表: {chart_path}")
        
        # 保存关键帧
        for i, frame_data in enumerate(frames_data):
            frame_filename = f"keyframe_{i:02d}_t{frame_data['timestamp']:.1f}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame_data['frame'])
            print(f"  ✅ 保存关键帧 {i+1}/{len(frames_data)}: {frame_filename}")
        
        # 创建摘要报告
        summary = {
            'video_file': os.path.basename(video_path),
            'resolution': f"{width}x{height}",
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration,
            'key_frames': []
        }
        
        for frame_data in frames_data:
            summary['key_frames'].append({
                'frame_index': frame_data['frame_idx'],
                'timestamp': frame_data['timestamp'],
                'mean_color_bgr': frame_data['mean_color'].tolist(),
                'brightness': frame_data['brightness']
            })
        
        # 保存JSON摘要
        summary_path = os.path.join(output_dir, 'video_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✅ 保存摘要报告: {summary_path}")
        
        cap.release()
        plt.close()
        
        print(f"\n🎉 视频摘要创建完成")
        print(f"   输出目录: {output_dir}")
        print(f"   分析图表: video_analysis.png")
        print(f"   关键帧: {len(frames_data)} 张")
        print(f"   摘要报告: video_summary.json")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建视频摘要失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="创建轨迹视频的可视化摘要")
    parser.add_argument("--video", type=str, help="要分析的视频文件名")
    parser.add_argument("--output", type=str, default="video_summaries", help="输出目录")
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
            print(f"  python {__file__} --video <视频文件名>")
            print(f"  例如: python {__file__} --video 000000.mp4")
        return
    
    if not args.video:
        print("请指定要分析的视频文件")
        print(f"使用方法: python {__file__} --video <视频文件名>")
        print(f"或使用: python {__file__} --list 查看所有视频")
        return
    
    # 构建完整的视频路径
    video_path = os.path.join("collected_trajectories/videos", args.video)
    
    # 创建视频摘要
    output_dir = os.path.join(args.output, os.path.splitext(args.video)[0])
    create_video_summary(video_path, output_dir)

if __name__ == "__main__":
    main()

