#!/usr/bin/env python3
"""
视频测试脚本 - 验证生成的轨迹视频文件
"""

import os
import cv2
import numpy as np
from pathlib import Path

def test_video_file(video_path):
    """测试单个视频文件"""
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    try:
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
        
        print(f"✅ 视频文件: {os.path.basename(video_path)}")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        print(f"   总帧数: {frame_count}")
        print(f"   时长: {duration:.1f} 秒")
        
        # 读取第一帧和最后一帧
        ret, first_frame = cap.read()
        if ret:
            print(f"   第一帧形状: {first_frame.shape}")
            print(f"   第一帧数据类型: {first_frame.dtype}")
            print(f"   第一帧值范围: [{first_frame.min()}, {first_frame.max()}]")
        
        # 跳转到最后一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        if ret:
            print(f"   最后一帧形状: {last_frame.shape}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ 测试视频文件失败: {video_path}")
        print(f"   错误: {e}")
        return False

def main():
    """主函数"""
    print("🎬 轨迹视频文件测试")
    print("=" * 50)
    
    videos_dir = "collected_trajectories/videos"
    if not os.path.exists(videos_dir):
        print(f"❌ 视频目录不存在: {videos_dir}")
        return
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    if not video_files:
        print("❌ 没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    print()
    
    # 测试每个视频文件
    success_count = 0
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        if test_video_file(video_path):
            success_count += 1
        print()
    
    # 总结
    print("=" * 50)
    print(f"测试完成: {success_count}/{len(video_files)} 个视频文件正常")
    
    if success_count == len(video_files):
        print("🎉 所有视频文件都可以正常播放！")
    else:
        print("⚠️  部分视频文件存在问题，请检查")

if __name__ == "__main__":
    main()

