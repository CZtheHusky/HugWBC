#!/usr/bin/env python3
"""
HugWBC 轨迹数据使用示例

这个脚本展示了如何加载和使用收集到的轨迹数据
"""

import numpy as np
import json
import os
from typing import Dict, List


def load_trajectory_data(trajectory_file: str) -> Dict[str, np.ndarray]:
    """
    加载轨迹数据文件
    
    Args:
        trajectory_file: .npz文件路径
        
    Returns:
        包含所有数据的字典
    """
    data = np.load(trajectory_file)
    return {key: data[key] for key in data.keys()}


def load_metadata(metadata_file: str) -> Dict:
    """
    加载轨迹元数据
    
    Args:
        metadata_file: JSON元数据文件路径
        
    Returns:
        元数据字典
    """
    with open(metadata_file, 'r') as f:
        return json.load(f)


def analyze_trajectory(trajectory_file: str, metadata: dict):
    """
    分析单条轨迹数据
    
    Args:
        trajectory_file: 轨迹文件路径
        metadata: 轨迹元数据
    """
    # 显示轨迹索引和ID
    trajectory_index = metadata.get('trajectory_index', metadata.get('trajectory_id', 'Unknown'))
    env_id = metadata.get('env_id', 'Unknown')
    print(f"\n=== 轨迹分析: 索引 {trajectory_index} (环境 {env_id}) ===")
    print(f"类型: {metadata['trajectory_type']}")
    print(f"长度: {metadata['length_s']}秒 ({metadata['length_timesteps']} 时间步)")
    print(f"收集时间: {metadata['collection_time']}")
    
    # 检查视频文件
    if metadata.get('video_file'):
        print(f"视频文件: {metadata['video_file']}")
        if os.path.exists(metadata['video_file']):
            print("✅ 视频文件存在")
        else:
            print("❌ 视频文件不存在")
    else:
        print("视频文件: 无")
    
    # 加载轨迹数据
    try:
        data = np.load(trajectory_file)
        print(f"\n数据形状:")
        print(f"  obs: {data['obs'].shape}")
        print(f"  critic_obs: {data['critic_obs'].shape}")
        print(f"  actions: {data['actions'].shape}")
        print(f"  rewards: {data['rewards'].shape}")
        print(f"  dones: {data['dones'].shape}")
        print(f"  root_states: {data['root_states'].shape}")
        print(f"  commands: {data['commands'].shape}")
        print(f"  dt: {data['dt']}")
        
        # 分析命令信息
        commands = data['commands']
        if len(commands.shape) == 2:
            # 单时间步命令
            command = commands[0]
        else:
            # 多时间步命令
            command = commands[0]
        
        print(f"\n命令信息:")
        if metadata['trajectory_type'] == 'constant':
            print(f"  命令类型: 恒定")
            print(f"  线性速度: vx={command[0]:.3f}, vy={command[1]:.3f}, yaw={command[2]:.3f}")
            print(f"  步态参数: 频率={command[3]:.3f}, 相位={command[4]:.1f}, 摆动高度={command[6]:.3f}")
            print(f"  身体姿态: 高度={command[7]:.3f}, 俯仰={command[8]:.3f}, 侧倾={command[9]:.3f}")
        else:
            print(f"  命令类型: 变化")
            if len(commands.shape) == 2:
                print(f"  命令1: vx={command[0]:.3f}, vy={command[1]:.3f}, yaw={command[2]:.3f}")
                if commands.shape[0] > 1:
                    command2 = commands[1]
                    print(f"  命令2: vx={command2[0]:.3f}, vy={command2[1]:.3f}, yaw={command2[2]:.3f}")
        
        # 统计信息
        print(f"\n统计信息:")
        print(f"  奖励范围: [{data['rewards'].min():.3f}, {data['rewards'].max():.3f}]")
        print(f"  平均奖励: {data['rewards'].mean():.3f}")
        print(f"  动作范围: [{data['actions'].min():.3f}, {data['actions'].max():.3f}]")
        print(f"  动作标准差: {data['actions'].std():.3f}")
        
    except Exception as e:
        print(f"❌ 加载轨迹数据失败: {e}")


def main():
    """主函数"""
    print("加载轨迹元数据...")
    
    # 加载元数据
    metadata_file = "collected_trajectories/trajectory_metadata.json"
    if not os.path.exists(metadata_file):
        print(f"❌ 元数据文件不存在: {metadata_file}")
        return
    
    with open(metadata_file, 'r') as f:
        trajectory_metadata = json.load(f)
    
    print(f"找到 {len(trajectory_metadata)} 条轨迹")
    
    # 按环境ID分组显示
    env_groups = {}
    for trajectory_id, metadata in trajectory_metadata.items():
        env_id = metadata.get('env_id', 0)
        if env_id not in env_groups:
            env_groups[env_id] = []
        env_groups[env_id].append(metadata)
    
    print(f"\n环境分组:")
    for env_id in sorted(env_groups.keys()):
        trajectories = env_groups[env_id]
        constant_count = sum(1 for t in trajectories if t['trajectory_type'] == 'constant')
        changing_count = sum(1 for t in trajectories if t['trajectory_type'] == 'changing')
        print(f"  环境 {env_id}: {len(trajectories)} 条轨迹 ({constant_count} 恒定, {changing_count} 变化)")
    
    # 分析每条轨迹
    for trajectory_id, metadata in trajectory_metadata.items():
        trajectory_file = metadata['file_path']
        if os.path.exists(trajectory_file):
            print(f"\n加载轨迹文件: {trajectory_file}")
            analyze_trajectory(trajectory_file, metadata)
        else:
            print(f"❌ 轨迹文件不存在: {trajectory_file}")
    
    # 总结
    print(f"\n=== 总结 ===")
    constant_count = sum(1 for m in trajectory_metadata.values() if m['trajectory_type'] == 'constant')
    changing_count = sum(1 for m in trajectory_metadata.values() if m['trajectory_type'] == 'changing')
    video_count = sum(1 for m in trajectory_metadata.values() if m.get('video_file') and os.path.exists(m['video_file']))
    
    print(f"恒定命令轨迹: {constant_count} 条")
    print(f"变化命令轨迹: {changing_count} 条")
    print(f"总轨迹数: {len(trajectory_metadata)} 条")
    print(f"视频文件: {video_count} 个")
    
    # 检查视频目录
    videos_dir = "collected_trajectories/videos"
    if os.path.exists(videos_dir):
        video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
        print(f"视频目录: {videos_dir}")
        print(f"视频文件数量: {len(video_files)}")
        if video_files:
            print("视频文件列表:")
            for video_file in sorted(video_files):
                video_path = os.path.join(videos_dir, video_file)
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                print(f"  {video_file} ({file_size:.1f} MB)")
    else:
        print(f"视频目录不存在: {videos_dir}")
    
    print(f"\n=== 数据使用建议 ===")
    print("1. 训练数据增强: 使用这些轨迹来增强训练数据集")
    print("2. 策略评估: 在不同命令下评估策略性能")
    print("3. 行为克隆: 学习人类专家的运动模式")
    print("4. 逆强化学习: 从轨迹中学习奖励函数")
    print("5. 安全验证: 验证策略在各种命令下的安全性")
    print("6. 可视化分析: 通过视频分析机器人运动模式")


if __name__ == "__main__":
    main()
