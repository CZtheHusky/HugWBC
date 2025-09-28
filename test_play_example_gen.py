#!/usr/bin/env python3
"""
小规模测试 play_example_gen.py 的修改
"""

import os
import sys
import subprocess

def test_play_example_gen():
    """测试修改后的 play_example_gen.py"""
    
    print("=" * 60)
    print("测试 play_example_gen.py 的 latent state 收集功能")
    print("=" * 60)
    
    # 切换到项目目录
    os.chdir('/root/workspace/HugWBC')
    
    # 运行小规模测试（只收集2个命令，每个命令2条轨迹）
    cmd = [
        'python', 'legged_gym/scripts/play_example_gen.py',
        '--task', 'h1int',
        '--save_latent',
        '--headless'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print("开始测试...")
    
    try:
        # 运行命令
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("=" * 60)
        print("命令输出:")
        print("=" * 60)
        print(result.stdout)
        
        if result.stderr:
            print("=" * 60)
            print("错误输出:")
            print("=" * 60)
            print(result.stderr)
        
        print("=" * 60)
        print(f"返回码: {result.returncode}")
        print("=" * 60)
        
        # 检查输出目录
        output_dir = "example_trajectories_test"
        if os.path.exists(output_dir):
            print(f"输出目录 {output_dir} 存在")
            files = os.listdir(output_dir)
            print(f"生成的文件: {files}")
            
            # 检查是否有 .zarr 文件
            zarr_files = [f for f in files if f.endswith('.zarr')]
            print(f"Zarr 文件数量: {len(zarr_files)}")
            
            if zarr_files:
                print("✓ 成功生成数据文件")
                return True
            else:
                print("✗ 未找到数据文件")
                return False
        else:
            print(f"✗ 输出目录 {output_dir} 不存在")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 命令执行超时")
        return False
    except Exception as e:
        print(f"✗ 执行出错: {e}")
        return False

if __name__ == "__main__":
    success = test_play_example_gen()
    if success:
        print("\n✓ 测试通过！")
    else:
        print("\n✗ 测试失败！")
        sys.exit(1)
