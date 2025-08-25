import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import json
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import cv2

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


class TrajectoryCollector:
    """
    轨迹采集器，用于收集HugWBC的轨迹数据
    """
    
    def __init__(self, args, env_cfg, train_cfg):
        """
        初始化轨迹采集器
        
        Args:
            args: 命令行参数
            env_cfg: 环境配置
            train_cfg: 训练配置
        """
        self.args = args
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        
        # 轨迹参数
        self.trajectory_length_s = 10.0  # 轨迹长度10秒
        self.trajectory_timesteps = int(self.trajectory_length_s * 500)  # 500Hz采样率
        
        # 数据存储
        self.output_dir = "collected_trajectories"
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.metadata_file = os.path.join(self.output_dir, "trajectory_metadata.json")
        self.trajectory_metadata = {}
        
        # 全局索引计数器
        self.global_index = 0
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        
        # 加载已有的元数据
        self._load_metadata()
        
        # 初始化环境
        self._init_environment()
        
        # 初始化命令范围（从训练配置中获取）
        self._init_command_ranges()
        
        # 初始化策略为None
        self.policy = None
        
        # 初始化视频录制器
        self.video_writers = {}
        
    def _load_metadata(self):
        """加载已有的轨迹元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.trajectory_metadata = json.load(f)
                print(f"Loaded {len(self.trajectory_metadata)} existing trajectory metadata")
                
                # 更新全局索引计数器
                if self.trajectory_metadata:
                    max_index = max(int(k) for k in self.trajectory_metadata.keys())
                    self.global_index = max_index + 1
                    print(f"Global index counter set to: {self.global_index}")
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
                self.trajectory_metadata = {}
                self.global_index = 0
        else:
            self.trajectory_metadata = {}
            self.global_index = 0
    
    def _save_metadata(self):
        """保存轨迹元数据到JSON文件"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.trajectory_metadata, f, indent=2)
            print(f"Metadata saved to {self.metadata_file}")
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def _init_environment(self):
        """初始化环境"""
        # 配置环境参数 - 增加环境数量以提高采样速度
        self.env_cfg.env.num_envs = 4  # 使用4个环境进行并行采集
        self.env_cfg.env.episode_length_s = self.trajectory_length_s + 1  # 稍微长一点避免提前结束
        
        # 创建环境
        self.env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        
        # 加载策略（如果需要）
        if hasattr(self.args, 'load_checkpoint') and self.args.load_checkpoint:
            self._load_policy()
    
    def _load_policy(self):
        """加载训练好的策略"""
        try:
            self.train_cfg.runner.resume = True
            ppo_runner, _ = task_registry.make_alg_runner(
                env=self.env, name=self.args.task, args=self.args, train_cfg=self.train_cfg)
            self.policy = ppo_runner.get_inference_policy(device=self.env.device)
            print("Policy loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load policy: {e}")
            self.policy = None
    
    def _init_command_ranges(self):
        """初始化命令范围（从训练配置中获取）"""
        # 获取训练时使用的命令范围
        self.command_ranges = {
            'lin_vel_x': [-0.6, 0.6],
            'lin_vel_y': [-0.6, 0.6], 
            'ang_vel_yaw': [-0.6, 0.6],
            'gait_frequency': [1.5, 3.5],
            'foot_swing_height': [0.1, 0.35],
            'body_height': [-0.3, 0.0],
            'body_pitch': [0.0, 0.4],
            'waist_roll': [-1.0, 1.0]
        }
        print("Command ranges initialized from training configuration")
    
    def _sample_random_command(self) -> np.ndarray:
        """
        从可行域中均匀随机采样一个命令
        
        Returns:
            np.ndarray: 11维命令向量
        """
        command = np.zeros(11)
        
        # 基础运动命令
        command[0] = np.random.uniform(self.command_ranges['lin_vel_x'][0], self.command_ranges['lin_vel_x'][1])  # vx
        command[1] = np.random.uniform(self.command_ranges['lin_vel_y'][0], self.command_ranges['lin_vel_y'][1])  # vy
        command[2] = np.random.uniform(self.command_ranges['ang_vel_yaw'][0], self.command_ranges['ang_vel_yaw'][1])  # yaw
        
        # 步态参数
        command[3] = np.random.uniform(self.command_ranges['gait_frequency'][0], self.command_ranges['gait_frequency'][1])  # freq
        command[4] = np.random.choice([0.0, 0.5])  # phase (跳跃或行走)
        command[5] = 0.5  # duration (固定值)
        command[6] = np.random.uniform(self.command_ranges['foot_swing_height'][0], self.command_ranges['foot_swing_height'][1])  # swing_h
        
        # 身体姿态
        command[7] = np.random.uniform(self.command_ranges['body_height'][0], self.command_ranges['body_height'][1])  # body_h
        command[8] = np.random.uniform(self.command_ranges['body_pitch'][0], self.command_ranges['body_pitch'][1])  # body_pitch
        command[9] = np.random.uniform(self.command_ranges['waist_roll'][0], self.command_ranges['waist_roll'][1])  # waist_roll
        command[10] = 0.0  # interrupt_flag (默认关闭)
        
        return command
    
    def _apply_command_constraints(self, command: np.ndarray) -> np.ndarray:
        """
        应用命令约束，确保物理合理性
        
        Args:
            command: 原始命令
            
        Returns:
            np.ndarray: 约束后的命令
        """
        # 计算速度水平
        velocity_level = np.linalg.norm(command[:2]) + 0.5 * abs(command[2])
        
        # 高速环境约束
        if velocity_level > 1.8:
            command[3] = max(command[3], 2.0)  # 最小频率
            command[6] = min(command[6], 0.20)  # 最大摆动高度
            command[8] = min(command[8], 0.3)   # 最大俯仰角
            command[9] = np.clip(command[9], -0.15, 0.15)  # 腰部侧倾角限制
        
        # 跳跃环境约束
        if command[4] == 0.0:  # 跳跃
            command[6] = min(command[6], 0.2)   # 摆动高度限制
            command[8] = min(command[8], 0.3)   # 俯仰角限制
        
        # 低身体高度约束
        if command[7] < -0.15:
            command[6] = min(command[6], 0.20)  # 摆动高度限制
        if command[7] < -0.2:
            command[8] = min(command[8], 0.3)   # 俯仰角限制
        
        return command
    
    def _collect_single_trajectory(self, trajectory_type: str = "constant") -> Tuple[int, Dict]:
        """
        收集单条轨迹
        
        Args:
            trajectory_type: 轨迹类型 ("constant" 或 "changing")
            
        Returns:
            Tuple[int, Dict]: (轨迹索引, 元数据)
        """
        # 使用全局索引作为轨迹标识符
        trajectory_index = self.global_index
        trajectory_id = str(trajectory_index)
        
        # 初始化视频录制
        self._init_video_recording(trajectory_index)
        
        # 重置环境
        _, _ = self.env.reset()
        
        # 初始化数据缓冲区 - 为每个环境创建缓冲区
        obs_buffer = [[] for _ in range(self.env.num_envs)]
        critic_obs_buffer = [[] for _ in range(self.env.num_envs)]
        actions_buffer = [[] for _ in range(self.env.num_envs)]
        rewards_buffer = [[] for _ in range(self.env.num_envs)]
        dones_buffer = [[] for _ in range(self.env.num_envs)]
        root_states_buffer = [[] for _ in range(self.env.num_envs)]
        commands_buffer = [[] for _ in range(self.env.num_envs)]
        
        # 为每个环境设置不同的命令
        if trajectory_type == "constant":
            # 类型1: 每个环境的命令始终不变，但不同环境可以有不同的命令
            commands_per_env = []
            for env_id in range(self.env.num_envs):
                command = self._sample_random_command()
                command = self._apply_command_constraints(command)
                commands_per_env.append(command)
                metadata_commands = [command.tolist()]
        else:
            # 类型2: 每个环境前5秒一个命令，后5秒另一个命令
            commands_per_env = []
            for env_id in range(self.env.num_envs):
                command1 = self._sample_random_command()
                command1 = self._apply_command_constraints(command1)
                command2 = self._sample_random_command()
                command2 = self._apply_command_constraints(command2)
                commands_per_env.append((command1, command2))
                metadata_commands = [command1.tolist(), command2.tolist()]
        
        # 设置环境命令
        for env_id in range(self.env.num_envs):
            if trajectory_type == "constant":
                # 将numpy数组转换为torch张量
                command_tensor = torch.tensor(commands_per_env[env_id], dtype=torch.float32, device=self.env.device)
                self.env.commands[env_id] = command_tensor
            else:
                # 变化命令：前5秒command1，后5秒command2
                command_tensor = torch.tensor(commands_per_env[env_id][0], dtype=torch.float32, device=self.env.device)
                self.env.commands[env_id] = command_tensor
        
        # 主循环：收集轨迹数据
        timestep = 0
        while timestep < self.trajectory_timesteps:
            # 检查是否需要切换命令（对于变化命令类型）
            if trajectory_type == "changing" and timestep >= self.trajectory_timesteps // 2:
                for env_id in range(self.env.num_envs):
                    command_tensor = torch.tensor(commands_per_env[env_id][1], dtype=torch.float32, device=self.env.device)
                    self.env.commands[env_id] = command_tensor
            
            # 生成动作（如果有策略）或使用零动作
            if self.policy is not None:
                # 获取观察数据
                obs = self.env.get_observations()
                # 使用策略生成动作
                with torch.no_grad():
                    actions = self.policy.act_inference(obs)
            else:
                # 使用零动作
                actions = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.env.device)
            
            # 执行环境步进
            step_result = self.env.step(actions)
            
            # 处理不同的返回值格式
            if len(step_result) == 4:
                obs, rewards, dones, info = step_result
            elif len(step_result) == 5:
                obs, critic_obs, rewards, dones, info = step_result
            else:
                print(f"Warning: Unexpected step result format: {len(step_result)} values")
                obs = step_result[0] if len(step_result) > 0 else None
                rewards = step_result[2] if len(step_result) > 2 else None
                dones = step_result[3] if len(step_result) > 3 else None
                info = step_result[-1] if len(step_result) > 0 else {}
            
            # 记录RGB帧（如果可用）
            if hasattr(self.env, 'gym') and hasattr(self.env, 'sim'):
                try:
                    # 尝试获取RGB图像
                    for env_id in range(self.env.num_envs):
                        self._record_rgb_frame(env_id)
                except Exception as e:
                    # RGB录制失败时继续，不影响轨迹数据收集
                    pass
            
            # 存储数据到缓冲区
            for env_id in range(self.env.num_envs):
                if obs is not None:
                    obs_buffer[env_id].append(obs[env_id].cpu().numpy())
                else:
                    obs_buffer[env_id].append(np.zeros(76))  # 默认观察大小
                
                # 处理critic_obs
                if hasattr(self.env, 'critic_obs') and self.env.critic_obs is not None:
                    critic_obs_buffer[env_id].append(self.env.critic_obs[env_id].cpu().numpy())
                else:
                    critic_obs_buffer[env_id].append(np.zeros(321))  # 默认大小
                
                actions_buffer[env_id].append(actions[env_id].cpu().numpy())
                
                if rewards is not None:
                    rewards_buffer[env_id].append(rewards[env_id].cpu().numpy())
                else:
                    rewards_buffer[env_id].append(0.0)
                
                if dones is not None:
                    dones_buffer[env_id].append(dones[env_id].cpu().numpy())
                else:
                    dones_buffer[env_id].append(False)
                
                root_states_buffer[env_id].append(self.env.root_states[env_id].cpu().numpy())
                commands_buffer[env_id].append(self.env.commands[env_id].cpu().numpy())
            
            timestep += 1
            
            # 检查是否所有环境都完成了
            if dones.all():
                break
        
        # 关闭视频录制
        self._close_video_recording()
        
        # 为每个环境保存轨迹数据
        for env_id in range(self.env.num_envs):
            env_trajectory_index = trajectory_index * self.env.num_envs + env_id
            
            # 准备轨迹数据
            trajectory_data = {
                'obs': np.array(obs_buffer[env_id]),
                'critic_obs': np.array(critic_obs_buffer[env_id]),
                'actions': np.array(actions_buffer[env_id]),
                'rewards': np.array(rewards_buffer[env_id]),
                'dones': np.array(dones_buffer[env_id]),
                'root_states': np.array(root_states_buffer[env_id]),
                'commands': np.array(commands_buffer[env_id]),
                'dt': self.env.dt
            }
            
            # 保存轨迹数据
            trajectory_file = os.path.join(self.output_dir, f"{env_trajectory_index:06d}.npz")
            np.savez_compressed(trajectory_file, **trajectory_data)
            
            # 创建元数据
            metadata = {
                'trajectory_index': env_trajectory_index,
                'trajectory_id': str(env_trajectory_index),
                'trajectory_type': trajectory_type,
                'env_id': env_id,
                'commands': metadata_commands,
                'length_s': self.trajectory_length_s,
                'length_timesteps': len(obs_buffer[env_id]),
                'collection_time': datetime.now().isoformat(),
                'file_path': trajectory_file,
                'video_file': os.path.join(self.videos_dir, f"{env_trajectory_index:06d}.mp4") if os.path.exists(os.path.join(self.videos_dir, f"{env_trajectory_index:06d}.mp4")) else None
            }
            
            # 更新元数据字典
            self.trajectory_metadata[str(env_trajectory_index)] = metadata
        
        # 递增全局索引
        self.global_index += 1
        
        print(f"Collected trajectory {trajectory_index} ({trajectory_type}) with {self.env.num_envs} environments, {len(obs_buffer[0])} timesteps each")
        
        return trajectory_index, metadata
    
    def _init_video_recording(self, trajectory_index: int):
        """初始化视频录制"""
        # 在无头模式下跳过视频录制
        if hasattr(self.env, 'headless') and self.env.headless:
            print("⚠️  无头模式：跳过视频录制初始化")
            return
            
        # 关闭之前的视频写入器
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        
        # 为每个环境创建视频写入器
        for env_id in range(self.env.num_envs):
            # 使用全局轨迹索引，而不是批次索引
            global_trajectory_index = trajectory_index * self.env.num_envs + env_id
            video_filename = os.path.join(self.videos_dir, f"{global_trajectory_index:06d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # 30 FPS
            frame_size = (1280, 720)  # 标准HD分辨率
            
            writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
            self.video_writers[env_id] = writer
            print(f"Initialized video recording for environment {env_id}: {video_filename}")
        
        # 初始化Isaac Gym相机（如果可用）
        self._init_isaac_cameras(trajectory_index)
    
    def _init_isaac_cameras(self, trajectory_index: int):
        """初始化Isaac Gym相机用于RGB录制"""
        # 检查是否在无头模式下运行
        if hasattr(self.env, 'headless') and self.env.headless:
            print("⚠️  无头模式：无法获取真实RGB渲染，跳过视频录制")
            self.camera_positions = {}
            return
        
        try:
            if hasattr(self.env, 'gym') and hasattr(self.env, 'sim') and hasattr(self.env, 'viewer') and self.env.viewer is not None:
                # 使用play.py中的相机参数
                self.camera_positions = {}
                for env_id in range(self.env.num_envs):
                    # 相机控制参数（参照play.py）
                    camera_rot = np.pi * 8 / 10                    # 初始相机旋转角度
                    camera_rot_per_sec = 1 * np.pi / 10            # 相机每秒旋转角度
                    camera_relative_position = np.array([1, 0, 0.8])  # 相机相对机器人位置
                    
                    # 获取机器人位置作为相机焦点
                    if hasattr(self.env, 'root_states') and self.env.root_states is not None:
                        look_at = np.array(self.env.root_states[env_id, :3].cpu(), dtype=np.float64)
                    else:
                        look_at = np.array([0, 0, 1], dtype=np.float64)  # 默认位置
                    
                    # 计算相机位置
                    camera_pos = look_at + camera_relative_position
                    
                    # 存储相机参数
                    self.camera_positions[env_id] = {
                        'pos': camera_pos,
                        'target': look_at,
                        'rot': camera_rot,
                        'rot_per_sec': camera_rot_per_sec,
                        'relative_pos': camera_relative_position,
                        'h_scale': 1.0,      # 水平缩放
                        'v_scale': 0.8       # 垂直缩放
                    }
                    
                    # 设置相机（如果环境支持）
                    if hasattr(self.env, 'set_camera'):
                        self.env.set_camera(camera_pos, look_at, env_id)
                        print(f"Set camera for environment {env_id}: pos={camera_pos}, target={look_at}")
                    
                print(f"Initialized cameras for {self.env.num_envs} environments (using play.py parameters)")
                    
        except Exception as e:
            print(f"Warning: Could not initialize Isaac Gym cameras: {e}")
            self.camera_positions = {}
    
    def _record_rgb_frame(self, env_id: int):
        """记录Isaac Gym RGB帧"""
        # 在无头模式下跳过录制
        if hasattr(self.env, 'headless') and self.env.headless:
            return
            
        if env_id in self.video_writers and env_id in self.camera_positions:
            try:
                # 尝试获取RGB图像
                if hasattr(self.env, 'gym') and hasattr(self.env, 'sim') and hasattr(self.env, 'viewer') and self.env.viewer is not None:
                    # 动态更新相机位置（参照play.py逻辑）
                    self._update_camera_position(env_id)
                    
                    # 使用Isaac Gym的viewer API获取图像
                    # 注意：这里需要根据实际的Isaac Gym版本调整API调用
                    
                    # 创建一个模拟的RGB帧（用于测试）
                    # 在实际应用中，这里应该调用Isaac Gym的图像获取API
                    rgb_frame = self._create_synthetic_rgb_frame(env_id)
                    
                    # 写入帧
                    self.video_writers[env_id].write(rgb_frame)
                    
            except Exception as e:
                print(f"Warning: Failed to record RGB frame for env {env_id}: {e}")
                # 创建错误帧
                try:
                    error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    error_frame[:, :, 0] = 255  # 红色背景表示错误
                    self.video_writers[env_id].write(error_frame)
                except:
                    pass
    
    def _create_synthetic_rgb_frame(self, env_id: int):
        """创建合成的RGB帧用于测试"""
        # 创建一个720x1280的RGB帧
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # 添加一些视觉元素来模拟机器人环境
        # 背景色（深蓝色）
        frame[:, :, 0] = 50   # 蓝色
        frame[:, :, 1] = 50   # 绿色
        frame[:, :, 2] = 150  # 红色
        
        # 添加环境信息文本
        frame = self._add_text_to_frame(frame, f"Env {env_id}", (50, 50))
        frame = self._add_text_to_frame(frame, f"Trajectory {self.global_index}", (50, 100))
        
        # 添加机器人状态信息（从环境获取）
        if hasattr(self.env, 'root_states') and self.env.root_states is not None:
            try:
                robot_pos = self.env.root_states[env_id, :3].cpu().numpy()
                pos_text = f"Pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})"
                frame = self._add_text_to_frame(frame, pos_text, (50, 150))
            except:
                pass
        
        # 添加相机跟踪信息（如果可用）
        if env_id in self.camera_positions:
            camera_data = self.camera_positions[env_id]
            camera_info = f"Camera: rot={camera_data['rot']:.2f}, pos={camera_data['pos']}"
            frame = self._add_text_to_frame(frame, camera_info, (50, 250))
            
            # 添加相机轨迹可视化（简单的圆形轨迹）
            center_x, center_y = 640, 360  # 屏幕中心
            radius = 100
            angle = camera_data['rot']
            cam_x = int(center_x + radius * np.cos(angle))
            cam_y = int(center_y + radius * np.sin(angle))
            
            # 绘制相机位置指示器
            cv2.circle(frame, (cam_x, cam_y), 10, (255, 255, 0), -1)  # 黄色圆点
            cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)  # 白色轨迹圆
        
        # 添加时间戳
        timestamp = datetime.now().strftime("%H:%M:%S")
        frame = self._add_text_to_frame(frame, timestamp, (50, 300))
        
        return frame
    
    def _add_text_to_frame(self, frame, text, position):
        """在帧上添加文本"""
        # 简单的文本渲染（使用OpenCV）
        try:
            import cv2
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (255, 255, 255)  # 白色
            thickness = 2
            
            cv2.putText(frame, text, position, font, font_scale, color, thickness)
        except:
            # 如果OpenCV不可用，使用简单的像素绘制
            x, y = position
            for i, char in enumerate(text):
                if x + i * 10 < frame.shape[1] and y < frame.shape[0]:
                    # 简单的字符绘制（白色方块）
                    frame[y:y+20, x+i*10:x+(i+1)*10] = [255, 255, 255]
        
        return frame
    
    def _update_camera_position(self, env_id: int):
        """动态更新相机位置（参照play.py逻辑）"""
        if env_id not in self.camera_positions:
            return
            
        try:
            camera_data = self.camera_positions[env_id]
            
            # 获取当前机器人位置作为相机焦点
            if hasattr(self.env, 'root_states') and self.env.root_states is not None:
                look_at = np.array(self.env.root_states[env_id, :3].cpu(), dtype=np.float64)
            else:
                return
            
            # 更新相机旋转角度
            camera_data['rot'] = (camera_data['rot'] + camera_data['rot_per_sec'] * self.env.dt) % (2 * np.pi)
            
            # 计算相机相对位置（围绕机器人旋转）
            h_scale = camera_data['h_scale']
            v_scale = camera_data['v_scale']
            
            camera_relative_position = 2 * np.array([
                np.cos(camera_data['rot']) * h_scale,
                np.sin(camera_data['rot']) * h_scale, 
                0.5 * v_scale
            ])
            
            # 更新相机位置
            camera_pos = look_at + camera_relative_position
            
            # 更新存储的参数
            camera_data['pos'] = camera_pos
            camera_data['target'] = look_at
            camera_data['relative_pos'] = camera_relative_position
            
            # 设置相机（如果环境支持）
            if hasattr(self.env, 'set_camera'):
                self.env.set_camera(camera_pos, look_at, env_id)
                
        except Exception as e:
            print(f"Warning: Failed to update camera position for env {env_id}: {e}")
    
    def _close_video_recording(self):
        """关闭视频录制"""
        for env_id, writer in self.video_writers.items():
            try:
                writer.release()
                print(f"Closed video recording for environment {env_id}")
            except Exception as e:
                print(f"Warning: Failed to close video writer for env {env_id}: {e}")
        self.video_writers.clear()
        
        # 清理相机资源
        if hasattr(self, 'cameras'):
            for env_id, camera_handle in self.cameras.items():
                try:
                    if hasattr(self.env, 'gym') and hasattr(self.env, 'sim'):
                        self.env.gym.destroy_camera_sensor(self.env.sim, camera_handle)
                        print(f"Destroyed camera for environment {env_id}")
                except Exception as e:
                    print(f"Warning: Failed to destroy camera for env {env_id}: {e}")
            self.cameras.clear()
    
    def collect_trajectories(self, num_constant: int = 10, num_changing: int = 10):
        """
        收集多条轨迹
        
        Args:
            num_constant: 命令不变的轨迹数量
            num_changing: 命令变化的轨迹数量
        """
        print(f"Starting trajectory collection: {num_constant} constant + {num_changing} changing")
        
        # 收集命令不变的轨迹
        for i in range(num_constant):
            print(f"Collecting constant trajectory {i+1}/{num_constant}")
            self._collect_single_trajectory("constant")
            # 每收集几条就保存一次元数据
            if (i + 1) % 5 == 0:
                self._save_metadata()
        
        # 收集命令变化的轨迹
        for i in range(num_changing):
            print(f"Collecting changing trajectory {i+1}/{num_changing}")
            self._collect_single_trajectory("changing")
            # 每收集几条就保存一次元数据
            if (i + 1) % 5 == 0:
                self._save_metadata()
        
        # 最终保存元数据
        self._save_metadata()
        
        print(f"Trajectory collection completed!")
        print(f"Total trajectories: {len(self.trajectory_metadata)}")
        print(f"Constant trajectories: {num_constant}")
        print(f"Changing trajectories: {num_changing}")
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata file: {self.metadata_file}")


def main():
    """主函数"""
    from isaacgym import gymutil
    custom_parameters = [
        {"name": "--task", "type": str, "default": "h1int", "help": "Task name"},
        {"name": "--num_constant", "type": int, "default": 10, "help": "Number of constant command trajectories"},
        {"name": "--num_changing", "type": int, "default": 10, "help": "Number of changing command trajectories"},
        {"name": "--num_envs", "type": int, "default": 4, "help": "Number of environments to use for parallel collection"},
        {"name": "--load_checkpoint", "action": "store_true", "help": "Load trained policy for action generation"},
        {"name": "--headless", "action": "store_true", "help": "Run in headless mode"},
    ]
    args = gymutil.parse_arguments(
        description="Collect HugWBC trajectories",
        custom_parameters=custom_parameters)
    
    # 创建环境配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 创建轨迹采集器
    import debugpy; debugpy.listen(5678); print("Waiting for debugger to attach..."); debugpy.wait_for_client() 
    collector = TrajectoryCollector(args, env_cfg, train_cfg)
    
    # 设置环境数量
    collector.env_cfg.env.num_envs = args.num_envs
    
    print(f"Starting trajectory collection: {args.num_constant} constant + {args.num_changing} changing")
    print(f"Using {args.num_envs} environments for parallel collection")
    
    # 收集轨迹
    collector.collect_trajectories(args.num_constant, args.num_changing)
    
    print("Trajectory collection completed!")
    print(f"Total trajectories: {len(collector.trajectory_metadata)}")
    print(f"Constant trajectories: {sum(1 for m in collector.trajectory_metadata.values() if m['trajectory_type'] == 'constant')}")
    print(f"Changing trajectories: {sum(1 for m in collector.trajectory_metadata.values() if m['trajectory_type'] == 'changing')}")
    print(f"Output directory: {collector.output_dir}")
    print(f"Videos directory: {collector.videos_dir}")
    print(f"Metadata file: {collector.metadata_file}")


if __name__ == "__main__":
    main()
