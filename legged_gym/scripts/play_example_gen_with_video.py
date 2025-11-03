import os
import sys
sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, update_class_from_dict
from legged_gym.dataset.replay_buffer import ReplayBuffer
from isaacgym import gymapi
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from isaacgym import gymapi
from typing import Dict, List, Any, Tuple
import threading
import time
import shutil
import imageio.v2 as imageio

class CommandDataCollector:
    """为每个预定义命令收集轨迹数据的数据收集器"""
    
    def __init__(self, args):
        self.args = args
        self.task_name = args.task
        self.device = 'cuda:0'
        
        # 输出路径设置
        self.output_root = "dataset/example_trajectories_test"
        os.makedirs(self.output_root, exist_ok=True)
        
        # 每个命令收集的轨迹数量
        self.trajectories_per_command = 30  # 支持多个 rollout
        
        # 默认开启latent收集（与data_collector_runner.py一致）
        self.save_latent = getattr(args, "save_latent", True)
        self.proprio_dim = 44
        self.action_dim = 19
        self.cmd_dim = 11
        self.clock_dim = 2
        self.privileged_dim = 24
        self.terrain_dim = 221
        self.total_obs_dim = self.proprio_dim + self.action_dim + self.cmd_dim + self.clock_dim
        self.total_privileged_dim = self.privileged_dim + self.terrain_dim + self.total_obs_dim
        # 环境和训练配置
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=self.task_name)
        
        # 环境参数设置 - 设置为1个环境以便录制单个机器人视频
        self.env_cfg.env.num_envs = 1
        self.env_cfg.env.episode_length_s = 10.0  # 每个episode 10秒
        
        # 地形和域随机化设置
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.noise.add_noise = True
        self.env_cfg.domain_rand.randomize_friction = True
        self.env_cfg.domain_rand.randomize_load = False
        self.env_cfg.domain_rand.randomize_gains = False
        self.env_cfg.domain_rand.randomize_link_props = False
        self.env_cfg.domain_rand.randomize_base_mass = False
        
        # 命令重采样时间设置为episode长度，避免中途改变命令
        self.env_cfg.commands.resampling_time = self.env_cfg.env.episode_length_s
        self.env_cfg.rewards.penalize_curriculum = False
        
        # 地形设置为平地
        self.env_cfg.terrain.mesh_type = 'trimesh'
        self.env_cfg.terrain.num_rows = 1
        self.env_cfg.terrain.num_cols = 1
        self.env_cfg.terrain.max_init_terrain_level = 1
        self.env_cfg.terrain.selected = True
        self.env_cfg.terrain.selected_terrain_type = "random_uniform"
        self.env_cfg.terrain.terrain_kwargs = {
            "random_uniform": {
                "min_height": -0.00,
                "max_height": 0.00,
                "step": 0.005,
                "downsampled_scale": 0.2
            },
        }
        
        # 创建环境
        self.env, _ = task_registry.make_env(name=self.task_name, args=args, env_cfg=self.env_cfg)
        self.train_cfg.runner.resume_path = "/cpfs/user/caozhe/workspace/HugWBC/logs/h1_interrupt/Aug21_13-31-13_/model_40000.pt"
        # 加载策略
        self.train_cfg.runner.resume = True
        ppo_runner, _ = task_registry.make_alg_runner(
            env=self.env, name=self.task_name, args=args, train_cfg=self.train_cfg)
        self.policy = ppo_runner.get_inference_policy(device=self.env.device)
        
        # 初始化环境
        _, _ = self.env.reset()
        
        # 设置视频录制参数
        self.video_width = 720
        self.video_height = 720
        self.video_fps = 30
        self.frame_skip = max(1, int(round(1.0 / (self.video_fps * self.env.dt))))
        
        # 相机控制参数
        self.camera_rot = np.pi * 8 / 10
        self.camera_rot_per_sec = 1 * np.pi / 10
        self.track_index = 0
        
        # 设置机器人外观
        self._setup_robot_appearance()
        
        # 初始化相机
        self._setup_camera()
        
        print("=" * 100)
        print(f"Data collector with video recording initialized:")
        print(f"  Total obs dim: {self.total_obs_dim}")
        print(f"  Total privileged dim: {self.total_privileged_dim}")
        print(f"  Save latent: {self.save_latent}")
        print(f"  Video resolution: {self.video_width}x{self.video_height}")
        print(f"  Video FPS: {self.video_fps}")
        print("=" * 100)
    
    def _setup_robot_appearance(self):
        """设置机器人外观"""
        for i in range(self.env.num_bodies):
            self.env.gym.set_rigid_body_color(
                self.env.envs[0], self.env.actor_handles[0], i,
                gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3)
            )
    
    def _setup_camera(self):
        """设置相机和视频录制"""
        # 获取初始相机位置
        look_at = np.array(self.env.root_states[0, :3].cpu(), dtype=np.float64)
        camera_relative_position = np.array([1, 0, 0.8])
        
        # 创建相机传感器
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.video_width
        cam_props.height = self.video_height
        self.cam_handle = self.env.gym.create_camera_sensor(self.env.envs[0], cam_props)
        
        # 设置相机位置
        self.env.gym.set_camera_location(
            self.cam_handle, self.env.envs[0],
            gymapi.Vec3(*(look_at + camera_relative_position)),
            gymapi.Vec3(*look_at)
        )
        
        # 安全设置viewer相机的函数
        def _safe_set_viewer_cam(pos, look, track_index):
            if getattr(self.env, "viewer", None) is not None:
                self.env.set_camera(pos, look, track_index)
        
        self._safe_set_viewer_cam = _safe_set_viewer_cam
        _safe_set_viewer_cam(look_at + camera_relative_position, look_at, self.track_index)
    
    def _to_rgb_frame(self, img_any):
        """将相机图像转换为RGB帧 - 从play_video.py复制"""
        W, H = self.video_width, self.video_height
        
        # 统一拿到 numpy 数组
        if isinstance(img_any, np.ndarray):
            arr = img_any
        else:
            arr = np.frombuffer(img_any, np.uint8)

        # (H*W*4,) 一维 RGBA buffer
        if arr.ndim == 1 and arr.size == H * W * 4:
            rgba = arr.reshape(H, W, 4)
            return rgba[..., :3].copy()  # RGB

        # (H, W*4) 二维 RGBA 展平
        if arr.ndim == 2 and arr.shape[0] == H and arr.shape[1] == W * 4:
            rgba = arr.reshape(H, W, 4)
            return rgba[..., :3].copy()

        # (H, W, C) 理想三维
        if arr.ndim == 3 and arr.shape[0] == H and arr.shape[1] == W:
            C = arr.shape[2]
            if C >= 3:
                return arr[..., :3].copy()
            elif C == 1:
                ch = arr[..., 0]
                if ch.dtype != np.uint8:
                    gmin, gmax = float(np.nanmin(ch)), float(np.nanmax(ch))
                    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax - gmin < 1e-12:
                        ch_u8 = np.zeros_like(ch, dtype=np.uint8)
                    else:
                        ch_u8 = np.clip((ch - gmin) / (gmax - gmin) * 255.0, 0, 255).astype(np.uint8)
                else:
                    ch_u8 = ch
                return np.stack([ch_u8, ch_u8, ch_u8], axis=-1)

        # (H, W) 灰度/深度
        if arr.ndim == 2 and arr.shape == (H, W):
            gray = arr
            if gray.dtype != np.uint8:
                gmin, gmax = float(np.nanmin(gray)), float(np.nanmax(gray))
                if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax - gmin < 1e-12:
                    gray_u8 = np.zeros_like(gray, dtype=np.uint8)
                else:
                    gray_u8 = np.clip((gray - gmin) / (gmax - gmin) * 255.0, 0, 255).astype(np.uint8)
            else:
                gray_u8 = gray
            return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        # 兜底：尝试按 RGBA 解释
        if arr.size == H * W * 4:
            rgba = arr.reshape(H, W, 4)
            return rgba[..., :3].copy()

        raise RuntimeError(f"Unexpected camera image shape/dtype: shape={arr.shape}, dtype={arr.dtype}")

    def _set_command(self, command_values: List[float]):
        """设置环境命令"""
        command_tensor = torch.tensor(command_values, device=self.env.device, dtype=torch.float32)
        # 确保命令维度正确
        cmd_dim = min(len(command_values), self.env.commands.shape[1])
        self.env.commands[:, :cmd_dim] = command_tensor[:cmd_dim].unsqueeze(0).expand(self.env.num_envs, -1)
    
    def _collect_single_trajectory(self, command_name: str, command_values: List[float], rollout_idx: int) -> Dict[str, Any]:
        """收集单条轨迹数据并录制视频"""
        # 重置环境
        obs, critic_obs = self.env.reset()
        self.env.use_disturb = False
        
        # 设置命令
        self._set_command(command_values)
        
        # 为当前命令和 rollout 创建目录结构
        command_video_dir = os.path.join(self.output_root, f"{command_name}.zarr", "videos")
        os.makedirs(command_video_dir, exist_ok=True)
        
        # 初始化视频录制 - 包含 rollout 索引
        video_path = os.path.join(command_video_dir, f"rollout_{rollout_idx:03d}.mp4")
        writer = imageio.get_writer(video_path, fps=self.video_fps)
        
        print(f"开始录制命令 '{command_name}' 第 {rollout_idx+1} 个 rollout 的视频: {video_path}")
        
        # 初始化缓冲区 - 与data_collector_runner.py完全一致
        last_obs_buffers = []
        last_critic_obs_buffers = []
        action_buffers = []
        reward_buffers = []
        done_buffers = []
        if self.save_latent:
            latent_buffers = []
        env_valid_steps = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
        
        # Track active environments
        active_mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

        # Episode loop - 与data_collector_runner.py完全一致
        t = 0
        max_steps = int(self.env_cfg.env.episode_length_s / self.env.dt)
        last_obs, last_critic_obs, _, _, _ = self.env.step(torch.zeros(self.env.num_envs, self.env.num_actions, dtype=torch.float, device=self.env.device))
        ep_start_obs = last_obs[:, :-1].detach().cpu().numpy()
        assert len(last_critic_obs.shape) == 2
        assert len(last_obs.shape) == 3
        assert last_obs.shape[-1] == self.total_obs_dim
        assert last_critic_obs.shape[-1] == self.total_privileged_dim
        
        try:
            while t < max_steps and active_mask.any():
                with torch.inference_mode():
                    actions, _ = self.policy.act_inference(last_obs, privileged_obs=last_critic_obs)
                    if self.save_latent:
                        latent = self.policy.actor.mem.detach().cpu().numpy()

                # Step environment
                self.env.use_disturb = False
                obs, critic_obs, step_rewards, dones, infos = self.env.step(actions)

                # 相机跟踪与旋转 (参考play_video.py)
                look_at = np.array(self.env.root_states[self.track_index, :3].cpu(), dtype=np.float64)
                self.camera_rot = (self.camera_rot + self.camera_rot_per_sec * self.env.dt) % (2 * np.pi)
                h_scale, v_scale = 1.0, 0.8
                camera_relative_position = 2 * np.array([
                    np.cos(self.camera_rot) * h_scale,
                    np.sin(self.camera_rot) * h_scale,
                    0.5 * v_scale
                ])

                # 更新viewer相机
                self._safe_set_viewer_cam(look_at + camera_relative_position, look_at, self.track_index)
                
                # 同步传感器相机（用于录制）
                self.env.gym.set_camera_location(
                    self.cam_handle, self.env.envs[0],
                    gymapi.Vec3(*(look_at + camera_relative_position)),
                    gymapi.Vec3(*look_at)
                )

                # 视频录制
                if t % self.frame_skip == 0:
                    # 确保渲染管线推进
                    self.env.gym.step_graphics(self.env.sim)
                    self.env.gym.render_all_camera_sensors(self.env.sim)

                    img_any = self.env.gym.get_camera_image(
                        self.env.sim, self.env.envs[0], self.cam_handle, gymapi.IMAGE_COLOR
                    )
                    rgb = self._to_rgb_frame(img_any)
                    writer.append_data(rgb)

                # 数据收集 - 与data_collector_runner.py完全一致
                last_obs_buffers.append(last_obs[:, -1].detach().cpu().numpy())
                last_critic_obs_buffers.append(last_critic_obs[:, last_obs.shape[-1]:].detach().cpu().numpy())
                action_buffers.append(actions.detach().cpu().numpy())
                reward_buffers.append(step_rewards.cpu().numpy())
                done_buffers.append(dones.cpu().numpy())
                if self.save_latent:
                    latent_buffers.append(latent)
                
                last_obs = obs.detach()
                last_critic_obs = critic_obs.detach()
                t += 1
                # inactivate envs that are done at this step
                env_valid_steps[active_mask] += 1
                if dones.any():
                    active_mask[dones > 0] = False
                # break when all envs finished early
                if (~active_mask).all():
                    break
        finally:
            # 确保视频文件被正确关闭
            writer.close()
            print(f"视频已保存: {video_path}")

        # finalize: build episodes - 与data_collector_runner.py完全一致
        episodes: List[Dict[str, Any]] = []
        saved_rewards: List[float] = []
        last_obs_buffers = np.stack(last_obs_buffers, axis=1)
        last_critic_obs_buffers = np.stack(last_critic_obs_buffers, axis=1)
        action_buffers = np.stack(action_buffers, axis=1)
        reward_buffers = np.stack(reward_buffers, axis=1).astype(np.float32)
        done_buffers = np.stack(done_buffers, axis=1).astype(bool)
        if self.save_latent:
            latent_buffers = np.stack(latent_buffers, axis=1)
        env_valid_steps = env_valid_steps.cpu().numpy()
        
        for eid in range(self.env.num_envs):
            env_steps = env_valid_steps[eid]
            traj_reward = reward_buffers[eid, :env_steps]

            traj_proprio = last_obs_buffers[eid, :env_steps, :self.proprio_dim]
            traj_cmd = last_obs_buffers[eid, :env_steps, -self.cmd_dim - self.clock_dim:-self.clock_dim]
            traj_clock = last_obs_buffers[eid, :env_steps, -self.clock_dim:]

            traj_privileged = last_critic_obs_buffers[eid, :env_steps, :self.privileged_dim]
            traj_terrain = last_critic_obs_buffers[eid, :env_steps, self.privileged_dim:]

            traj_action = action_buffers[eid, :env_steps]
            traj_done = done_buffers[eid, :env_steps]

            traj_rew = float(sum(traj_reward))
            traj_step_reward = np.mean(traj_reward)
            saved_rewards.append(traj_rew)
            
            data = {
                "proprio": np.array(traj_proprio),
                "commands": np.array(traj_cmd),
                "clock": np.array(traj_clock),
                # other step data
                "privileged": np.array(traj_privileged),
                "terrain": np.array(traj_terrain),
                "actions": np.array(traj_action),
                "rewards": np.array(traj_reward).astype(np.float32),
                "dones": np.array(traj_done).astype(bool),
            }
            if self.save_latent:
                data['latent'] = latent_buffers[eid, :env_steps]

            meta_entry: Dict[str, Any] = {
                "episode_reward": traj_rew,
                "episode_step_reward": traj_step_reward,
                "ep_start_obs": ep_start_obs[eid],
                "episode_command": command_values,
            }
            episodes.append({"data": data, "meta": meta_entry})
        return episodes, saved_rewards
    
    def _save_trajectories_to_buffer(self, command_name: str, trajectories: List[Dict[str, Any]]):
        """将轨迹数据保存到ReplayBuffer"""
        buffer_path = os.path.join(self.output_root, f"{command_name}.zarr")
        
        # # 创建或打开ReplayBuffer
        # if os.path.exists(buffer_path):
        #     shutil.rmtree(buffer_path)
        replay_buffer = ReplayBuffer.create_from_path(buffer_path, mode="a")
        
        data_keys = list(trajectories[0]['data'].keys())
        meta_keys = list(trajectories[0]['meta'].keys())
        episode_lengths = np.array([len(traj['data']['actions']) for traj in trajectories], dtype=np.int64)
        episode_ends = np.cumsum(episode_lengths)
        episode_ends += replay_buffer.n_steps
        data_dict = {key: [] for key in data_keys}
        meta_dict = {key: [] for key in meta_keys}

        # 处理每条轨迹
        for traj_idx, trajectory in enumerate(trajectories):
            for key in data_keys:
                data_dict[key].append(trajectory['data'][key])
            for key in meta_keys:
                meta_dict[key].append(trajectory['meta'][key])
        for k, v in data_dict.items():
            data_dict[k] = np.concatenate(v, axis=0)
        for k, v in meta_dict.items():
            meta_dict[k] = np.array(v)
        for k, v in data_dict.items():
            print(k, v.shape)
        for k, v in meta_dict.items():
            print(k, v.shape)
        replay_buffer.add_chunked_data(data_dict, target_chunk_bytes=128 * 1024 * 1024)
        # 合并写入meta并包含episode_ends，符合最新writer风格
        meta_dict["episode_ends"] = episode_ends.astype(np.int64)
        replay_buffer.add_chunked_meta(meta_dict, target_chunk_bytes=64 * 1024 * 1024)
        
        print(f"命令 '{command_name}' 的所有轨迹已保存到 {buffer_path}")
        print(replay_buffer)
    
    def collect_all_commands(self):
        """为所有预定义命令收集轨迹数据并录制视频"""
        print("开始收集命令轨迹数据并录制视频...")
        print(f"每个命令将收集 {self.trajectories_per_command} 条轨迹")
        print(f"数据将保存到 {self.output_root} 目录")
        
        total_commands = len(CANDATE_ENV_COMMANDS)
        for cmd_idx, (command_name, command_values) in enumerate(CANDATE_ENV_COMMANDS.items()):
            print(f"\n[{cmd_idx+1}/{total_commands}] 正在处理命令: {command_name}")
            print(f"命令值: {command_values}")
            
            # 为当前命令收集多个 rollout
            all_rewards = []
            for rollout_idx in range(self.trajectories_per_command):
                print(f"  开始第 {rollout_idx+1}/{self.trajectories_per_command} 个 rollout...")
                
                # 收集单个轨迹并录制视频
                trajectory, collected_rewards = self._collect_single_trajectory(command_name, command_values, rollout_idx)
                all_rewards.extend(collected_rewards)
                
                # 保存到ReplayBuffer
                self._save_trajectories_to_buffer(command_name, trajectory)
                
                print(f"  第 {rollout_idx+1} 个 rollout 完成，奖励: {collected_rewards[0]:.2f}")
            
            # 输出当前命令的整体统计
            print(f"命令 '{command_name}' 完成 - 奖励统计:")
            print(f"  max: {max(all_rewards):.2f}, min: {min(all_rewards):.2f}, mean: {sum(all_rewards) / len(all_rewards):.2f}")


def play(args):
    """主函数：收集所有命令的轨迹数据"""
    collector = CommandDataCollector(args)
    collector.collect_all_commands()


# 预定义的候选环境命令
CANDATE_ENV_COMMANDS = {
    "default": [0, 0, 0, 2, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_backward_walk": [-0.6, 0, 0, 1, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_forward_walk": [0.6, 0, 0, 1, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_backward_walk_low_height": [-0.6, 0, 0, 1, 0.15, 0.5, -0.3, 0, 0, 0],
    "slow_forward_walk_low_height": [0.6, 0, 0, 1, 0.15, 0.5, -0.3, 0, 0, 0],
    "fast_walk": [2, 0, 0, 2.5, 0.15, 0.5, 0.3, 0, 0, 0],
    "slow_turn": [0.5, 0, -0.5, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
    "fast_turn": [0.5, 0, 0.5, 2.5, 0.15, 0.5, 0.2, 0, 0, 0],
    "crab_right_walk": [0, 0.6, 0, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
    "crab_left_walk": [0, -0.6, 0, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
}


if __name__ == '__main__':
    import argparse
    
    # 添加命令行参数支持，与data_collector_runner.py保持一致
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="h1int", help="Task name")
    parser.add_argument("--save_latent", action="store_true", default=True, help="Save latent states")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless")
    
    # 为了兼容性，也支持原有的get_args方式
    try:
        args = get_args()
        # 如果get_args成功，添加save_latent属性（默认True）
        if not hasattr(args, 'save_latent'):
            args.save_latent = True
    except:
        # 如果get_args失败，使用argparse
        args = parser.parse_args()
    
    play(args)