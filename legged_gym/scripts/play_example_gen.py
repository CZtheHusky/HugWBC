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

class CommandDataCollector:
    """为每个预定义命令收集轨迹数据的数据收集器"""
    
    def __init__(self, args):
        self.args = args
        self.task_name = args.task
        self.device = 'cuda:0'
        
        # 输出路径设置
        self.output_root = "example_trajectories"
        os.makedirs(self.output_root, exist_ok=True)
        
        # 每个命令收集的轨迹数量
        self.trajectories_per_command = 10
        
        # 环境和训练配置
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=self.task_name)
        
        # 环境参数设置
        self.env_cfg.env.num_envs = self.trajectories_per_command
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
        
        # 加载策略
        self.train_cfg.runner.resume = True
        ppo_runner, _ = task_registry.make_alg_runner(
            env=self.env, name=self.task_name, args=args, train_cfg=self.train_cfg)
        self.policy = ppo_runner.get_inference_policy(device=self.env.device)
        
        # 初始化环境
        _, _ = self.env.reset()
    
    def _set_command(self, command_values: List[float]):
        """设置环境命令"""
        command_tensor = torch.tensor(command_values, device=self.env.device, dtype=torch.float32)
        # 确保命令维度正确
        cmd_dim = min(len(command_values), self.env.commands.shape[1])
        self.env.commands[:, :cmd_dim] = command_tensor[:cmd_dim].unsqueeze(0).expand(self.env.num_envs, -1)
    
    def _collect_single_trajectory(self, command_name: str, command_values: List[float]) -> Dict[str, Any]:
        """收集单条轨迹数据"""
        # 重置环境
        obs, critic_obs = self.env.reset()
        obs, critic_obs, _, _, _ = self.env.step(torch.zeros(self.env.num_envs, self.env.num_actions, dtype=torch.float, device=self.env.device))
        obs_dim = obs.shape[-1]
        cmd_dim = int(self.env.cfg.commands.num_commands)
        clock_dim = 2
        proprio_dim = int(obs_dim - (cmd_dim + clock_dim))
        # 设置命令
        self._set_command(command_values)
        
        # 存储轨迹数据的列表
        trajectory_data = {
            'actions': [[] for _ in range(self.env.num_envs)],
            'clock': [[] for _ in range(self.env.num_envs)],
            'commands': [[] for _ in range(self.env.num_envs)],
            'critic_obs': [[] for _ in range(self.env.num_envs)],
            'dones': [[] for _ in range(self.env.num_envs)],
            'history_action': [[] for _ in range(self.env.num_envs)],
            'proprio': [[] for _ in range(self.env.num_envs)],
            'rewards': [[] for _ in range(self.env.num_envs)],
            'root_states': [[] for _ in range(self.env.num_envs)],
        }
        
        # 收集episode数据
        step_count = 0
        max_steps = int(self.env_cfg.env.episode_length_s / self.env.dt)
        active_mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        last_obs = obs.detach()
        last_critic_obs = critic_obs.detach()
        while step_count < max_steps and active_mask.any():
            # 获取动作
            with torch.inference_mode():
                actions, _ = self.policy.act_inference(last_obs, privileged_obs=last_critic_obs)

            # Step environment
            obs, critic_obs, step_rewards, dones, infos = self.env.step(actions)
            for eid in range(self.env.num_envs):
                if not bool(active_mask[eid]):
                    continue
                proprio = last_obs[eid, ..., :proprio_dim].cpu().numpy()
                commands = last_obs[eid, ..., proprio_dim:proprio_dim+cmd_dim].cpu().numpy()
                clock = last_obs[eid, ..., proprio_dim+cmd_dim:].cpu().numpy()
                trajectory_data['actions'][eid].append(actions[eid].cpu().numpy())
                trajectory_data['clock'][eid].append(clock)
                trajectory_data['critic_obs'][eid].append(last_critic_obs[eid].cpu().numpy())
                trajectory_data['proprio'][eid].append(proprio)
                trajectory_data['root_states'][eid].append(self.env.root_states[eid].cpu().numpy())
                trajectory_data['rewards'][eid].append(step_rewards[eid].cpu().numpy())
                trajectory_data['dones'][eid].append(dones[eid].cpu().numpy())
                trajectory_data['commands'][eid].append(commands)
            last_obs = obs.detach()
            last_critic_obs = critic_obs.detach()
            step_count += 1
            # inactivate envs that are done at this step
            if dones.any():
                active_mask[dones > 0] = False
            # break when all envs finished early
            if (~active_mask).all():
                break
        episodes = []
        saved_rewards = []
        for eid in range(self.env.num_envs):
            traj_rew = float(sum(trajectory_data['rewards'][eid]))
            saved_rewards.append(traj_rew)
            data = {
                "proprio": np.array(trajectory_data['proprio'][eid]),
                "commands": np.array(trajectory_data['commands'][eid]),
                "clock": np.array(trajectory_data['clock'][eid]),
                "critic_obs": np.array(trajectory_data['critic_obs'][eid]),
                "actions": np.array(trajectory_data['actions'][eid]),
                "rewards": np.array(trajectory_data['rewards'][eid]),
                "dones": np.array(trajectory_data['dones'][eid]),
                "root_states": np.array(trajectory_data['root_states'][eid]),
            }
            meta_entry = {
                "episode_reward": traj_rew,
                "episode_command_A": command_values,
            }
            episodes.append({"data": data, "meta": meta_entry})
        return episodes, saved_rewards
    
    def _save_trajectories_to_buffer(self, command_name: str, trajectories: List[Dict[str, Any]]):
        """将轨迹数据保存到ReplayBuffer"""
        buffer_path = os.path.join(self.output_root, f"{command_name}.zarr")
        
        # 创建或打开ReplayBuffer
        if os.path.exists(buffer_path):
            shutil.rmtree(buffer_path)
        replay_buffer = ReplayBuffer.create_from_path(buffer_path, mode="a")
        
        data_keys = list(trajectories[0]['data'].keys())
        meta_keys = list(trajectories[0]['meta'].keys())
        episode_lengths = np.array([len(traj['data']['actions']) for traj in trajectories], dtype=np.int64)
        episode_ends = np.cumsum(episode_lengths)
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
        replay_buffer.add_chunked_data(data_dict, target_chunk_bytes=1024 * 1024 * 1024 * 64)
        replay_buffer.add_chunked_meta(meta_dict, target_chunk_bytes=1024 * 1024 * 1024 * 64)
        replay_buffer.add_chunked_meta({
            "episode_ends": episode_ends.astype(np.int64),
        }, target_chunk_bytes=1024 * 1024 * 1024 * 64)
        
        print(f"命令 '{command_name}' 的所有轨迹已保存到 {buffer_path}")
        print(replay_buffer)
    
    def collect_all_commands(self):
        """为所有预定义命令收集轨迹数据"""
        print("开始收集命令轨迹数据...")
        print(f"每个命令将收集 {self.trajectories_per_command} 条轨迹")
        print(f"数据将保存到 {self.output_root} 目录")
        
        for command_name, command_values in CANDATE_ENV_COMMANDS.items():
            print(f"\n正在收集命令: {command_name}")
            print(f"命令值: {command_values}")
            
            # 为当前命令收集多条轨迹
            trajectory, collected_rewards = self._collect_single_trajectory(command_name, command_values)
            print(f"max reward: {max(collected_rewards)}, min reward: {min(collected_rewards)}, mean reward: {sum(collected_rewards) / len(collected_rewards)}")
            
            # 保存到ReplayBuffer
            self._save_trajectories_to_buffer(command_name, trajectory)
        
        print(f"\n所有命令的轨迹数据收集完成！")
        print(f"总共收集了 {len(CANDATE_ENV_COMMANDS)} 个命令，每个命令 {self.trajectories_per_command} 条轨迹")


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
    args = get_args()
    play(args)