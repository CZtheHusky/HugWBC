import os
import sys
sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, update_class_from_dict
from isaacgym import gymapi
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from isaacgym import gymapi
from legged_gym.dataset.replay_buffer import ReplayBuffer
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MyDataset_legacy(Dataset):
    def __init__(self, rb):
        self.data_dict = {}
        for key in rb.data.keys():
            self.data_dict[key] = rb.data[key][:]
        
    
    def __len__(self):
        return len(self.data_dict['actions']) - 1
    
    def __getitem__(self, idx):
        proprio = self.data_dict['proprio'][idx]
        cmd = self.data_dict['commands'][idx]
        clock = self.data_dict['clock'][idx]
        obs = np.concatenate([proprio, cmd, clock], axis=-1)
        critic_obs = self.data_dict['critic_obs'][idx]
        actions = self.data_dict['actions'][idx]
        return obs, critic_obs, actions

class MyDataset(Dataset):
    def __init__(self, rb, horizon=5):
        self.data_dict = {}
        for key in rb.data.keys():
            self.data_dict[key] = rb.data[key][:]
        self.horizon = horizon
        episode_ends = rb.meta['episode_ends'][:]
        # map the index to the episode id
        episode_id = np.repeat(np.arange(len(episode_ends)), np.diff([0, *episode_ends]))
        self.episode_id = episode_id
        self.episode_ends = episode_ends
        self.ep_start_obs = rb.meta['ep_start_obs'][:]
    
    def __len__(self):
        return len(self.data_dict['actions'])
    
    def __getitem__(self, idx):
        ep_id = self.episode_id[idx]
        ep_start = 0 if ep_id == 0 else self.episode_ends[ep_id - 1]
        horizon_start = max(ep_start, idx - self.horizon + 1)
        horizon_end = idx + 1
        cmd = self.data_dict['commands'][horizon_start:horizon_end]
        clock = self.data_dict['clock'][horizon_start:horizon_end]
        proprio = self.data_dict['proprio'][horizon_start:horizon_end]
        valid_len = cmd.shape[0]
        history_action = np.zeros((valid_len, self.data_dict['actions'].shape[-1]))
        if valid_len > 1:
            his_a_start = max(ep_start, idx - self.horizon)
            his_a_end = idx
            history_len = his_a_end - his_a_start
            history_action[-history_len:] = self.data_dict['actions'][his_a_start:his_a_end]
        obs = np.concatenate([proprio, history_action, cmd, clock], axis=-1)
        if valid_len < self.horizon:
            obs = np.concatenate([self.ep_start_obs[ep_id, -(self.horizon - valid_len):], obs], axis=0)
        terrain = self.data_dict['terrain'][idx]
        privileged = self.data_dict['privileged'][idx]
        critic_obs = np.concatenate([obs[-1], privileged, terrain], axis=-1)
        actions = self.data_dict['actions'][idx]
        return obs.astype(np.float32), critic_obs.astype(np.float32), actions.astype(np.float32)

def play(args):
    # 获取环境配置和训练配置
    args.task = "h1int"
    args.load_run = "Aug21_13-31-13_"
    args.checkpoint = 40000
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    
    # ==================== 环境参数覆盖设置 ====================
    # 覆盖训练时的环境数量，测试时只使用1个环境，便于观察和调试
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    # 设置很长的episode长度，避免自动重置，便于长时间观察
    env_cfg.env.episode_length_s = 100000

    # ==================== 地形设置 ====================
    # 关闭地形课程学习，保持固定难度
    env_cfg.terrain.curriculum = False
    # 启用噪声添加
    env_cfg.noise.add_noise = True
    
    # ==================== 域随机化设置 ====================
    # 启用摩擦系数随机化，增加测试的真实性
    env_cfg.domain_rand.randomize_friction = True
    # 关闭负载随机化，保持一致性
    env_cfg.domain_rand.randomize_load = False
    # 关闭增益随机化，保持一致性
    env_cfg.domain_rand.randomize_gains = False 
    # 关闭链接属性随机化，保持一致性
    env_cfg.domain_rand.randomize_link_props = False
    # 关闭基座质量随机化，保持一致性
    env_cfg.domain_rand.randomize_base_mass = False

    # ==================== 命令和奖励设置 ====================
    # 设置命令重采样时间（100秒），减少命令变化频率
    env_cfg.commands.resampling_time = 100000
    # 关闭奖励课程学习，保持固定奖励结构
    env_cfg.rewards.penalize_curriculum = False
    
    # ==================== 地形类型和参数设置 ====================
    # 设置地形类型为三角网格
    env_cfg.terrain.mesh_type = 'trimesh'
    # 简化为单行单列地形
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    # 设置初始地形难度等级
    env_cfg.terrain.max_init_terrain_level = 1
    # 选择特定地形类型而不是随机生成
    env_cfg.terrain.selected = True
    # 选择均匀随机地形类型
    env_cfg.terrain.selected_terrain_type = "random_uniform"
    # 地形参数配置：创建平地地形
    env_cfg.terrain.terrain_kwargs = {  # Dict of arguments for selected terrain
        "random_uniform":
            {
                "min_height": -0.00,      # 最小高度
                "max_height": 0.00,       # 最大高度（平地）
                "step": 0.005,            # 高度步长
                "downsampled_scale": 0.2  # 下采样比例
            },
    }

    # ==================== 环境创建 ====================
    # 创建环境实例
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # ==================== 机器人外观设置 ====================
    # 设置所有刚体为灰色，便于观察
    for i in range(env.num_bodies):
        env.gym.set_rigid_body_color(env.envs[0], env.actor_handles[0], i, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))
    
    # ==================== 策略加载 ====================
    # 加载训练好的策略
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # rb_path = "/root/workspace/HugWBC/example_trajectories/crab_left_walk.zarr"
    # rb_path = "/root/workspace/HugWBC/collected_trajectories_v2/switch.zarr"
    # rb_path = "/root/workspace/HugWBC/dataset/example_trajectories/crab_right_walk.zarr"
    # rb_path = "/root/workspace/HugWBC/dataset/model_40000/switch.zarr"
    rb_path = "dataset/test/switch.zarr"
    rb = ReplayBuffer.create_from_path(rb_path)
    dataset = MyDataset(rb)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)
    gt_action_log_probs = []
    privileged_recon_losses = []
    mean_mse_errors = []
    min_mse_errors = []
    max_mse_errors = []
    policy.eval()
    for i, (obs, critic_obs, actions) in enumerate(dataloader):
        with torch.inference_mode():
            obs = torch.as_tensor(obs).to(env.device)
            critic_obs = torch.as_tensor(critic_obs).to(env.device)
            actions = torch.as_tensor(actions).to(env.device)
            action_sample = policy.act(obs, privileged_obs=critic_obs, sync_update=True)
            privileged_recon_loss = policy.actor.privileged_recon_loss.item()
            # action_mse = ((action_sample - actions) ** 2).mean()
            privileged_recon_losses.append(privileged_recon_loss)
            gt_action_log_prob = policy.get_actions_log_prob(actions).mean()
            dist = policy.distribution
            action_mse = ((dist.mean - actions) ** 2)
            min_mse = action_mse.min()
            max_mse = action_mse.max()
            mean_mse = action_mse.mean()
            # sample_mse_errors.append(action_mse.item())
            mean_mse_errors.append(mean_mse.item())
            min_mse_errors.append(min_mse.item())
            max_mse_errors.append(max_mse.item())
            gt_action_log_probs.append(gt_action_log_prob.item())
        if i % 100 == 0:
            print(f"Iteration {i}, mean_mse_errors: {np.mean(mean_mse_errors)}, min_mse_errors: {np.mean(min_mse_errors)}, max_mse_errors: {np.mean(max_mse_errors)}, gt_action_log_probs: {np.mean(gt_action_log_probs)} , privileged_recon_loss: {np.mean(privileged_recon_losses)}")

    print(f"mean_mse_errors: {np.mean(mean_mse_errors)}, min_mse_errors: {np.mean(min_mse_errors)}, max_mse_errors: {np.mean(max_mse_errors)}, gt_action_log_probs: {np.mean(gt_action_log_probs)}, privileged_recon_loss: {np.mean(privileged_recon_losses)}")

# python -m debugpy --listen 5678 --wait-for-client /root/workspace/HugWBC/legged_gym/scripts/play_ds_check.py --task=h1int --headless --load_run=Aug21_13-31-13_ --checkpoint=40000
    
CANDATE_ENV_COMMANDS = {
    "default": [0, 0, 0, 2, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_backward_walk": [-0.6, 0, 0, 1, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_forward_walk": [0.6, 0, 0, 1, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_backward_walk_low_height": [-0.6, 0, 0, 1, 0.15, 0.5,-0.3, 0, 0, 0],
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