import os
import sys
sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from isaacgym import gymapi
import numpy as np
import torch
import tqdm


def play(args):
    # 获取环境配置和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    resume_path = train_cfg.runner.resume_path
    print(resume_path)
    
    # ==================== 环境参数覆盖设置 ====================
    # 覆盖训练时的环境数量，测试时只使用1个环境，便于观察和调试
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    # 设置很长的episode长度，避免自动重置，便于长时间观察
    env_cfg.env.episode_length_s = 100000

    # ==================== 地形设置 ====================
    # 关闭地形课程学习，保持固定难度
    env_cfg.terrain.curriculum = False

    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load policy for inference
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Camera setup (orbiting around the robot as in play.py)
    track_index = 0
    camera_rot = np.pi * 8 / 10
    camera_rot_per_sec = 1 * np.pi / 10
    h_scale = 1
    v_scale = 0.8
    look_at = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
    camera_relative_position = 2 * np.array([
        np.cos(camera_rot) * h_scale,
        np.sin(camera_rot) * h_scale,
        0.5 * v_scale,
    ])
    env.set_camera(look_at + camera_relative_position, look_at, track_index)

    # Reset and take one zero-action step to fetch initial observations
    _, _ = env.reset()
    obs, critic_obs, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device))

    # Run loop
    timesteps = int(env_cfg.env.episode_length_s / max(env.dt, 1e-6))
    for _ in tqdm.tqdm(range(timesteps)):
        with torch.inference_mode():
            actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)
            obs, critic_obs, _, _, _ = env.step(actions)

            # Update orbiting camera each step
            look_at = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
            camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
            camera_relative_position = 2 * np.array([
                np.cos(camera_rot) * h_scale,
                np.sin(camera_rot) * h_scale,
                0.5 * v_scale,
            ])
            env.set_camera(look_at + camera_relative_position, look_at, track_index)


if __name__ == '__main__':
    args = get_args()
    play(args)


