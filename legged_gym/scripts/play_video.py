from curses import echo
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
import imageio.v2 as imageio



def play(args):
    # 获取环境配置和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    resume_path = train_cfg.runner.resume_path
    print(resume_path)
    
    # ==================== 环境参数覆盖设置 ====================
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    switch_interval = 10  # 秒

    commands_names = list(CANDATE_ENV_COMMANDS.keys())
    env_cfg.env.episode_length_s = switch_interval * len(commands_names)

    # ==================== 地形设置 ====================
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    
    # ==================== 域随机化设置 ====================
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_load = False
    env_cfg.domain_rand.randomize_gains = False 
    env_cfg.domain_rand.randomize_link_props = False
    env_cfg.domain_rand.randomize_base_mass = False

    # ==================== 命令与奖励设置 ====================
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s
    env_cfg.rewards.penalize_curriculum = False
    
    # ==================== 地形类型和参数设置 ====================
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.selected = True
    env_cfg.terrain.selected_terrain_type = "random_uniform"
    env_cfg.terrain.terrain_kwargs = {
        "random_uniform": {
            "min_height": -0.00,
            "max_height": 0.00,
            "step": 0.005,
            "downsampled_scale": 0.2
        },
    }

    # ==================== 环境创建 ====================
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    def _safe_set_viewer_cam(pos, look, track_index):
        if getattr(env, "viewer", None) is not None:
            env.set_camera(pos, look, track_index)
    # ==================== 机器人外观设置 ====================
    for i in range(env.num_bodies):
        env.gym.set_rigid_body_color(
            env.envs[0], env.actor_handles[0], i,
            gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3)
        )
    
    # ==================== 策略加载 ====================
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ==================== 相机控制配置 ====================
    camera_rot = np.pi * 8 / 10
    camera_rot_per_sec = 1 * np.pi / 10
    camera_relative_position = np.array([1, 0, 0.8])
    track_index = 0

    # ==================== 环境重置和初始化 ====================
    look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
    _safe_set_viewer_cam(look_at + camera_relative_position, look_at, track_index)
    # env.set_camera(look_at + camera_relative_position, look_at, track_index)

    _, _ = env.reset()
    obs, critic_obs, _, _, _ = env.step(torch.zeros(
        env.num_envs, env.num_actions, dtype=torch.float, device=env.device))

    look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)


    # 用 _safe_set_viewer_cam 替换你代码中的 env.set_camera(...)
    _safe_set_viewer_cam(look_at + camera_relative_position, look_at, track_index)
    # env.set_camera(look_at + camera_relative_position, look_at, track_index)

    # ==================== 相机传感器与视频写出器 ====================
    VIDEO_PATH = "rgb.mp4"
    W, H, FPS = 1280, 720, 30
    frame_skip = max(1, int(round(1.0 / (FPS * env.dt))))  # 模拟 -> 视频帧率下采样

    cam_props = gymapi.CameraProperties()
    cam_props.width = W
    cam_props.height = H
    cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)

    # 让传感器相机与 viewer 相机初始对齐
    env.gym.set_camera_location(
        cam_handle, env.envs[0],
        gymapi.Vec3(*(look_at + camera_relative_position)),
        gymapi.Vec3(*look_at)
    )

    # imageio 写 mp4，依赖 ffmpeg（imageio-ffmpeg）
    writer = imageio.get_writer(VIDEO_PATH, fps=FPS)

    # --- 工具函数：将 get_camera_image 的任意返回形态转成 RGB 帧 ---
    def _to_rgb_frame(img_any):
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
    # ==================== 主循环：策略执行、相机控制与录制 ====================
    timesteps = int(env_cfg.env.episode_length_s / env.dt) + 1
    switch_steps = int(switch_interval / env.dt)
    print(f"switch_steps: {switch_steps}")
    command_name = commands_names[0]
    current_command = torch.tensor(np.array(CANDATE_ENV_COMMANDS[command_name]), device=env.device)
    env.commands[:, :10] = current_command
    current_reward = 0

    try:
        for timestep in tqdm.tqdm(range(timesteps)):
            with torch.inference_mode():
                actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)
                obs, critic_obs, reward, dones, _ = env.step(actions)
                current_reward += reward.item()
                if dones.any():
                    print(f"command: {command_name}, current reward: {current_reward}")
                    current_reward = 0

                # ===== 相机跟踪与旋转 =====
                look_at = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
                camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
                h_scale, v_scale = 1.0, 0.8
                camera_relative_position = 2 * np.array(
                    [np.cos(camera_rot) * h_scale, np.sin(camera_rot) * h_scale, 0.5 * v_scale]
                )

                # 更新 viewer 相机
                _safe_set_viewer_cam(look_at + camera_relative_position, look_at, track_index)
                # env.set_camera(look_at + camera_relative_position, look_at, track_index)
                # 同步传感器相机（用于录制）
                env.gym.set_camera_location(
                    cam_handle, env.envs[0],
                    gymapi.Vec3(*(look_at + camera_relative_position)),
                    gymapi.Vec3(*look_at)
                )

                # ===== 切换命令（循环取模）=====
                if timestep % switch_steps == 0:
                    idx = (timestep // switch_steps) % len(commands_names)
                    command_name = commands_names[idx]
                    current_command = torch.tensor(np.array(CANDATE_ENV_COMMANDS[command_name]), device=env.device)
                    print(f"command: {command_name}, current commands: {current_command}")
                env.commands[:, :10] = current_command

                # ===== 干扰和中断设置 =====
                env.use_disturb = True
                env.disturb_masks[:] = True
                env.disturb_isnoise[:] = True
                env.disturb_rad_curriculum[:] = 1.0
                env.interrupt_mask[:] = env.disturb_masks[:]
                env.standing_envs_mask[:] = True

                # ===== 抓帧与写视频（CPU 路径）=====
                if timestep % frame_skip == 0:
                    # 确保渲染管线推进
                    env.gym.step_graphics(env.sim)
                    env.gym.render_all_camera_sensors(env.sim)

                    img_any = env.gym.get_camera_image(
                        env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR
                    )
                    rgb = _to_rgb_frame(img_any)
                    writer.append_data(rgb)
    finally:
        writer.close()
        print(f"RGB 已保存到: {VIDEO_PATH}")

CANDATE_ENV_COMMANDS = {
    "default": [0, 0, 0, 2, 0.15, 0.5, 0.2, 0, 0, 0],
    # "slow_backward_walk": [-0.6, 0, 0, 1, 0.15, 0.5, 0.2, 0, 0, 0],
    # "slow_forward_walk": [0.6, 0, 0, 1, 0.15, 0.5, 0.2, 0, 0, 0],
    # "slow_backward_walk_low_height": [-0.6, 0, 0, 1, 0.15, 0.5,-0.3, 0, 0, 0],
    # "slow_forward_walk_low_height": [0.6, 0, 0, 1, 0.15, 0.5, -0.3, 0, 0, 0],
    # "fast_walk": [2, 0, 0, 2.5, 0.15, 0.5, 0.3, 0, 0, 0],
    # "slow_turn": [0.5, 0, -0.5, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
    # "fast_turn": [0.5, 0, 0.5, 2.5, 0.15, 0.5, 0.2, 0, 0, 0],
    # "crab_right_walk": [0, 0.6, 0, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
    # "crab_left_walk": [0, -0.6, 0, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
}


if __name__ == '__main__':
    args = get_args()
    play(args)