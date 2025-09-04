import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
import queue

import numpy as np
import zarr

# ensure local repo is importable
sys.path.append(os.getcwd())
from tqdm import tqdm
from legged_gym.envs import *  # registers tasks
from legged_gym.utils import task_registry
from legged_gym.dataset.replay_buffer import ReplayBuffer
import torch
from isaacgym.torch_utils import torch_rand_float
import shutil

# Actor MLP: MlpAdaptModel(
#   (state_estimator): Sequential(
#     (0): Linear(in_features=32, out_features=64, bias=True)
#     (1): ELU(alpha=1.0)
#     (2): Linear(in_features=64, out_features=32, bias=True)
#     (3): ELU(alpha=1.0)
#     (4): Linear(in_features=32, out_features=3, bias=True)
#   )
#   (low_level_net): Sequential(
#     (0): Linear(in_features=111, out_features=256, bias=True)
#     (1): ELU(alpha=1.0)
#     (2): Linear(in_features=256, out_features=128, bias=True)
#     (3): ELU(alpha=1.0)
#     (4): Linear(in_features=128, out_features=32, bias=True)
#     (5): ELU(alpha=1.0)
#     (6): Linear(in_features=32, out_features=19, bias=True)
#   )
#   (mem_encoder): Sequential(
#     (0): Linear(in_features=315, out_features=256, bias=True)
#     (1): ELU(alpha=1.0)
#     (2): Linear(in_features=256, out_features=128, bias=True)
#     (3): ELU(alpha=1.0)
#     (4): Linear(in_features=128, out_features=32, bias=True)
#   )
# )
# Critic MLP: Sequential(
#   (0): Linear(in_features=321, out_features=512, bias=True)
#   (1): ELU(alpha=1.0)
#   (2): Linear(in_features=512, out_features=256, bias=True)
#   (3): ELU(alpha=1.0)
#   (4): Linear(in_features=256, out_features=128, bias=True)
#   (5): ELU(alpha=1.0)
#   (6): Linear(in_features=128, out_features=1, bias=True)
# )

class ReplayBufferWriter:
    def __init__(self, zarr_path: str, traj_type: str, stop_event: threading.Event, max_queue_size: int = 8, small_chunks: bool = False):
        self.zarr_path = zarr_path
        self.traj_type = traj_type  # "constant" or "switch"
        self.stop_event = stop_event
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._run, name=f"ReplayWriter-{traj_type}", daemon=False)
        self._started = False
        self.error = None
        self.small_chunks = small_chunks
        self.data_keys = None
        self.disk_keys = None

    def start(self):
        if not self._started:
            self._started = True
            self.thread.start()

    def join(self):
        
        if not self._started:
            return
        # signal stop to allow thread to break idle waits
        self.stop_event.set()
        # wait until all queued items processed
        self.queue.join()
        # and wait for thread exit
        if self.thread.is_alive():
            self.thread.join()


    def put_batch(self, batch: Dict[str, Any]):
        # blocks if queue full
        self.queue.put(batch)

    def _open_or_repair(self) -> ReplayBuffer:
        try:
            return ReplayBuffer.create_from_path(self.zarr_path, mode="a")
        except AssertionError as e:
            print(f"[ReplayBufferWriter-{self.traj_type}] detected mismatched lengths, attempting repair: {e}")
            group = zarr.open(self.zarr_path, mode="a")
            meta = group["meta"]
            if "episode_ends" not in meta or meta["episode_ends"].size == 0:
                # empty meta, reset dataset
                del group["data"]
                del group["meta"]
                return ReplayBuffer.create_from_path(self.zarr_path, mode="a")
            target = int(meta["episode_ends"][-1])
            data_grp = group.get("data", None)
            if data_grp is not None:
                for key, arr in data_grp.items():
                    if arr.shape[0] != target:
                        print(f"[Repair] Truncating {key} from {arr.shape[0]} to {target}")
                        arr.resize((target,) + arr.shape[1:])
            # done, reopen
            return ReplayBuffer.create_from_path(self.zarr_path, mode="a")

    def _run(self):
        
        # Open or create zarr replay buffer (with repair if needed)
        buffer = self._open_or_repair()
        while True:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            try:
                episodes: List[Dict[str, Any]] = item["episodes"]
                t0 = time.time()
                # Build bulk arrays by concatenating along time dimension
                lengths = np.array([len(ep["data"]["proprio"]) for ep in episodes]).astype(np.int64)
                total_T = int(lengths.sum())

                # Collect and concat keys present
                # we now split obs into proprio/commands/clock ahead of time
                if self.data_keys is None:
                    data_keys = list(episodes[0]["data"].keys())
                    if self.small_chunks:
                        disk_keys = ["proprio", "privileged", "terrain"]
                        for key in disk_keys:
                            data_keys.remove(key)
                    else:
                        disk_keys = []
                    self.data_keys = data_keys
                    self.disk_keys = disk_keys
                concat_data: Dict[str, np.ndarray] = {}
                for key in self.data_keys:
                        arrays = []
                        for ep in episodes:
                            arr = ep["data"][key]
                            if not isinstance(arr, np.ndarray):
                                arr = np.asarray(arr)
                            arrays.append(arr)
                        if arrays:
                            concat_data[key] = np.concatenate(arrays, axis=0)

                # Append all step data in one call
                if concat_data:
                    # reduce chunk target to avoid huge memory spikes
                    buffer.add_chunked_data(concat_data, target_chunk_bytes=128 * 1024 * 1024)

                for ep in episodes:
                    disk_data = {}
                    for key in self.disk_keys:
                        disk_data[key] = np.asarray(ep["data"][key])
                    # set the chunk targe to be 4kb
                    buffer.add_chunked_data(disk_data, target_chunk_bytes=2 * 1024 * 1024)

                # Compute episode_ends based on current steps
                curr_steps = buffer.n_steps
                start_steps = curr_steps
                episode_ends = start_steps + np.cumsum(lengths)

                rewards = np.array([ep["meta"]["episode_reward"] for ep in episodes]).astype(np.float32)
                step_rewards = np.array([ep["meta"]["episode_step_reward"] for ep in episodes]).astype(np.float32)
                if self.traj_type == "switch":
                    cmd_A = np.stack([np.asarray(ep["meta"]["episode_command_A"]).astype(np.float32) for ep in episodes])
                    cmd_B = np.stack([np.asarray(ep["meta"]["episode_command_B"]).astype(np.float32) for ep in episodes])
                    meta_bulk = {
                        "episode_ends": episode_ends.astype(np.int64),
                        "episode_reward": rewards,
                        "episode_step_reward": step_rewards,
                        "episode_command_A": cmd_A,
                        "episode_command_B": cmd_B,
                    }
                else:
                    cmd = np.stack([np.asarray(ep["meta"]["episode_command"]).astype(np.float32) for ep in episodes])
                    meta_bulk = {
                        "episode_ends": episode_ends.astype(np.int64),
                        "episode_reward": rewards,
                        "episode_step_reward": step_rewards,
                        "episode_command": cmd,
                    }
                buffer.add_chunked_meta(meta_bulk, target_chunk_bytes=64 * 1024 * 1024)

                dt = time.time() - t0
                print(f"[ReplayBufferWriter-{self.traj_type}] wrote {len(episodes)} episodes (T={total_T}) in {dt:.3f}s (total_eps={buffer.n_episodes})")
            except Exception as e:
                # record error so the producer thread can detect it
                self.error = e
                # mark this item done to avoid queue.join() deadlock
                try:
                    self.queue.task_done()
                finally:
                    # propagate to stop quickly
                    self.stop_event.set()
                raise
            else:
                # mark success
                self.queue.task_done()
        # graceful exit of thread



def _parse_merged_args():
    from isaacgym import gymutil
    # merge helpers.get_args() custom_parameters with collector-specific ones
    base_custom_parameters = [
        {"name": "--task", "type": str, "default": "h1int", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": True,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--sim_joystick", "action": "store_true", "default": False, "help": "Sample commands from sim joystick"},
    ]
    collector_parameters = [
        {"name": "--load_checkpoint", "type": str},
        {"name": "--output_root", "type": str, "default": "test"},
        {"name": "--num_total", "type": int, "default": 10},
        {"name": "--const_prob", "type": float, "default": 0},
        {"name": "--switch_prob", "type": float, "default": 0},
        {"name": "--flush_episodes", "type": int, "default": 1000},
        {"name": "--episode_length_s", "type": float, "default": 10.0},
        {"name": "--small_chunks", "action": "store_true", "default": False},
        {"name": "--overwrite", "action": "store_true", "default": False},
    ]
    
    
    args = gymutil.parse_arguments(
        description="Trajectory Data Collector",
        custom_parameters=base_custom_parameters + collector_parameters,
    )
    # align names like helpers.get_args
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


class TrajectoryDataCollector:
    def __init__(
        self,
        args: Optional[Any] = None,
        max_learning_iter: int = 40000,
    ) -> None:
        self.args = args if args is not None else _parse_merged_args()
        self.task_name = getattr(self.args, "task", "h1int")
        self.num_envs = getattr(self.args, "num_envs", 4)
        self.headless = bool(getattr(self.args, "headless", True))
        self.load_checkpoint = getattr(self.args, "load_checkpoint")
        self.output_root = getattr(self.args, "output_root", "collected")
        self.output_root = os.path.join("dataset", self.output_root)
        self.flush_episodes = int(getattr(self.args, "flush_episodes", 1000))
        self.max_learning_iter = max_learning_iter
        self.proprio_dim = 44
        self.action_dim = 19
        self.cmd_dim = 11
        self.clock_dim = 2
        self.privileged_dim = 24
        self.terrain_dim = 221
        self.total_obs_dim = self.proprio_dim + self.action_dim + self.cmd_dim + self.clock_dim
        self.total_privileged_dim = self.privileged_dim + self.terrain_dim + self.total_obs_dim
        print("=" * 100)
        print(self.total_obs_dim, self.total_privileged_dim)
        print("=" * 100)



        # Saving via ReplayBuffer writers (two buffers: constant and switch)
        self.constant_rb_path = os.path.join(self.output_root, "constant.zarr")
        self.switch_rb_path = os.path.join(self.output_root, "switch.zarr")
        print("Save path: ", self.output_root)
        if os.path.exists(self.output_root):
            if getattr(self.args, "overwrite", False):
                print(f"[Collector] Removing existing directory: {self.output_root}")
                shutil.rmtree(self.output_root)
            else:
                print(f"[Collector] Directory exists: {self.output_root} (appending)")
        os.makedirs(self.output_root, exist_ok=True)

        self._stop_event = threading.Event()

        # per-type accumulation before flush
        self._accum_constant: List[Dict[str, Any]] = []
        self._accum_switch: List[Dict[str, Any]] = []

        # create env with 10s episode
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=self.task_name)
        self.env_cfg.env.num_envs = self.num_envs
        self.env_cfg.env.episode_length_s = args.episode_length_s

        # prevent in-episode command resampling; we will control commands manually
        self.env_cfg.commands.resampling_time = args.episode_length_s * 10  # still equals episode length; first step won't hit modulo

        # build env with merged args
        self.args.headless = self.headless or getattr(self.args, "headless", False)
        self.args.num_envs = self.num_envs
        self.env, _ = task_registry.make_env(name=self.task_name, args=self.args, env_cfg=self.env_cfg)
        self.device = self.env.device

        # optional policy for actions
        if self.load_checkpoint:
            try:
                self.train_cfg.runner.resume = True
                self.train_cfg.runner.resume_path = self.load_checkpoint
                ppo_runner, _ = task_registry.make_alg_runner(env=self.env, name=self.task_name, args=self.args, train_cfg=self.train_cfg, log_root=None)
                self.policy = ppo_runner.get_inference_policy(device=self.device)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                raise e

        # queue size small to apply backpressure
        if args.const_prob > 0:
            self.constant_writer = ReplayBufferWriter(self.constant_rb_path, "constant", self._stop_event, max_queue_size=2, small_chunks=args.small_chunks)
            self.constant_writer.start()
        else:
            self.constant_writer = None
        if args.switch_prob > 0:
            self.switch_writer = ReplayBufferWriter(self.switch_rb_path, "switch", self._stop_event, max_queue_size=2, small_chunks=args.small_chunks)
            self.switch_writer.start()
        else:
            self.switch_writer = None
        # reset once to init buffers
        _, _ = self.env.reset()

    def _check_writers(self):
        for w in (self.constant_writer, self.switch_writer):
            if w is None:
                continue
            if getattr(w, "error", None) is not None:
                raise RuntimeError(f"Writer {w.traj_type} crashed") from w.error
            if not w.thread.is_alive() and w._started:
                raise RuntimeError(f"Writer {w.traj_type} unexpectedly stopped")

    def _get_actions(self) -> torch.Tensor:
        with torch.inference_mode():
            obs = self.env.get_observations()
            critic_obs = self.env.get_privileged_observations()
            actions, _ = self.policy.act_inference(obs, privileged_obs=critic_obs)
            return actions

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """Resample commands for specified environments.
        Prefer the environment's curriculum-based sampler if enabled; otherwise
        sample uniformly from configured ranges.
        """
        cfg_ranges = self.env.cfg.commands.ranges
        use_curriculum = getattr(self.env.cfg.commands, "curriculum", False)

        if use_curriculum:
            # Use curriculum grid to sample vx/yaw
            self.env.update_command_curriculum_grid(env_ids)

            # Fill remaining command dimensions (3..9)
            gait_freq = torch_rand_float(cfg_ranges.gait_frequency[0], cfg_ranges.gait_frequency[1], (len(env_ids), 1), device=self.device).squeeze(1)
            phases = torch.tensor([0.0, 0.5], device=self.device)
            rand_idx = torch.randint(0, len(phases), (len(env_ids), ), device=self.device)
            phase = phases[rand_idx]
            swing_height = torch_rand_float(cfg_ranges.foot_swing_height[0], cfg_ranges.foot_swing_height[1], (len(env_ids), 1), device=self.device).squeeze(1)

            # Body pose parameters
            if self.env.cfg.env.observe_body_height:
                body_height = torch_rand_float(cfg_ranges.body_height[0], cfg_ranges.body_height[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                body_height = torch.zeros(len(env_ids), device=self.device)
            if self.env.cfg.env.observe_body_pitch:
                body_pitch = torch_rand_float(cfg_ranges.body_pitch[0], cfg_ranges.body_pitch[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                body_pitch = torch.zeros(len(env_ids), device=self.device)
            if self.env.cfg.env.observe_waist_roll:
                waist_roll = torch_rand_float(cfg_ranges.waist_roll[0], cfg_ranges.waist_roll[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                waist_roll = torch.zeros(len(env_ids), device=self.device)

            # Assign to env commands
            self.env.commands[env_ids, 3] = gait_freq
            self.env.commands[env_ids, 4] = phase
            self.env.commands[env_ids, 5] = 0.5  # duration fixed as in env
            self.env.commands[env_ids, 6] = swing_height
            self.env.commands[env_ids, 7] = body_height
            if self.env.cfg.env.observe_body_pitch:
                self.env.commands[env_ids, 8] = body_pitch
            if self.env.cfg.env.observe_waist_roll:
                self.env.commands[env_ids, 9] = waist_roll

            # Apply high-speed constraints similar to env
            velocity_level = torch.clip(1.0*torch.norm(self.env.commands[env_ids, :2], dim=-1) + 0.5*torch.abs(self.env.commands[env_ids, 2]), min=1)
            high_speed_mask = velocity_level > 1.8
            if high_speed_mask.any():
                self.env.commands[env_ids[high_speed_mask], 3] = torch.clip(self.env.commands[env_ids[high_speed_mask], 3], min=2.0)
                if self.env.cfg.env.observe_body_pitch:
                    self.env.commands[env_ids[high_speed_mask], 8] = torch.clip(self.env.commands[env_ids[high_speed_mask], 8], max=0.3)
                if self.env.cfg.env.observe_waist_roll:
                    self.env.commands[env_ids[high_speed_mask], 9] = torch.clip(self.env.commands[env_ids[high_speed_mask], 9], min=-0.15, max=0.15)

            # Low body height constraints
            if self.env.cfg.env.observe_body_height:
                low_height_mask = self.env.commands[env_ids, 7] < -0.15
                if low_height_mask.any():
                    self.env.commands[env_ids[low_height_mask], 6] = torch.clip(self.env.commands[env_ids[low_height_mask], 6], max=0.20)
                if self.env.cfg.env.observe_body_pitch:
                    lower_height_mask = self.env.commands[env_ids, 7] < -0.2
                    if lower_height_mask.any():
                        self.env.commands[env_ids[lower_height_mask], 8] = torch.clip(self.env.commands[env_ids[lower_height_mask], 8], max=0.3)
        else:
            # Fallback: sample uniformly from configured static ranges (no curriculum)
            lin_vel_x = torch_rand_float(cfg_ranges.lin_vel_x[0], cfg_ranges.lin_vel_x[1], (len(env_ids), 1), device=self.device).squeeze(1)
            lin_vel_y = torch_rand_float(cfg_ranges.lin_vel_y[0], cfg_ranges.lin_vel_y[1], (len(env_ids), 1), device=self.device).squeeze(1)
            ang_vel_yaw = torch_rand_float(cfg_ranges.ang_vel_yaw[0], cfg_ranges.ang_vel_yaw[1], (len(env_ids), 1), device=self.device).squeeze(1)

            gait_freq = torch_rand_float(cfg_ranges.gait_frequency[0], cfg_ranges.gait_frequency[1], (len(env_ids), 1), device=self.device).squeeze(1)
            phases = torch.tensor([0.0, 0.5], device=self.device)
            rand_idx = torch.randint(0, len(phases), (len(env_ids), ), device=self.device)
            phase = phases[rand_idx]
            swing_height = torch_rand_float(cfg_ranges.foot_swing_height[0], cfg_ranges.foot_swing_height[1], (len(env_ids), 1), device=self.device).squeeze(1)

            if self.env.cfg.env.observe_body_height:
                body_height = torch_rand_float(cfg_ranges.body_height[0], cfg_ranges.body_height[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                body_height = torch.zeros(len(env_ids), device=self.device)
            if self.env.cfg.env.observe_body_pitch:
                body_pitch = torch_rand_float(cfg_ranges.body_pitch[0], cfg_ranges.body_pitch[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                body_pitch = torch.zeros(len(env_ids), device=self.device)
            if self.env.cfg.env.observe_waist_roll:
                waist_roll = torch_rand_float(cfg_ranges.waist_roll[0], cfg_ranges.waist_roll[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                waist_roll = torch.zeros(len(env_ids), device=self.device)

            self.env.commands[env_ids, 0] = lin_vel_x
            self.env.commands[env_ids, 1] = lin_vel_y
            self.env.commands[env_ids, 2] = ang_vel_yaw
            self.env.commands[env_ids, 3] = gait_freq
            self.env.commands[env_ids, 4] = phase
            self.env.commands[env_ids, 5] = 0.5
            self.env.commands[env_ids, 6] = swing_height
            self.env.commands[env_ids, 7] = body_height
            if self.env.cfg.env.observe_body_pitch:
                self.env.commands[env_ids, 8] = body_pitch
            if self.env.cfg.env.observe_waist_roll:
                self.env.commands[env_ids, 9] = waist_roll

    def _collect_episode(self, trajectory_type: str) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Collect a single episode for all environments and return episodes list"""
        # Reset environment
        obs, critic_obs = self.env.reset()
        
        # Sample initial commands for all envs
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        self._resample_commands(env_ids)
        
        # Always store the initial commands as cmd_A
        cmd_A = self.env.commands.detach()
        
        # For switch trajectories, prepare second command set
        if trajectory_type == "switch":
            # Sample second command set
            self._resample_commands(env_ids)
            cmd_B = self.env.commands.detach()
            # Reset to first command set
            self.env.commands.copy_(cmd_A)
        else:
            # For constant trajectories, cmd_B is not used
            cmd_B = None

        last_obs_buffers = []
        last_critic_obs_buffers = []
        action_buffers = []
        reward_buffers = []
        done_buffers = []
        env_valid_steps = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
        
        # Track active environments
        active_mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

        # Episode loop
        t = 0
        command_scales = torch.as_tensor(np.array(
            [2, 2, 0.25, 1, 1, 1, 0.15, 2.0, 0.5, 0.5, 1]
        )).to(self.device)
        max_steps = int(self.env_cfg.env.episode_length_s / self.env.dt)
        last_obs, last_critic_obs, _, _, _ = self.env.step(torch.zeros(self.env.num_envs, self.env.num_actions, dtype=torch.float, device=self.env.device))
        assert len(last_critic_obs.shape) == 2
        assert len(last_obs.shape) == 3
        assert last_obs.shape[-1] == self.total_obs_dim
        assert last_critic_obs.shape[-1] == self.total_privileged_dim
        current_commands = cmd_A.detach()
        while t < max_steps and active_mask.any():
            with torch.inference_mode():
                actions, _ = self.policy.act_inference(last_obs, privileged_obs=last_critic_obs)
            
            # Step environment
            obs, critic_obs, step_rewards, dones, infos = self.env.step(actions)
            
            # Switch commands halfway through for switch trajectories
            if trajectory_type == "switch" and t == max_steps // 2:
                current_commands = cmd_B.detach() 
            self.env.commands.copy_(current_commands)

            last_obs_buffers.append(last_obs[:, -1].detach().cpu().numpy())
            last_critic_obs_buffers.append(last_critic_obs[:, last_obs.shape[-1]:].detach().cpu().numpy())
            action_buffers.append(actions.detach().cpu().numpy())
            reward_buffers.append(step_rewards.cpu().numpy())
            done_buffers.append(dones.cpu().numpy())
            
            # # Record data for active environments
            # for eid in range(self.env.num_envs):
            #     if not bool(active_mask[eid]):
            #         continue
            #     last_proprio = last_obs[eid, -1, :self.proprio_dim].detach().cpu().numpy()
            #     last_privileged = last_critic_obs[eid, -self.privileged_dim - self.terrain_dim:-self.terrain_dim].detach().cpu().numpy()
            #     last_terrain = last_critic_obs[eid, -self.terrain_dim:].detach().cpu().numpy()
            #     last_cmd = last_obs[eid, -1, -self.cmd_dim - self.clock_dim:-self.clock_dim].detach().cpu().numpy()
            #     last_clock = last_obs[eid, -1, -self.clock_dim:].detach().cpu().numpy()
            #     proprio_buffers[eid].append(last_proprio)
            #     privileged_buffers[eid].append(last_privileged)
            #     terrain_buffers[eid].append(last_terrain)
            #     cmd_buffers[eid].append(last_cmd)
            #     clock_buffers[eid].append(last_clock)
            #     act_buffers[eid].append(actions[eid].detach().cpu().numpy())
            #     rew_buffers[eid].append(float(step_rewards[eid].item()))
            #     done_buffers[eid].append(bool(dones[eid].item()))
            last_obs = obs.detach()
            last_critic_obs = critic_obs.detach()
            t += 1
            # inactivate envs that are done at this step
            env_valid_steps[active_mask] += 1
            if dones.any():
                active_mask[dones > 0] = False
            # print(env_valid_steps.cpu().numpy())
            # break when all envs finished early
            if (~active_mask).all():
                break

        # finalize: build episodes
        episodes: List[Dict[str, Any]] = []
        saved_rewards: List[float] = []
        last_obs_buffers = np.stack(last_obs_buffers, axis=1)
        last_critic_obs_buffers = np.stack(last_critic_obs_buffers, axis=1)
        action_buffers = np.stack(action_buffers, axis=1)
        reward_buffers = np.stack(reward_buffers, axis=1).astype(np.float32)
        done_buffers = np.stack(done_buffers, axis=1).astype(bool)
        # assert done_buffers.shape == (self.env.num_envs, max_steps)
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

            meta_entry: Dict[str, Any] = {
                "episode_reward": traj_rew,
                "episode_step_reward": traj_step_reward,
            }
            if trajectory_type == "switch":
                meta_entry["episode_command_A"] = cmd_A[eid].cpu().numpy()
                meta_entry["episode_command_B"] = cmd_B[eid].cpu().numpy()
            else:
                meta_entry["episode_command"] = cmd_A[eid].cpu().numpy()

            episodes.append({"data": data, "meta": meta_entry})

        # allow curriculum tick
        self.env.training_curriculum(num_steps=self.curriculum_factor)
        return episodes, saved_rewards

    def _flush_if_needed(self, which: str) -> None:
        if which == "constant" and len(self._accum_constant) >= self.flush_episodes:
            # print queue size before enqueue
            print(f"[Collector] constant queue size before put: {self.constant_writer.queue.qsize()} (accum={len(self._accum_constant)})")
            batch = {"episodes": self._accum_constant[:self.flush_episodes]}
            self.constant_writer.put_batch(batch)
            print(f"[Collector] constant queue size after put: {self.constant_writer.queue.qsize()}")
            self._accum_constant = self._accum_constant[self.flush_episodes:]
        elif which == "switch" and len(self._accum_switch) >= self.flush_episodes:
            print(f"[Collector] switch queue size before put: {self.switch_writer.queue.qsize()} (accum={len(self._accum_switch)})")
            batch = {"episodes": self._accum_switch[:self.flush_episodes]}
            self.switch_writer.put_batch(batch)
            print(f"[Collector] switch queue size after put: {self.switch_writer.queue.qsize()}")
            self._accum_switch = self._accum_switch[self.flush_episodes:]

    def _final_flush(self) -> None:
        if len(self._accum_constant) > 0:
            print(f"[Collector] final flush constant (accum={len(self._accum_constant)}) queue={self.constant_writer.queue.qsize()}")
            self.constant_writer.put_batch({"episodes": self._accum_constant})
            self._accum_constant = []
        if len(self._accum_switch) > 0:
            print(f"[Collector] final flush switch (accum={len(self._accum_switch)}) queue={self.switch_writer.queue.qsize()}")
            self.switch_writer.put_batch({"episodes": self._accum_switch})
            self._accum_switch = []
        # wait writers
        if self.constant_writer is not None:
            self.constant_writer.join()
        if self.switch_writer is not None:
            self.switch_writer.join()

    def collect(self, num_total: int, const_prob: float, switch_prob: float) -> None:
        max_learning_iter = self.max_learning_iter
        total_eps_num = num_total * self.num_envs
        self.curriculum_factor = max_learning_iter / total_eps_num
        assert abs(const_prob + switch_prob - 1.0) < 1e-6
        try:
            produced_total = 0
            produced_const = 0
            produced_switch = 0
            while produced_total < num_total:
                self._check_writers()
                produced_total += 1
                next_type = "constant" if np.random.rand() < const_prob else "switch"
                if next_type == "constant":
                    episodes, saved_rewards = self._collect_episode("constant")
                    print(f"[Collector] collecting constant ({produced_const+1}/{produced_total}/{num_total}) | q_const={self.constant_writer.queue.qsize()} saved_rewards={np.mean(saved_rewards)} saved_rewards_max={np.max(saved_rewards)} saved_rewards_min={np.min(saved_rewards)}")
                    self._accum_constant.extend(episodes)
                    self._flush_if_needed("constant"); self._check_writers()
                    produced_const += 1
                else:
                    episodes, saved_rewards = self._collect_episode("switch")
                    print(f"[Collector] collecting switch ({produced_switch+1}/{produced_total}/{num_total}) | q_switch={self.switch_writer.queue.qsize()} saved_rewards={np.mean(saved_rewards)} saved_rewards_max={np.max(saved_rewards)} saved_rewards_min={np.min(saved_rewards)}")
                    self._accum_switch.extend(episodes)
                    self._flush_if_needed("switch"); self._check_writers()
                    produced_switch += 1
        finally:
            # Signal stop first, then flush and join writers
            self._stop_event.set()
            self._check_writers()
            self._final_flush()

    def __del__(self):
        # signal stop and flush
        try:
            self._stop_event.set()
            self._check_writers()
            self._final_flush()
        except Exception:
            pass


def main():
    args = _parse_merged_args()
    collector = TrajectoryDataCollector(args=args, max_learning_iter=40000)
    collector.collect(args.num_total, args.const_prob, args.switch_prob)


if __name__ == "__main__":
    main()