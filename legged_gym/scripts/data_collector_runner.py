import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time

import numpy as np

# ensure local repo is importable
sys.path.append(os.getcwd())

from legged_gym.envs import *  # registers tasks
from legged_gym.utils import task_registry
import torch
from isaacgym.torch_utils import torch_rand_float


class TrajectoryDataCollector:
    def __init__(
        self,
        task_name: str = "h1int",
        num_envs: int = 4,
        headless: bool = True,
        load_checkpoint: Optional[str] = None,
        output_root: str = "collected_trajectories_v2",
    ) -> None:
        self.task_name = task_name
        self.num_envs = num_envs
        self.headless = headless
        self.load_checkpoint = load_checkpoint
        self.output_root = output_root

        # Thread pool for concurrent data saving
        num_cpus = os.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=num_cpus, thread_name_prefix="DataSaver")
        self.pending_tasks: List[Future] = []
        self.task_lock = threading.Lock()
        self.max_pending_tasks = num_envs  # Maximum pending tasks before blocking

        self.constant_dir = os.path.join(self.output_root, "constant")
        self.switch_dir = os.path.join(self.output_root, "switch")
        os.makedirs(self.constant_dir, exist_ok=True)
        os.makedirs(self.switch_dir, exist_ok=True)

        self.constant_meta_path = os.path.join(self.constant_dir, "meta_info.json")
        self.switch_meta_path = os.path.join(self.switch_dir, "meta_info.json")
        self.constant_meta: Dict[str, Dict] = self._load_meta(self.constant_meta_path)
        self.switch_meta: Dict[str, Dict] = self._load_meta(self.switch_meta_path)

        # per-env trajectory counters per type
        self.constant_counts: Dict[int, int] = {}
        self.switch_counts: Dict[int, int] = {}

        # create env with 10s episode
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=self.task_name)
        self.env_cfg.env.num_envs = self.num_envs
        self.env_cfg.env.episode_length_s = 10.0

        # prevent in-episode command resampling; we will control commands manually
        self.env_cfg.commands.resampling_time = 10.0  # still equals episode length; first step won't hit modulo

        # build args and env
        from isaacgym import gymutil
        args = gymutil.parse_arguments(description="Trajectory Data Collector", custom_parameters=[
            {"name": "--task", "type": str, "default": self.task_name},
            {"name": "--num_envs", "type": int, "default": self.num_envs},
            {"name": "--headless", "action": "store_true", "default": self.headless},
            # allow extra CLI flags so gymutil doesn't error out
            {"name": "--num_constant", "type": int, "default": 0},
            {"name": "--num_switch", "type": int, "default": 0},
            {"name": "--load_checkpoint", "type": str},
            {"name": "--output_root", "type": str},
        ])
        args.headless = self.headless or getattr(args, "headless", False)
        args.num_envs = self.num_envs

        self.env, _ = task_registry.make_env(name=self.task_name, args=args, env_cfg=self.env_cfg)
        self.device = self.env.device

        # optional policy for actions
        self.policy = None
        if self.load_checkpoint:
            try:
                self.train_cfg.runner.resume = True
                self.train_cfg.runner.resume_path = self.load_checkpoint
                ppo_runner, _ = task_registry.make_alg_runner(env=self.env, name=self.task_name, args=args, train_cfg=self.train_cfg, log_root=None)
                self.policy = ppo_runner.get_inference_policy(device=self.device)
            except Exception:
                self.policy = None

        # reset once to init buffers
        _, _ = self.env.reset()

    def _wait_for_task_queue(self) -> None:
        """Wait until the number of pending tasks is less than num_envs"""
        while True:
            with self.task_lock:
                # Clean up completed tasks
                self.pending_tasks = [task for task in self.pending_tasks if not task.done()]
                if len(self.pending_tasks) < self.max_pending_tasks:
                    break
            # Wait a bit before checking again
            time.sleep(0.01)

    def _submit_save_task(self, data: Dict[str, Any], fpath: str, meta_entry: Dict[str, Any], 
                         trajectory_type: str, fname: str) -> Future:
        """Submit a data saving task to the thread pool"""
        def save_data():
            try:
                np.savez_compressed(fpath, **data)
                return fname, meta_entry
            except Exception as e:
                print(f"Error saving {fpath}: {e}")
                return None

        # Wait if too many pending tasks
        self._wait_for_task_queue()
        
        # Submit task
        with self.task_lock:
            future = self.thread_pool.submit(save_data)
            self.pending_tasks.append(future)
            return future

    def _load_meta(self, path: str) -> Dict[str, Dict]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_meta(self) -> None:
        with open(self.constant_meta_path, "w") as f:
            json.dump(self.constant_meta, f, indent=2)
        with open(self.switch_meta_path, "w") as f:
            json.dump(self.switch_meta, f, indent=2)

    def _get_next_index(self, env_id: int, is_switch: bool) -> int:
        if is_switch:
            if env_id not in self.switch_counts:
                self.switch_counts[env_id] = 0
            self.switch_counts[env_id] += 1
            return self.switch_counts[env_id] - 1
        else:
            if env_id not in self.constant_counts:
                self.constant_counts[env_id] = 0
            self.constant_counts[env_id] += 1
            return self.constant_counts[env_id] - 1

    def _get_actions(self) -> torch.Tensor:
        if self.policy is not None:
            with torch.inference_mode():
                obs = self.env.get_observations()
                critic_obs = self.env.get_privileged_observations()
                actions, _ = self.policy.act_inference(obs, privileged_obs=critic_obs)
                return actions
        return torch.zeros(self.env.num_envs, self.env.num_actions, device=self.env.device)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """Resample commands for specified environments.
        Prefer the environment's curriculum-based sampler if enabled; otherwise
        sample uniformly from configured ranges.
        """
        cfg_ranges = self.env.cfg.commands.ranges
        use_curriculum = getattr(self.env.cfg.commands, "curriculum", False)

        if use_curriculum:
            # Use curriculum grid to sample vx/yaw; it internally sets:
            # - self.commands[env_ids, 0] (vx)
            # - self.commands[env_ids, 1] (vy, limited)
            # - self.commands[env_ids, 2] (yaw)
            # and updates internal curriculum state/weights.
            self.env.update_command_curriculum_grid(env_ids)

            # Fill remaining command dimensions (3..9) consistent with env logic.
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

    def _collect_episode(self, trajectory_type: str) -> Tuple[List[str], List[float]]:
        """Collect a single episode for all environments"""
        # Reset environment
        obs, critic_obs = self.env.reset()
        
        # Sample initial commands for all envs
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        self._resample_commands(env_ids)
        
        # Always store the initial commands as cmd_A
        cmd_A = self.env.commands.clone()
        
        # For switch trajectories, prepare second command set
        if trajectory_type == "switch":
            # Sample second command set
            self._resample_commands(env_ids)
            cmd_B = self.env.commands.clone()
            # Reset to first command set
            self.env.commands = cmd_A.clone()
        else:
            # For constant trajectories, cmd_B is not used
            cmd_B = None
        
        # Initialize buffers
        obs_buffers = [[] for _ in range(self.env.num_envs)]
        critic_obs_buffers = [[] for _ in range(self.env.num_envs)]
        act_buffers = [[] for _ in range(self.env.num_envs)]
        rew_buffers = [[] for _ in range(self.env.num_envs)]
        done_buffers = [[] for _ in range(self.env.num_envs)]
        root_buffers = [[] for _ in range(self.env.num_envs)]
        cmd_buffers = [[] for _ in range(self.env.num_envs)]
        
        # Track active environments
        active_mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        
        # Episode loop
        t = 0
        max_steps = int(self.env_cfg.env.episode_length_s / self.env.dt)
        
        while t < max_steps and active_mask.any():
            # Get actions
            actions = self._get_actions()
            
            # Step environment
            obs, critic_obs, step_rewards, dones, infos = self.env.step(actions)
            
            # Switch commands halfway through for switch trajectories
            if trajectory_type == "switch" and t == max_steps // 2:
                self.env.commands = cmd_B.clone()
            
            # Record data for active environments
            for eid in range(self.env.num_envs):
                if not bool(active_mask[eid]):
                    continue
                obs_buffers[eid].append(obs[eid].detach().cpu().numpy())
                critic_obs_buffers[eid].append(critic_obs[eid].detach().cpu().numpy())
                act_buffers[eid].append(actions[eid].detach().cpu().numpy())
                rew_buffers[eid].append(float(step_rewards[eid].item()))
                done_buffers[eid].append(bool(dones[eid].item()))
                root_buffers[eid].append(self.env.root_states[eid].detach().cpu().numpy())
                cmd_buffers[eid].append(self.env.commands[eid].detach().cpu().numpy())

            t += 1
            # inactivate envs that are done at this step
            if dones.any():
                active_mask[dones > 0] = False
            # break when all envs finished early
            if (~active_mask).all():
                break

        # finalize and save per-env
        saved_files: List[str] = []
        saved_rewards: List[float] = []
        save_futures: List[Future] = []

        for eid in range(self.env.num_envs):
            traj_len = len(obs_buffers[eid])
            traj_rew = float(sum(rew_buffers[eid]))
            saved_rewards.append(traj_rew)

            b_idx = self._get_next_index(eid, is_switch=(trajectory_type == "switch"))
            fname = f"{eid}_{b_idx}_{np.round(traj_rew, 2)}.npz"
            out_dir = self.switch_dir if trajectory_type == "switch" else self.constant_dir
            fpath = os.path.join(out_dir, fname)

            data = {
                "obs": np.array(obs_buffers[eid]),
                "critic_obs": np.array(critic_obs_buffers[eid]),
                "actions": np.array(act_buffers[eid]),
                "rewards": np.array(rew_buffers[eid]),
                "dones": np.array(done_buffers[eid]),
                "root_states": np.array(root_buffers[eid]),
                "commands": np.array(cmd_buffers[eid]),
                "dt": self.env.dt,
            }

            meta_entry = {
                "length": traj_len,
                "reward": traj_rew,
                "commands": [],
            }
            if trajectory_type == "switch":
                meta_entry["commands"].append(cmd_A[eid].detach().cpu().numpy().tolist())
                meta_entry["commands"].append(cmd_B[eid].detach().cpu().numpy().tolist())
            else:
                meta_entry["commands"].append(cmd_A[eid].detach().cpu().numpy().tolist())

            # Submit save task to thread pool
            future = self._submit_save_task(data, fpath, meta_entry, trajectory_type, fname)
            save_futures.append(future)
            saved_files.append(fname)

        # Wait for all save tasks to complete and update metadata
        for future in save_futures:
            try:
                result = future.result()
                if result is not None:
                    fname, meta_entry = result
                    if trajectory_type == "switch":
                        self.switch_meta[fname] = meta_entry
                    else:
                        self.constant_meta[fname] = meta_entry
            except Exception as e:
                print(f"Error in save task: {e}")

        # Remove completed tasks from pending list
        with self.task_lock:
            self.pending_tasks = [task for task in self.pending_tasks if not task.done()]

        # allow curriculum tick
        self.env.training_curriculum()
        return saved_files, saved_rewards

    def collect(self, num_constant: int, num_switch: int) -> None:
        try:
            for _ in range(num_constant):
                self._collect_episode("constant")
                self._save_meta()
            for _ in range(num_switch):
                self._collect_episode("switch")
                self._save_meta()
        finally:
            # Ensure all pending tasks complete before shutdown
            self._wait_for_all_tasks()
            self.thread_pool.shutdown(wait=True)

    def _wait_for_all_tasks(self) -> None:
        """Wait for all pending tasks to complete"""
        while True:
            with self.task_lock:
                self.pending_tasks = [task for task in self.pending_tasks if not task.done()]
                if len(self.pending_tasks) == 0:
                    break
            time.sleep(0.01)

    def __del__(self):
        """Cleanup thread pool on destruction"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trajectory data collector (runner-style)")
    parser.add_argument("--task", type=str, default="h1int")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="collected_trajectories_v2")
    parser.add_argument("--num_constant", type=int, default=10)
    parser.add_argument("--num_switch", type=int, default=10)
    args = parser.parse_args()

    collector = TrajectoryDataCollector(
        task_name=args.task,
        num_envs=args.num_envs,
        headless=args.headless,
        load_checkpoint=args.load_checkpoint,
        output_root=args.output_root,
    )
    collector.collect(args.num_constant, args.num_switch)


if __name__ == "__main__":
    main()


