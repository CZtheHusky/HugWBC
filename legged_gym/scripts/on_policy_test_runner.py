import os
import sys

# ensure isaacgym is imported before torch
import isaacgym  # noqa: F401

sys.path.append(os.getcwd())
from legged_gym.envs import *  # registers tasks
from legged_gym.utils import task_registry
import torch


class OnPolicyTestRunner:
    def __init__(self, task: str, num_envs: int, headless: bool = True, load_checkpoint: str = None) -> None:
        from isaacgym import gymutil
        args = gymutil.parse_arguments(description="OnPolicyTestRunner (no learning)", custom_parameters=[
            {"name": "--task", "type": str, "default": task},
            {"name": "--num_envs", "type": int, "default": num_envs},
            {"name": "--headless", "action": "store_true", "default": headless},
            {"name": "--episodes_per_env", "type": int, "default": 1},
        ])
        args.headless = headless or getattr(args, "headless", False)
        args.num_envs = num_envs

        # make env, and enforce 10s episode to match collector
        self.env, self.env_cfg = task_registry.make_env(name=task, args=args)
        try:
            self.env_cfg.env.episode_length_s = 10.0
        except Exception:
            pass
        self.device = self.env.device

        # optional policy
        self.policy = None
        if load_checkpoint:
            try:
                env = self.env
                train_runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args)
                # if a path is given, try use it
                if os.path.isfile(load_checkpoint):
                    train_runner.load(load_checkpoint)
                self.policy = train_runner.get_inference_policy(device=self.device)
            except Exception:
                self.policy = None

        _, _ = self.env.reset()

    def _act(self, obs: torch.Tensor) -> torch.Tensor:
        if self.policy is not None:
            with torch.no_grad():
                return self.policy.act_inference(obs)
        return torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device)

    def rollout(self, num_episodes_per_env: int = 2, print_prefix: str = "") -> None:
        # track per-env episode order and reward
        episode_order = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        episode_reward = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        finished_counts = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

        total_needed = num_episodes_per_env * self.env.num_envs
        done_episodes = 0

        obs = self.env.get_observations()
        critic_obs = self.env.get_privileged_observations()

        while done_episodes < total_needed:
            actions = self._act(obs)
            step_result = self.env.step(actions)
            if len(step_result) == 4:
                obs, rewards, dones, infos = step_result
                critic_obs = self.env.obs_buf
            else:
                obs, critic_obs, rewards, dones, infos = step_result

            episode_reward += rewards.squeeze(-1) if rewards.dim() > 1 else rewards

            if dones.any():
                env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                for eid in env_ids.tolist():
                    order = int(episode_order[eid].item())
                    rew = float(episode_reward[eid].item())
                    print(f"{print_prefix}env={eid}, ep={order}, reward={rew}")
                    episode_order[eid] += 1
                    finished_counts[eid] += 1
                    done_episodes += 1
                    episode_reward[eid] = 0.0

                # let env handle resets and curriculum
                self.env.training_curriculum()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run rollouts without learning and print per-env rewards")
    parser.add_argument("--task", type=str, default="h1int")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episodes_per_env", type=int, default=2)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    args = parser.parse_args()

    runner = OnPolicyTestRunner(task=args.task, num_envs=args.num_envs, headless=args.headless, load_checkpoint=args.load_checkpoint)
    runner.rollout(num_episodes_per_env=args.episodes_per_env)


if __name__ == "__main__":
    main()


