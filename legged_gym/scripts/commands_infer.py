import os
import numpy as np
import sys
sys.path.append(os.getcwd())
import torch
import tqdm
from legged_gym.dataset.replay_buffer import ReplayBuffer
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import wandb

class MyDataset(Dataset):
    def __init__(self, rb, horizon=5):
        self.data_dict = {}
        keys_to_load = ["proprio", "commands"]
        for key in keys_to_load:
            self.data_dict[key] = rb.data[key][:]
        self.horizon = horizon
        episode_ends = rb.meta['episode_ends'][:]
        # map the index to the episode id
        episode_id = np.repeat(np.arange(len(episode_ends)), np.diff([0, *episode_ends]))
        self.episode_id = episode_id
        self.episode_ends = episode_ends
        self.ep_start_obs = rb.meta['ep_start_obs'][:]
    
    def __len__(self):
        return len(self.data_dict['proprio'])
    
    def __getitem__(self, idx):
        ep_id = self.episode_id[idx]
        ep_start = 0 if ep_id == 0 else self.episode_ends[ep_id - 1]
        horizon_start = max(ep_start, idx - self.horizon + 1)
        horizon_end = idx + 1
        cmd = self.data_dict['commands'][idx]
        # clock = self.data_dict['clock'][horizon_start:horizon_end]
        proprio = self.data_dict['proprio'][horizon_start:horizon_end]
        valid_len = proprio.shape[0]
        # history_action = np.zeros((valid_len, self.data_dict['actions'].shape[-1]))
        # if valid_len > 1:
        #     his_a_start = max(ep_start, idx - self.horizon)
        #     his_a_end = idx
        #     history_len = his_a_end - his_a_start
        #     history_action[-history_len:] = self.data_dict['actions'][his_a_start:his_a_end]
        # obs = np.concatenate([proprio, history_action, cmd, clock], axis=-1)
        obs = proprio
        if valid_len < self.horizon:
            obs = np.concatenate([self.ep_start_obs[ep_id, -(self.horizon - valid_len):, :obs.shape[-1]], obs], axis=0)
        # terrain = self.data_dict['terrain'][idx]
        # privileged = self.data_dict['privileged'][idx]
        # critic_obs = np.concatenate([obs[-1], privileged, terrain], axis=-1)
        # actions = self.data_dict['actions'][idx]
        return obs.astype(np.float32), cmd.astype(np.float32)

class CommandsInfer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )
        
    
    def forward(self, proprio):
        return self.mlp(proprio)

def play():
    rb_path = "/cpfs/user/caozhe/workspace/HugWBC/dataset/collected_single_short_new/constant.zarr"
    rb = ReplayBuffer.create_from_path(rb_path)
    dataset = MyDataset(rb)
    device = "cuda:4"
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.95, 0.05]) 
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=True, num_workers=4)
    mse_criterion = nn.MSELoss()
    cos_sim_criterion = nn.CosineSimilarity()
    obs, cmd = train_dataset[0]
    obs = obs.reshape(-1)
    cmd = cmd.reshape(-1)
    infer_net = CommandsInfer(obs.shape[-1], cmd.shape[-1]).to(device)
    optimizer = torch.optim.Adam(infer_net.parameters(), lr=1e-4)
    num_epochs = 100
    wandb.init(project="commands_infer", name="commands_infer")
    for epoch in range(num_epochs):
        for obs, cmd in tqdm.tqdm(train_dataloader):
            obs = torch.as_tensor(obs).to(device)
            obs = obs.reshape(obs.shape[0], -1)
            cmd = torch.as_tensor(cmd).to(device)
            commands = infer_net(obs)
            loss = mse_criterion(commands, cmd).mean()
            cos_sim = cos_sim_criterion(commands, cmd).mean()
            norm_delta_persent = torch.norm(commands - cmd, p=2, dim=-1) / torch.norm(cmd, p=2, dim=-1)
            norm_delta_persent = norm_delta_persent.mean()
            wandb.log({"mse_loss": loss, "cos_sim": cos_sim, "norm_delta_persent": norm_delta_persent})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = 0
        val_cos_sim = 0
        val_norm_delta_persent = 0
        for obs, cmd in tqdm.tqdm(val_dataloader):
            obs = torch.as_tensor(obs).to(device)
            obs = obs.reshape(obs.shape[0], -1)
            cmd = torch.as_tensor(cmd).to(device)
            commands = infer_net(obs)
            loss = mse_criterion(commands, cmd).mean()
            cos_sim = cos_sim_criterion(commands, cmd).mean()
            norm_delta_persent = torch.norm(commands - cmd, p=2, dim=-1) / torch.norm(cmd, p=2, dim=-1)
            norm_delta_persent = norm_delta_persent.mean()
            val_loss += loss.item()
            val_cos_sim += cos_sim.item()
            val_norm_delta_persent += norm_delta_persent.item()
        wandb.log({"val_loss": val_loss / len(val_dataloader), "val_cos_sim": val_cos_sim / len(val_dataloader), "val_norm_delta_persent": val_norm_delta_persent / len(val_dataloader)})
    wandb.finish()

if __name__ == '__main__':
    play()