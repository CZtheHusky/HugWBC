import sys, os
from tqdm import tqdm
sys.path.append(os.getcwd())
from legged_gym.dataset.replay_buffer import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
from legged_gym.hilbert_utils.humanoid_utils import get_cosine_sim, calc_phi_loss_upper, collect_nonzero_losses, collect_nonzero_losses_with_idx, compute_global_color_limits, plot_matrix_heatmaps, plot_tripanel_heatmaps_with_line, calc_z_vector_matrixes, plot_per_step_z


def _to_time_feature(arr: np.ndarray) -> np.ndarray:
    """确保输入是 (T, F)。把除时间维外的维度都摊平。"""
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr

def check_latent_feature(rb_path: str, res_save_parent: str, dirn: str, max_T_heatmap: int = None, horizon=5):
    rb = ReplayBuffer.create_from_path(rb_path)
    os.makedirs(res_save_parent, exist_ok=True)
    # cos_sim_path = os.path.join(res_save_parent, "latent_sim")
    # os.makedirs(cos_sim_path, exist_ok=True)
    # phi_loss_path = os.path.join(res_save_parent, "phi_loss")
    # os.makedirs(phi_loss_path, exist_ok=True)
    goal_raw_cos_sim_save_parent = os.path.join(res_save_parent, "goal_raw_cos_sim")
    goal_z_cosine_sim_save_parent = os.path.join(res_save_parent, "goal_z_cosine_sim")
    os.makedirs(goal_raw_cos_sim_save_parent, exist_ok=True)
    os.makedirs(goal_z_cosine_sim_save_parent, exist_ok=True)


    episode_ends = rb.meta.episode_ends[:]
    latent_data  = rb.data.latent[:]      # 可能是 (T, z) 或 (T, ..., z)
    raw_state_data = rb.data.proprio[:]   # 可能是 (T, d) 或 (T, ..., d)
    ep_start_obs = rb.meta.ep_start_obs[:]

    # # 全局收集
    # phi_losses = []
    # phi_losses_with_idx = []
    # phi_loss_mats = []        # 各 episode 的 φ-loss 矩阵
    # phi_loss_ep_ids = []      # 对应的 episode id

    for eid in tqdm(range(len(episode_ends))[:20]):
        ep_end   = episode_ends[eid]
        ep_start = episode_ends[eid - 1] if eid > 0 else 0
        print(f"eid: {eid}, ep_start: {ep_start}, ep_end: {ep_end}")
        if ep_end - ep_start < 2:
            continue

        # --- 保证 (T, F) ---
        latent    = _to_time_feature(np.asarray(latent_data[ep_start:ep_end]))
        episode_reward = np.round(rb.meta.episode_reward[eid], 2)
        mean_step_reward = np.round(rb.meta.episode_step_reward[eid], 4)

        raw_state = np.concatenate([ep_start_obs[eid, :, :raw_state_data.shape[-1]], raw_state_data[ep_start:ep_end]], axis=0)

        raw_state = _to_time_feature(np.asarray(raw_state)) # EP_LEN D
        # take sliding window of horizon from raw_state, EP_LEN D -> EP_LEN Horizon D
        raw_index = np.arange(raw_state.shape[0] - horizon + 1)
        horizon_index = raw_index[:, None] + np.arange(horizon)
        raw_state = raw_state[horizon_index]
        raw_state = raw_state.reshape(raw_state.shape[0], -1)
    
        goal_z_cosine_sim_list, goal_distance_list, goal_absdist_list = calc_z_vector_matrixes(latent, goal_vector=latent[-1])

        middle_goal_z_cosine_sim_list, middle_goal_distance_list, middle_goal_absdist_list = calc_z_vector_matrixes(latent, goal_vector=latent[len(latent)//2])

        goal_raw_cosine_sim_list, goal_raw_distance_list, goal_raw_absdist_list = calc_z_vector_matrixes(raw_state, goal_vector=raw_state[-1])
        middle_goal_raw_cosine_sim_list, middle_goal_raw_distance_list, middle_goal_raw_absdist_list = calc_z_vector_matrixes(raw_state, goal_vector=raw_state[len(raw_state)//2])

        # plot_per_step_z(
        #     latent,
        #     eid,
        #     cos_sim_path,
        #     dirn,
        #     raw_state=raw_state,
        # )

        # --- 仅计算并收集 φ-loss，不在这里画 ---
        # phi_loss_latent = calc_phi_loss_upper(latent)
        # loss_list = collect_nonzero_losses(phi_loss_latent, eps=1e-8)
        # loss_list_with_idx = collect_nonzero_losses_with_idx(phi_loss_latent, eps=1e-8)

        # phi_losses_with_idx.extend(loss_list_with_idx)
        # phi_losses.extend(loss_list)
        # phi_loss_mats.append(phi_loss_latent)
        # phi_loss_ep_ids.append(eid)
        print("mean step reward: ", mean_step_reward)
        # print("Rounded mean step reward: ", np.round(mean_step_reward, 4))
        plot_tripanel_heatmaps_with_line(
            goal_z_cosine_sim_list,
            goal_distance_list,
            goal_absdist_list,
            [f"goal_z_{eid}={goal_z_cosine_sim_list[g_idx].shape[-1]}_{episode_reward}_{mean_step_reward:.4f}.png" for g_idx in range(len(goal_z_cosine_sim_list))],
            goal_z_cosine_sim_save_parent,
            title_cos=f'{dirn}_{eid} Z Cosine Similarity',
            title_dist=f'{eid} Latent Space Distance',
        )
        plot_tripanel_heatmaps_with_line(
            goal_raw_cosine_sim_list,
            goal_raw_distance_list,
            goal_raw_absdist_list,
            [f"goal_raw_{eid}={goal_raw_cosine_sim_list[g_idx].shape[-1]}_{episode_reward}_{mean_step_reward:.4f}.png" for g_idx in range(len(goal_raw_cosine_sim_list))],
            goal_raw_cos_sim_save_parent,
            title_cos=f'{dirn}_{eid} Z Cosine Similarity',
            title_dist=f'{dirn}_{eid} Latent Space Distance',
        )
        plot_tripanel_heatmaps_with_line(
            middle_goal_z_cosine_sim_list,
            middle_goal_distance_list,
            middle_goal_absdist_list,
            [f"middle_goal_z_{eid}={middle_goal_z_cosine_sim_list[g_idx].shape[-1]}_{episode_reward}_{mean_step_reward:.4f}.png" for g_idx in range(len(middle_goal_z_cosine_sim_list))],
            goal_z_cosine_sim_save_parent,
            title_cos=f'{dirn}_{eid} Z Cosine Similarity',
            title_dist=f'{dirn}_{eid} Latent Space Distance',
        )
        plot_tripanel_heatmaps_with_line(
            middle_goal_raw_cosine_sim_list,
            middle_goal_raw_distance_list,
            middle_goal_raw_absdist_list,
            [f"middle_goal_raw_{eid}={middle_goal_raw_cosine_sim_list[g_idx].shape[-1]}_{episode_reward}_{mean_step_reward:.4f}.png" for g_idx in range(len(middle_goal_raw_cosine_sim_list))],
            goal_raw_cos_sim_save_parent,
            title_cos=f'{dirn}_{eid} Z Cosine Similarity',
            title_dist=f'{dirn}_{eid} Latent Space Distance',
        )
    # # === 循环结束后：先算统一 vmin/vmax，再统一绘图 ===
    # vmin, vmax = compute_global_color_limits(phi_loss_mats, lower_q=5, upper_q=95)
    # plot_matrix_heatmaps(
    #     mats=phi_loss_mats,
    #     names=phi_loss_ep_ids,
    #     out_dir=phi_loss_path,
    #     vmin=vmin, vmax=vmax,
    #     max_T=max_T_heatmap,
    #     cmap_name="coolwarm",
    # )

    # # === 汇总统计 & 保存 ===
    # np.save(f"{res_save_parent}/phi_losses.npy", np.array(phi_losses))
    # np.save(f"{res_save_parent}/phi_losses_with_idx.npy", np.array(phi_losses_with_idx))
    # print("="*20)
    # if len(phi_losses) > 0:
    #     print(f"mean_phi_losses: {np.mean(phi_losses)}")
    #     print(f"max_phi_losses:  {np.max(phi_losses)}")
    #     print(f"min_phi_losses:  {np.min(phi_losses)}")
    #     print(f"std_phi_losses:  {np.std(phi_losses)}")
    #     print(f"mse_phi_losses: {np.mean(np.array(phi_losses) ** 2)}")
    # else:
    #     print("No phi losses collected.")
    # print("="*20)

# check_latent_feature("dataset/example_trajectories/crab_left_walk.zarr", "dataset/latent_check/constant")
# check_latent_feature("dataset/latent_test/switch.zarr",   "dataset/latent_check/switch")
parent_path = "/root/workspace/HugWBC/dataset/example_trajectories_test"
for dirn in os.listdir(parent_path):
    if os.path.isdir(os.path.join(parent_path, dirn)):
        horizons = [5, 1]
        for horizon in horizons:
            check_latent_feature(os.path.join(parent_path, dirn), os.path.join(f"dataset/latent_check_{horizon}", dirn), dirn, horizon=horizon)
