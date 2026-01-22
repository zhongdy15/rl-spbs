import gym
import numpy as np
import matplotlib.pyplot as plt
from rl_zoo3.wrappers import FrameSkip, ObsHistoryWrapper
from sklearn.neighbors import NearestNeighbors
import os
import datetime
import SemiPhysBuildingSim

# ================= 配置区域 =================
# 请修改为你的 npz 文件绝对路径
# knn_data_path = "knn_sft_dataset_construction/output3_obs_with_valid_mask_N50_ep0.00.npz"
# test_model_key = "BDQexpert"
knn_data_path = "sft_construction/sft_data_v9/obs_with_valid_mask_N50_ep0.00.npz"
test_model_key = "HUMANexpert"

save_folder = 'rl_baseline_0122_randomVSmask'
reward_mode = "Baseline_OCC_PPD_with_energy"
tradeoff_constant = 10
frame_skip = 5
seed_num = 1
# ===========================================

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 1. 加载专家数据并构建 KNN
print(f"Loading KNN data from {knn_data_path}...")
try:
    data = np.load(knn_data_path)
    expert_obs = data["obs"]  # 假设 shape 为 (N, obs_dim)
    expert_mask = data["valid_mask"]  # 假设 shape 为 (N, 7, 4)
    print(f"Data loaded. Obs shape: {expert_obs.shape}, Mask shape: {expert_mask.shape}")
except FileNotFoundError:
    print("错误: 找不到 npz 文件，请检查路径。")
    exit()

print("Building KNN model...")
# 使用欧式距离查找最近邻
knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
knn.fit(expert_obs)


def get_masked_random_action(current_obs, knn_model, expert_masks):
    """
    根据当前 obs 找到最近的专家 obs，获取其 mask，
    然后在 mask 允许的动作范围内随机采样。
    """
    # sklearn 需要 (n_samples, n_features) 形状
    # 注意：如果环境用了 ObsHistoryWrapper，obs 可能是平铺的一维数组，直接 reshape(1, -1) 即可
    query_obs = current_obs.reshape(1, -1)

    # 找到最近邻的索引
    _, indices = knn_model.kneighbors(query_obs)
    idx = indices[0][0]

    # 获取对应的 mask (7个房间, 每个房间4个动作)
    # 假设 mask: 1 代表可行，0 代表不可行
    mask = expert_masks[idx]

    action = []
    for room_i in range(7):
        room_mask = mask[room_i]  # e.g., [1, 1, 0, 0]
        valid_indices = np.where(room_mask == 1)[0]

        if len(valid_indices) == 0:
            # 极少数情况 mask 全为 0 (取决于数据清洗质量)，默认给 0 (关)
            action.append(0)
        else:
            # 在可行范围内随机选择
            action.append(np.random.choice(valid_indices))

    return np.array(action)


print(f"Starting KNN Masked verification. Results saved to {save_folder}")

for i in range(seed_num):
    env = gym.make("SemiPhysBuildingSim-v0",
                   reward_mode=reward_mode,
                   tradeoff_constant=tradeoff_constant,
                   eval_mode=True)

    env = ObsHistoryWrapper(env, horizon=frame_skip)
    env = FrameSkip(env, skip=frame_skip)

    np.random.seed(i)

    print(f"Run {i + 1}/{seed_num}...")

    action_list = []
    obs = env.reset()
    rewards = 0
    done = False
    step_cnt = 0

    while not done:
        step_cnt += 1

        # core: 使用 KNN Masked Random 策略
        action = get_masked_random_action(obs, knn, expert_mask)

        action_list.append(action)
        obs, r, done, info = env.step(action)
        rewards += r

    print(f"Episode finished. Total Reward: {rewards}")

    # --- 以下画图代码与脚本 1 完全一致 ---
    binary_data = np.array(action_list)
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle(f"KNN Masked Random Strategy - Run {i + 1}")
    axes = axes.flatten()

    data_recorder = env.data_recorder
    outdoor_temp = data_recorder["sensor_outdoor"]["outdoor_temp"]

    for r_idx in range(7):
        ax = axes[r_idx]
        room_str = "room" + str(r_idx + 1)
        room_temp = data_recorder[room_str]["room_temp"]
        occupancy = data_recorder[room_str]["occupant_num"]
        occupancy_total = [occupancy[t] for t in range(len(occupancy))]

        ax.plot(room_temp, marker='o', linestyle='-', color='b', label='Temperature')
        ax.plot(outdoor_temp, marker='o', linestyle='-', color='r', label='Outdoor Temperature')

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.set_title(room_str)
        ax.set_ylim(19, 31)
        ax.set_xlim(-20, 620)
        ax.yaxis.set_ticks(range(20, 30, 1))
        ax.xaxis.set_ticks(range(0, 600, 60))
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        ax_twin = ax.twinx()
        ax_twin.plot(occupancy_total, linestyle='-', color='k', label='total', alpha=1.0)
        ax_twin.set_ylabel('Occupancy')
        ax_twin.set_ylim(0, 11)
        ax_twin.yaxis.set_ticks(range(0, 5, 1))

        if r_idx == 6:
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

    ax2 = axes[8]
    reward_hist = data_recorder["training"]["reward"]
    ax2.plot(reward_hist, marker='o', linestyle='-', color='g', label='Reward')
    ax2.set_title(f'Total R: {round(np.sum(reward_hist), 1)}')
    ax2.grid(True)

    ax3 = axes[9]
    FCU_power = data_recorder["training"]["energy_consumption"]
    ax3.plot(FCU_power, marker='o', linestyle='-', color='g', label='FCU Power')
    ax3.set_title(f'Total Power: {np.sum(FCU_power):.1f}')
    ax3.grid(True)

    ax4 = axes[10]
    pmv_mean = data_recorder["training"]["mean_pmv"]
    ax4.plot(pmv_mean, marker='o', linestyle='-', color='g', label='PMV Mean')
    ax4.set_title(f'Avg PMV: {np.mean(pmv_mean):.2f}')
    ax4.grid(True)

    ax5 = axes[11]
    ppd_mean = data_recorder["training"]["mean_ppd"]
    ax5.plot(ppd_mean, marker='o', linestyle='-', color='g', label='PPD Mean')
    ax5.set_title(f'Avg PPD: {np.mean(ppd_mean):.2f}')
    ax5.grid(True)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_folder, f"KNN_{test_model_key}_{timestamp}.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    env.close()
