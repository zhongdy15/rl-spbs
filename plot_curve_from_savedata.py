import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# === 配置部分 ===
# 【请在这里填写第一步生成的pkl文件路径】
target_file = 'enjoy_data/llm_am_random_baseline_251126/Qwen2.5-72B-Instruct/2025-11-26_15-45-18.pkl'

# 为了方便，如果不知道具体文件名，这里自动查找文件夹里最新的pkl文件
save_folder = os.path.dirname(target_file)

print(f"Loading data from: {target_file}")
with open(target_file, 'rb') as f:
    all_episodes_data = pickle.load(f)

# 遍历每个回合的数据进行画图
for episode_data in all_episodes_data:

    # 解包数据
    idx = episode_data["episode_idx"]
    data_recorder = episode_data["data_recorder"]
    action_list = episode_data["action_list"]
    reward_mode = episode_data["reward_mode"]
    tradeoff_constant = episode_data["tradeoff_constant"]
    test_model_key = episode_data["test_model_key"]
    # 如果需要查看 action mask，可以访问 episode_data["step_records"][step]["mask_indices"]

    print(f"Plotting Episode {idx}...")

    # 准备绘图数据
    binary_data = action_list  # 这里直接使用保存的 array
    outdoor_temp = data_recorder["sensor_outdoor"]["outdoor_temp"]

    # === 原画图逻辑开始 ===
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))  # 3 rows, 4 columns
    fig.suptitle(f"Test Episode {idx} - {test_model_key}")

    axes = axes.flatten()

    for i in range(7):
        ax = axes[i]
        room_str = "room" + str(i + 1)

        room_temp = data_recorder[room_str]["room_temp"]
        occupancy = data_recorder[room_str]["occupant_num"]

        # 兼容原本的数据结构，如果是字典列表则取值，如果是列表则直接用
        if len(occupancy) > 0 and isinstance(occupancy[0], dict):
            occupancy_total = [occupancy[t]["sitting"] + occupancy[t]["standing"] + occupancy[t]["walking"] for t in
                               range(len(occupancy))]
        else:
            occupancy_total = [occupancy[t] for t in range(len(occupancy))]

        # FCU_power = data_recorder[room_str]["FCU_power"] # 原代码未使用，保留

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

        # Twin axis for occupancy
        ax_twin = ax.twinx()
        ax_twin.plot(occupancy_total, linestyle='-', color='k', label='total', alpha=1.0)
        ax_twin.set_ylabel('Occupancy (People)')
        ax_twin.set_ylim(0, 11)
        ax_twin.yaxis.set_ticks(range(0, 5, 1))

        if i == 6:
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

    # Subplot: Reward
    ax2 = axes[8]
    reward = data_recorder["training"]["reward"]
    total_reward = np.sum(reward)
    ax2.plot(reward, marker='o', linestyle='-', color='g', label='Reward')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Reward')
    ax2.set_title('Mode: ' + reward_mode + " C: " + str(tradeoff_constant) + " Total R: " + str(round(total_reward, 1)))
    ax2.legend()
    ax2.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Subplot: FCU Power
    ax3 = axes[9]
    FCU_power = data_recorder["training"]["energy_consumption"]
    FCU_power_total = np.sum(FCU_power)
    ax3.plot(FCU_power, marker='o', linestyle='-', color='g', label='FCU Power')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('FCU Power')
    ax3.set_title('Total FCU Power: ' + str(FCU_power_total))
    ax3.legend()
    ax3.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Subplot: PMV Mean
    ax4 = axes[10]
    pmv_mean = data_recorder["training"]["mean_pmv"]
    pmv_mean_avarege = np.mean(pmv_mean)
    ax4.plot(pmv_mean, marker='o', linestyle='-', color='g', label='PMV Mean')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('ABS PMV Mean')
    ax4.set_title('PMV Mean: ' + str(round(pmv_mean_avarege, 2)))
    ax4.legend()
    ax4.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Subplot: PPD Mean
    ax5 = axes[11]
    ppd_mean = data_recorder["training"]["mean_ppd"]
    ppd_mean_avarege = np.mean(ppd_mean)
    ax5.plot(ppd_mean, marker='o', linestyle='-', color='g', label='PPD Mean')
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('PPD Mean')
    ax5.set_title('PPD Mean: ' + str(round(ppd_mean_avarege, 2)))
    ax5.legend()
    ax5.grid(True, linestyle='--', linewidth=0.5, color='gray')

    plt.tight_layout()

    # === 保存图片 ===
    # 为了避免覆盖，在文件名中加入 episode 编号
    image_filename = os.path.basename(target_file).replace('.pkl', f'_ep{idx}.png')
    save_path = os.path.join(save_folder, image_filename)

    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    # plt.show() # 如果需要在窗口显示，取消注释
    plt.close()  # 关闭当前图表，释放内存
