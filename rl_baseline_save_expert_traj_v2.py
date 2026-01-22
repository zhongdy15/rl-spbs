from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from bdq.bdq import BDQ
import SemiPhysBuildingSim
import gym
import numpy as np
import matplotlib.pyplot as plt
from rl_zoo3.wrappers import FrameSkip, ObsHistoryWrapper, ActionMasker
import os
import datetime
from interpret_obs import interpret_obs

# ==========================================
# 1. 配置与初始化
# ==========================================
algo_classes = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "bdq": BDQ}

# 设定保存路径 (以你提供的最新路径为准)
save_folder = 'D:\\research\\rl-spbs\\rl-spbs\\knn_sft_dataset_construction'
rl_baseline_251105 = {
    "bdq": "D:\\research\\remote_logs\\260119_llmrl_a2cppo复现\\bdq_Baseline_OCC_PPD_with_energy_10_2026-01-19-10-51-25\\bdq\\SemiPhysBuildingSim-v0_1",
}

seed_num = 5

# 全局容器：用于存储最终 expert_trajectories.npz 的数据
collected_observations = []
collected_actions = []

# 检查并创建保存目录
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ==========================================
# 2. 主循环：遍历模型与Seed
# ==========================================
for test_model_key_base in rl_baseline_251105.keys():
    for i in range(seed_num):
        # ----------------------------------
        # 2.1 模型加载
        # ----------------------------------
        model_dir_base = rl_baseline_251105[test_model_key_base]
        model_dir = model_dir_base[:-2] + f"_{i + 1}"
        test_model_key = f"{test_model_key_base}_{i + 1}"

        # 加载模型
        model_path = os.path.join(model_dir, "best_model.zip")
        model = algo_classes[test_model_key_base].load(model_path)

        print(f"\nProcessing Seed {i + 1}/{seed_num}")
        print("Loading model Successfully: " + model_dir)

        # ----------------------------------
        # 2.2 环境设置
        # ----------------------------------
        reward_mode = "Baseline_OCC_PPD_with_energy"
        tradeoff_constant = 10
        frame_skip = 5

        env1 = gym.make("SemiPhysBuildingSim-v0",
                        reward_mode=reward_mode,
                        tradeoff_constant=tradeoff_constant,
                        eval_mode=True)

        env1 = ObsHistoryWrapper(env1, horizon=frame_skip)
        env1 = FrameSkip(env1, skip=frame_skip)

        print("Frame skip: " + str(frame_skip))

        # ----------------------------------
        # 2.3 运行 Episode 并收集数据
        # ----------------------------------
        # 本地列表，用于当前episode的绘图逻辑辅助（虽然主要用 data_recorder）
        current_episode_actions = []

        obs = env1.reset()
        rewards = 0
        done = False
        step_i = 0

        while not done:
            step_i += 1

            # 模型预测 (Expert Action)
            action, _state = model.predict(obs, deterministic=True)
            action = np.array(action)

            # >>>>>>>> 核心收集点 (保存至 .npz) <<<<<<<<
            # 保存 T 时刻的 Obs (Flattened 180 dim) 和 T 时刻决定的 Action
            collected_observations.append(obs.flatten())
            collected_actions.append(action)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # 保存至本地列表 (用于绘图逻辑)
            current_episode_actions.append(action)

            # 环境步进
            obs, r, done, info = env1.step(action)
            rewards += r

            # 调试打印 (仅第一步)
            if step_i == 1:
                print(f"Debug - Seed {i + 1} Step 1 Obs Shape: {obs.shape}")
                print(f"Debug - Seed {i + 1} Step 1 Action: {action}")

        print("Episode Rewards:" + str(rewards))

        # ==========================================
        # 3. 绘图逻辑 (针对当前 Episode)
        # ==========================================
        # 确保 action_list 是 numpy array，绘图代码可能依赖它
        binary_data = np.array(current_episode_actions)

        # 尝试获取 data_recorder (通常在 ObsHistoryWrapper 下需要访问 env 或直接属性)
        # 如果报错 AttributeError，请尝试使用 env1.unwrapped.data_recorder
        data_recorder = env1.data_recorder

        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle(f"Test Result - {test_model_key}")
        axes = axes.flatten()

        outdoor_temp = data_recorder["sensor_outdoor"]["outdoor_temp"]

        # --- 房间数据绘图 (Room 1-7) ---
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

            # 双轴画人数
            ax_twin = ax.twinx()
            ax_twin.plot(occupancy_total, linestyle='-', color='k', label='total', alpha=1.0)
            ax_twin.set_ylabel('Occupancy (People)')
            ax_twin.set_ylim(0, 11)
            ax_twin.yaxis.set_ticks(range(0, 5, 1))

            if r_idx == 6:
                ax.legend(loc='upper left')
                ax_twin.legend(loc='upper right')

        # --- 训练指标绘图 ---

        # Reward
        ax2 = axes[8]
        reward_data = data_recorder["training"]["reward"]
        total_reward = np.sum(reward_data)
        ax2.plot(reward_data, marker='o', linestyle='-', color='g', label='Reward')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Reward')
        ax2.set_title(f'Mode: {reward_mode}\nC: {tradeoff_constant} Total R: {round(total_reward, 1)}')
        ax2.legend()
        ax2.grid(True, linestyle='--', linewidth=0.5, color='gray')

        # FCU Power
        ax3 = axes[9]
        FCU_power = data_recorder["training"]["energy_consumption"]
        FCU_power_total = np.sum(FCU_power)
        ax3.plot(FCU_power, marker='o', linestyle='-', color='g', label='FCU Power')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('FCU Power')
        ax3.set_title(f'Total FCU Power: {FCU_power_total:.2f}')
        ax3.legend()
        ax3.grid(True, linestyle='--', linewidth=0.5, color='gray')

        # PMV Mean
        ax4 = axes[10]
        pmv_mean = data_recorder["training"]["mean_pmv"]
        pmv_mean_avarege = np.mean(pmv_mean)
        ax4.plot(pmv_mean, marker='o', linestyle='-', color='g', label='PMV Mean')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('ABS PMV Mean')
        ax4.set_title(f'PMV Mean: {round(pmv_mean_avarege, 2)}')
        ax4.legend()
        ax4.grid(True, linestyle='--', linewidth=0.5, color='gray')

        # PPD Mean
        ax5 = axes[11]
        ppd_mean = data_recorder["training"]["mean_ppd"]
        ppd_mean_avarege = np.mean(ppd_mean)
        ax5.plot(ppd_mean, marker='o', linestyle='-', color='g', label='PPD Mean')
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('PPD Mean')
        ax5.set_title(f'PPD Mean: {round(ppd_mean_avarege, 2)}')
        ax5.legend()
        ax5.grid(True, linestyle='--', linewidth=0.5, color='gray')

        plt.tight_layout()

        # --- 保存图片 ---
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_filename = f"{test_model_key}_{time_str}.png"
        plot_path = os.path.join(save_folder, plot_filename)

        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close(fig)  # 关闭图形释放内存

        # 关闭环境
        env1.close()

# ==========================================
# 4. 保存 expert_trajectories.npz
# ==========================================
final_obs = np.array(collected_observations)
final_actions = np.array(collected_actions)

print("\n" + "=" * 30)
print("Data Collection Finished")
print("=" * 30)
print(f"Final Obs Shape:    {final_obs.shape}")
print(f"Final Actions Shape: {final_actions.shape}")

# 验证维度
if final_obs.shape[1] != 180:
    print(f"Warning: Obs dimension is {final_obs.shape[1]}, expected 180.")
if final_actions.shape[1] != 7:
    print(f"Warning: Action dimension is {final_actions.shape[1]}, expected 7.")

output_filename = "expert_trajectories.npz"
output_path = os.path.join(save_folder, output_filename)
np.savez(output_path, obs=final_obs, actions=final_actions)

print(f"\nNPZ Saved successfully to: {os.path.abspath(output_path)}")
