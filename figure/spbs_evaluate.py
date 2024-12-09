from stable_baselines3 import DQN
import SemiPhysBuildingSim
import gym
import numpy as np
import matplotlib.pyplot as plt

# 扩增了状态空间之后的模型
model_dict_2 = {"1203_Baseline_with_energy_constant100": "logs/dqn_2024-12-02-20-52-04/dqn/SemiPhysBuildingSim-v0_1",
                "1203_Baseline_with_energy_constant10": "logs/dqn_2024-12-02-20-52-59/dqn/SemiPhysBuildingSim-v0_1",}

reward_mode_list = ["Baseline_without_energy",
                    "Baseline_with_energy",
                    "Baseline_OCC_PPD_without_energy",
                    "Baseline_OCC_PPD_with_energy",]

model_key = "1203_Baseline_with_energy_constant10"
print("Loading model: " + model_key)
model = DQN.load( model_dict_2[model_key]+"/best_model.zip")

# 加载环境
reward_mode = "Baseline_with_energy"
tradeoff_constant = 10
env1 = gym.make("SemiPhysBuildingSim-v0", reward_mode=reward_mode, tradeoff_constant=tradeoff_constant)

for _ in range(1):
    action_list = []
    obs = env1.reset()
    rewards = 0
    done = False
    i = 0
    while not done:
        i += 1
        action , _state = model.predict(obs)
        action_list.append(action)
        obs, r, done, info = env1.step(action)
        rewards += r
    print("rewards:" + str(rewards))

binary_data = np.array([[int(bit) for bit in f"{value:07b}"] for value in action_list])

fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # 3 rows, 4 columns
# Set the title for the entire figure
fig.suptitle("Model: " + model_key, fontsize=16)

axes = axes.flatten()  # Flatten the 2D array of axes to easily index each subplot



data_recorder = env1.data_recorder
outdoor_temp = data_recorder["sensor_outdoor"]["outdoor_temp"]


for i in range(7):
    ax = axes[i]
    room_str = "room" + str(i+1)

    room_temp = data_recorder[room_str]["room_temp"]

    on_times = np.where(binary_data[:, 7 - i - 1] == 1)[0]

    occupancy = data_recorder[room_str]["occupant_num"]
    occupancy_sitting = [ occupancy[t]["sitting"] for t in range(len(occupancy))]
    occupancy_standing = [ occupancy[t]["standing"] for t in range(len(occupancy))]
    occupancy_walking = [ occupancy[t]["walking"] for t in range(len(occupancy))]
    occupancy_total = [ occupancy_sitting[t] + occupancy_standing[t] + occupancy_walking[t] for t in range(len(occupancy))]
    FCU_power = data_recorder[room_str]["FCU_power"]

    ax.plot(room_temp, marker='o', linestyle='-', color='b', label='Temperature')
    ax.plot(outdoor_temp, marker='o', linestyle='-', color='r', label='Outdoor Temperature')
    # ax.scatter(on_times, [20] * len(on_times), color='black', s=10)

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.set_title(room_str)
    ax.set_ylim(19, 31)  # Set y-axis limits
    ax.set_xlim(-20, 620)  # Set x-axis limits
    # Set y-axis grid every 1 unit
    ax.yaxis.set_ticks(range(20, 30, 1))  # Set y-ticks at intervals of 1
    ax.xaxis.set_ticks(range(0, 600, 60))  # Set x-ticks at intervals of 60




    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Create a second y-axis for occupancy
    ax_twin = ax.twinx()
    ax_twin.plot(occupancy_sitting,  linestyle='-', color='m', label='sitting', alpha=0.7)
    ax_twin.plot(occupancy_standing,  linestyle='-', color='c', label='standing', alpha=0.7)
    ax_twin.plot(occupancy_walking,  linestyle='-', color='y', label='walking', alpha=0.7)
    ax_twin.plot(occupancy_total,  linestyle='-', color='k', label='total', alpha=1.0)

    ax_twin.set_ylabel('Occupancy (People)')
    ax_twin.set_ylim(0, 11)  # Set y-axis limits for occupancy
    ax_twin.yaxis.set_ticks(range(0, 5, 1))  # Set y-ticks for occupancy

    if i == 6:
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        # Add legends and move them outside

        # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Move legend to the right
        # ax_twin.legend(loc='upper left', bbox_to_anchor=(1.05, 0.8))  # Move second legend to the right

# 子图：Reward
ax2 = axes[8]
reward = data_recorder["training"]["reward"]
total_reward = np.sum(reward)
ax2.plot(reward, marker='o', linestyle='-', color='g', label='Reward')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Reward')
ax2.set_title('Mode: ' + reward_mode + " C: " + str(tradeoff_constant) + " Total R: " + str(round(total_reward, 1)))
ax2.legend()
ax2.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 子图： FCU Power
ax3 = axes[9]
FCU_power = data_recorder["training"]["energy_consumption"]
FCU_power_total = np.sum(FCU_power)
ax3.plot(FCU_power, marker='o', linestyle='-', color='g', label='FCU Power')
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('FCU Power')
ax3.set_title('Total FCU Power: ' + str(FCU_power_total))
ax3.legend()
ax3.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 子图：PMV 的ABS均值
ax4 = axes[10]
pmv_mean = data_recorder["training"]["mean_pmv"]
ax4.plot(pmv_mean, marker='o', linestyle='-', color='g', label='PMV Mean')
ax4.set_xlabel('Time Steps')
ax4.set_ylabel('ABS PMV Mean')
ax4.set_title('PMV Mean Over Time')
ax4.legend()
ax4.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 子图：PPD的均值
ax5 = axes[11]
ppd_mean = data_recorder["training"]["mean_ppd"]
ax5.plot(ppd_mean, marker='o', linestyle='-', color='g', label='PPD Mean')
ax5.set_xlabel('Time Steps')
ax5.set_ylabel('PPD Mean')
ax5.set_title('PPD Mean Over Time')
ax5.legend()
ax5.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 调整子图间距
plt.tight_layout()

plt.savefig('figure/test.png')




