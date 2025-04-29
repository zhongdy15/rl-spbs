from stable_baselines3 import DQN, PPO, A2C
from rl_zoo3.utils import BDQ, HGQN
# from robusthgqn.hgqn import HGQN as RobustHGQN
import SemiPhysBuildingSim
import gym
import numpy as np
import matplotlib.pyplot as plt
from rl_zoo3.wrappers import FrameSkip, DisabledWrapper
import os
import torch as th

# 扩增了状态空间之后的模型
model_dict_2 = {
                "0105_Baseline_OCC_PPD_with_energy_const10_HGQN": "logs/hgqn_Baseline_OCC_PPD_with_energy_10_2025-01-04-21-10-40/hgqn/SemiPhysBuildingSim-v0_1",
                }

reward_mode_list = ["Baseline_without_energy",
                    "Baseline_with_energy",
                    "Baseline_OCC_PPD_without_energy",
                    "Baseline_OCC_PPD_with_energy",]

algo_dict = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "bdq": BDQ, "hgqn": HGQN}

test_model_key_list = [
    # "0105_Baseline_OCC_PPD_with_energy_const10_A2C",
    # "0105_Baseline_OCC_PPD_with_energy_const10_PPO",
    # "0105_Baseline_OCC_PPD_with_energy_const1_BDQ",
    "0105_Baseline_OCC_PPD_with_energy_const10_HGQN",
    # "0105_Baseline_OCC_PPD_with_energy_const10_DQN",
]

# test_model_key_list = [
#                 # "0103_Baseline_with_energy_const10_A2C",
#                 # "0103_Baseline_with_energy_const10_PPO",
#                 # "0103_Baseline_with_energy_const10_BDQ",
#                 # "0103_Baseline_with_energy_const10_HGQN",
#                 "0103_Baseline_with_energy_const10_DQN",
# ]


save_folder = "figure/0107_HGQN_intuitive_plot/"

room_id = 1

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for model_key in test_model_key_list:

    # model_key = "1210_Baseline_with_energy_constant100_skip5"
    print("Loading model: " + model_key)
    model_path = model_dict_2[model_key]

    for algo_key in algo_dict.keys():
        if algo_key in model_path:
            algo = algo_dict[algo_key]

    model = algo.load( model_path+"/best_model.zip")

    # 加载环境
    # reward_mode = "Baseline_with_energy"
    # tradeoff_constant = 100
    # frame_skip = 5
    for mode in reward_mode_list:
        if mode in model_path:
            reward_mode = mode

    # Extract tradeoff_constant and frame_skip from model_key
    # Assuming the format "constantX_skipY"
    tradeoff_constant = float(model_key.split("const")[1].split("_")[0])
    frame_skip = 5

    env1 = gym.make("SemiPhysBuildingSim-v0",
                    reward_mode=reward_mode,
                    tradeoff_constant=tradeoff_constant,
                    eval_mode=True)
    env1 = FrameSkip(env1, skip=frame_skip)
    print("Frame skip: " + str(frame_skip))

    # env1 = DisabledWrapper(env1)

    for _ in range(1):
        action_list = []
        obs = env1.reset()
        rewards = 0
        done = False
        i = 0
        while not done:
            i += 1
            # with th.no_grad():
            #     action = model.policy.q_net._predict_with_disabled_action(model.policy.obs_to_tensor(obs)[0])\
            #         .cpu().numpy().reshape((-1,) + model.action_space.shape).squeeze(axis=0)
            action, _state = model.predict(obs)
            # print("action: " + str(action))
            # action = env1.action_space.sample()
            # action = np.array(action)
            # action = 127
            # action = np.array(action)
            action_list.append(action)
            obs, r, done, info = env1.step(action)
            rewards += r
        print("rewards:" + str(rewards))

    # binary_data = np.array([[int(bit) for bit in f"{value:07b}"] for value in action_list])




    fig, axes = plt.subplots(3, 1, figsize=(18, 9),sharex=True)  # 3 rows, 4 columns
    # Set the title for the entire figure
    # fig.suptitle("Model: " + model_key, fontsize=16)

    axes = axes.flatten()  # Flatten the 2D array of axes to easily index each subplot

    data_recorder = env1.data_recorder
    outdoor_temp = data_recorder["sensor_outdoor"]["outdoor_temp"]

    i = room_id
    ax = axes[0]
    room_str = "room" + str(i+1)

    room_temp = data_recorder[room_str]["room_temp"]

    # on_times = np.where(binary_data[:, 7 - i - 1] == 1)[0]

    occupancy = data_recorder[room_str]["occupant_num"]
    occupancy_sitting = [ occupancy[t]["sitting"] for t in range(len(occupancy))]
    occupancy_standing = [ occupancy[t]["standing"] for t in range(len(occupancy))]
    occupancy_walking = [ occupancy[t]["walking"] for t in range(len(occupancy))]
    occupancy_total = [ occupancy_sitting[t] + occupancy_standing[t] + occupancy_walking[t] for t in range(len(occupancy))]
    FCU_power = data_recorder[room_str]["FCU_power"]

    occupancy_sitting = np.array(occupancy_sitting)
    occupancy_standing = np.array(occupancy_standing)
    occupancy_walking = np.array(occupancy_walking)
    occupancy_total = np.array(occupancy_total)
    FCU_power = np.array(FCU_power)


    time = np.arange(0, 481, 15)  # 每15分钟一个点
    index_time = time.copy()
    index_time[18] = 269
    index_time[19] = 294
    occupancy_sitting = occupancy_sitting[index_time]  # 每15分钟选择一个坐着的人数
    occupancy_standing = occupancy_standing[index_time]  # 每15分钟选择一个站着的人数
    occupancy_walking = occupancy_walking[index_time]  # 每15分钟选择一个走动的人数
    occupancy_total = occupancy_total[index_time]  # 每15分钟选择一个总人数
    # 绘制第一个子图：堆叠条形图
    bar_width = 10  # 柱子宽度
    ax.bar(time, occupancy_sitting, label='Sitting', color='skyblue', width=bar_width)
    ax.bar(time, occupancy_standing, bottom=occupancy_sitting, label='Standing', color='orange', width=bar_width)
    ax.bar(time, occupancy_walking, bottom=occupancy_sitting + occupancy_standing, label='Walking', color='green', width=bar_width)

    ax.plot(time, occupancy_total, label='Total', color='black')

    # 设置第一个子图的标签和标题
    # ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Occupancy')
    # ax.set_title('Occupancy Over Time')
    ax.legend(loc='upper left',ncol=4)

    ax.set_ylim(0, 5)  # Set y-axis limits for occupancy
    ax.yaxis.set_ticks(range(0, 5, 1))  # Set y-ticks for occupancy

    ax.set_xlim(-20, 500)  # Set x-axis limits
    ax.xaxis.set_ticks(range(0, 481, 60))  # Set x-ticks at intervals of 60
    time_labels = ['9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00']
    ax.set_xticklabels(time_labels)  # Set x-tick labels
    ax.grid(True)


    ax1 = axes[2]
    temp_time = np.arange(0, 481, 60)
    room_temp = np.array(room_temp)
    outdoor_temp = np.array(outdoor_temp)
    ax1.plot(time,room_temp[time], marker='o', linestyle='-', color='green', label='Indoor Temperature')
    ax1.plot(temp_time,outdoor_temp[temp_time], marker='o', linestyle='-', color='blue', label='Outdoor Temperature')
    ax1.legend(loc='upper left',ncol=2)
    ax1.grid(True)
    ax1.set_ylim(21, 31)  # Set y-axis limits for occupancy
    ax1.yaxis.set_ticks(range(21, 31, 1))  # Set y-ticks for occupancy
    ax1.set_ylabel('Temperature (°C)')



    ax2 = axes[1]
    temp_time = np.arange(0, 480, frame_skip)
    action = [action_list[t][room_id] for t in range(len(action_list))]
    action = np.array(action)
    ax2.plot(temp_time, action[:480//frame_skip], marker='o', linestyle='-', color='red', label='FCU Speed Mode')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_ylim(-0.2, 4)  # Set y-axis limits for occupancy
    ax2.yaxis.set_ticks(range(0, 4, 1))  # Set y-ticks for occupancy
    ax2.set_ylabel('FCU Speed Mode')


    plt.tight_layout()

    # plt.show()
    plt.savefig(save_folder + '/' +  model_key + '.png')

    env1.close()





