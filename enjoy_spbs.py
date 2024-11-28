from stable_baselines3 import DQN
import SemiPhysBuildingSim
import gym
import numpy as np
# 加载模型
model_dict = {"1128_Squared_diff": "logs/dqn_2024-11-28-16-06-02/dqn/SemiPhysBuildingSim-v0_1",
              "1128_Baseline_without_energy": "logs/dqn_2024-11-28-16-06-55/dqn/SemiPhysBuildingSim-v0_1",}

model_key = "1128_Baseline_without_energy"
model = DQN.load( model_dict[model_key]+"/best_model.zip")

# 加载环境
env1 = gym.make("SemiPhysBuildingSim-v0")


for _ in range(1):
    action_list = []
    obs = env1.reset()
    rewards = 0
    done = False
    i = 0
    while not done:
        # print(str(i) + "th obs: " + str(obs))
        i += 1
        # action = env1.action_space.sample()
        # action = np.array(action)
        action , _state = model.predict(obs)
        action_list.append(action)
        # print(action_list)
        obs, r, done, info = env1.step(action)
        # print("action:"+str(action))
        # if i >= 100:
        #     break

        rewards += r
        # print(action_list)
    print("rewards:" + str(rewards))
data_recorder = env1.data_recorder

energy_consumption_list = []
for i in range(7):
    room_str = "room" + str(i+1)
    energy_consumption_room_i = data_recorder[room_str]["FCU_power"]
    energy_consumption_list.append(energy_consumption_room_i)

energy_consumption = np.array(energy_consumption_list)
energy_consumption_every_room = np.sum(energy_consumption,1)
print("Energy Consumption:"+str(energy_consumption_every_room))
energy_consumption_sum = np.sum(energy_consumption)
print("Energy Consumption Sum:"+str(energy_consumption_sum))

import matplotlib.pyplot as plt

# Data to plot
data = data_recorder["room1"]["room_temp"]
data2 = data_recorder["sensor_outdoor"]["outdoor_temp"]
data3 = data_recorder["training"]["reward"]

# 创建包含两个子图的图形，垂直排列
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# 第一个子图：Temperature 和 Outdoor Temperature
ax1.plot(data, marker='o', linestyle='-', color='b', label='Temperature')
ax1.plot(data2, marker='o', linestyle='-', color='r', label='Outdoor Temperature')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Value')
ax1.set_title('Temperature Over Time')
ax1.legend()
ax1.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 第二个子图：Reward
ax2.plot(data3, marker='o', linestyle='-', color='g', label='Reward')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Reward')
ax2.set_title('Reward Over Time')
ax2.legend()
ax2.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()


# # 生成一个示例数组，长度为600，每个元素在0到127之间的整数
# data = np.random.randint(0, 128, 600)

# 将每个数的二进制表示转换为7个开关状态
binary_data = np.array([[int(bit) for bit in f"{value:07b}"] for value in action_list])

# 设置图形
fig, ax = plt.subplots(figsize=(15, 5))

# 遍历每个控制器，绘制开启状态的点
for i in range(7):
    # 第7-i个房间
    # 找到当前控制器的开启状态的时间索引
    on_times = np.where(binary_data[:, i] == 1)[0]
    ax.scatter(on_times, [i] * len(on_times), color='black', s=10)

    Controller_Total_Open_time = len(on_times)
    print("Controller "+str(7-i) +" Open time:" + str(Controller_Total_Open_time))

# 添加轴标签和标题
ax.set_yticks(range(7))
ax.set_yticklabels([f'Controller {7-i}' for i in range(7)])
ax.set_xlabel('Time Step')
ax.set_title('Controller On Status Over Time')
# ax.legend(loc="upper right")

# 显示图形
plt.show()