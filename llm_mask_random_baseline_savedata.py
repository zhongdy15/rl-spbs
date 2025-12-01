import gym
import numpy as np
import os
import datetime
import configparser
import pickle  # 用于保存复杂结构的数据
from stable_baselines3 import DQN, PPO, A2C
from rl_zoo3.wrappers import FrameSkip, ObsHistoryWrapper
import SemiPhysBuildingSim
from llm_baseline_prompt.llm_chat import llm_chat
from rl_zoo3.action_mask_info_wrapper import get_action_mask_from_llm_2nd

# === 配置部分 ===
test_episode_num = 5

config = configparser.ConfigParser()
config.read('llm_baseline_prompt/config/config.ini', encoding='utf-8')
test_model_key = config['llm_api']['model'].split('/')[-1]

exp_name = "llm_am_random_baseline_251126"

save_folder = os.path.join('enjoy_data', exp_name, test_model_key)

print("Results will be saved in: " + save_folder)
print("Test model key: " + test_model_key)




# 生成本次实验的唯一标识时间戳
experiment_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_filename = f"{experiment_timestamp}.pkl"
save_path = os.path.join(save_folder, save_filename)

# 确保保存目录存在
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 用于存储所有回合的数据
all_episodes_data = []

reward_mode_list = ["Baseline_without_energy",
                    "Baseline_with_energy",
                    "Baseline_OCC_PPD_without_energy",
                    "Baseline_OCC_PPD_with_energy"]
reward_mode = "Baseline_OCC_PPD_with_energy"
tradeoff_constant = 10
frame_skip = 5

print("Frame skip: " + str(frame_skip))

# 创建环境 (在循环外创建一次即可，除非每回合需要完全重构)
env1 = gym.make("SemiPhysBuildingSim-v0",
                reward_mode=reward_mode,
                tradeoff_constant=tradeoff_constant,
                eval_mode=True)
env1 = ObsHistoryWrapper(env1, horizon=frame_skip)
env1 = FrameSkip(env1, skip=frame_skip)

for test_num in range(test_episode_num):
    print(f"--- Starting Episode {test_num + 1}/{test_episode_num} ---")

    obs = env1.reset()
    rewards = 0
    done = False
    i = 0

    # 用于存储单回合的动作和Step信息
    action_list = []
    step_records = []

    while not done:
        i += 1

        # 获取 Action Mask
        action_mask = get_action_mask_from_llm_2nd(obs, -1, env1.action_space)

        # 【关键修改】只保存 mask 为 1 的 index
        mask_indices = np.where(action_mask == 1)[0]

        # 选择动作
        action_index = np.random.choice(mask_indices)

        # ###for test###
        # action_mask = np.ones(env1.action_space.n, dtype=int)
        # mask_indices = np.where(action_mask == 1)[0]
        # action_index = np.random.choice(mask_indices)
        # ###for test###



        action = np.array(action_index)

        # 记录动作
        action_list.append(action)

        # 执行环境
        next_obs, r, done, info = env1.step(action)

        # 记录这一步的详细数据
        step_data = {
            "step_index": i,
            "mask_indices": mask_indices,  # 压缩后的 mask
            "action": action,
            "reward": r,
            "done": done,
            # 如果 obs 很大且不需要每步分析，可以注释掉下面这行以节省空间
            "obs": obs
        }
        step_records.append(step_data)

        rewards += r
        obs = next_obs  # 更新 obs

        if True:  # 可以根据需要调整打印频率
            print(f"Step {i} action: {action}, r: {r}, done: {done}")

    print("Episode rewards:" + str(rewards))

    # 提取环境内部记录器的数据
    # 注意：我们需要深拷贝或者确保这里的数据是最终状态
    env_recorder_data = env1.data_recorder.copy()

    # 整合本回合数据
    episode_data = {
        "episode_idx": test_num,
        "reward_mode": reward_mode,
        "tradeoff_constant": tradeoff_constant,
        "test_model_key": test_model_key,
        "data_recorder": env_recorder_data,  # 包含温度、能耗等详细历史
        "action_list": np.array(action_list),
        "step_records": step_records,  # 包含压缩的 mask
        "total_reward": rewards
    }

    all_episodes_data.append(episode_data)

env1.close()

# === 保存所有数据到文件 ===
print(f"Saving data to {save_path} ...")
with open(save_path, 'wb') as f:
    pickle.dump(all_episodes_data, f)
print("Save complete.")
