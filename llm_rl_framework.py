from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import SemiPhysBuildingSim
import gym
import numpy as np
import matplotlib.pyplot as plt
from rl_zoo3.wrappers import FrameSkip, ObsHistoryWrapper
import os
import datetime
import os # 导入 os 模块来创建文件夹
import configparser
from llm_baseline_prompt.llm_chat import llm_chat
import json
import json_repair
from interpret_obs import interpret_obs, get_current_fan_speed_from_obs
from SemiPhysBuildingSim.common.action_transformation import action_index_to_array, array_to_action_index, get_action_mask_fast
from typing import Optional


def sample_with_mask(action_space, mask: Optional[np.ndarray] = None):
    """
    在不修改库函数的前提下，模拟新版 gym.spaces.Discrete.sample(mask) 的功能。

    Args:
        action_space: 环境的动作空间对象 (例如 env.action_space)。
                      这个对象应该有 'n' (动作数量) 和 'np_random' (随机数生成器) 属性。
        mask: 一个可选的掩码，用于指定可以选择的动作。
              期望是 shape 为 (n,) 的 np.ndarray，dtype 为 np.int8，
              其中 1 代表有效动作，0 代表无效/不可行动作。
              如果没有有效动作 (即 mask 全为 0)，将返回 action_space.start (通常是 0)。

    Returns:
        一个从有效动作中采样得到的整数。
    """
    # 检查 action_space 是否有必要的属性
    if not hasattr(action_space, 'n'):
        raise AttributeError("action_space 对象缺少 'n' 属性")
    if not hasattr(action_space, 'np_random'):
        raise AttributeError("action_space 对象缺少 'np_random' 属性")

    # 模拟新版库函数的 start 属性，对于旧版库，它通常是 0
    start = getattr(action_space, 'start', 0)

    # 如果没有提供 mask，就调用原始的 sample 方法
    if mask is None:
        # 旧版的 sample() 等价于 np_random.randint(n)
        return start + action_space.np_random.randint(action_space.n)

    # --- 以下是处理 mask 的逻辑 ---

    # 校验 mask 的类型和格式 (与新版库函数保持一致)
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask 的期望类型是 np.ndarray, 实际类型是: {type(mask)}")
    if mask.dtype != np.int8:
        # 注意：为了更强的兼容性，这里可以放宽对 dtype 的严格要求，
        # 或者进行类型转换，但为了严格模拟，我们保留检查。
        print(f"警告: mask 的期望 dtype 是 np.int8, 实际是: {mask.dtype}。尝试继续...")
    if mask.shape != (action_space.n,):
        raise ValueError(f"mask 的期望 shape 是 {(action_space.n,)}, 实际是: {mask.shape}")

    # 找出所有有效的动作
    valid_actions_indices = np.where(mask == 1)[0]

    # 如果有有效的动作
    if valid_actions_indices.size > 0:
        # 从有效动作索引中随机选择一个
        # 使用 action_space.np_random 以确保结果可复现 (如果设置了种子)
        return int(start + action_space.np_random.choice(valid_actions_indices))
    else:
        # 如果没有有效的动作，按规定返回 start
        return int(start)


def get_action_from_llm(obs: np.ndarray, prompt_template: str, config: configparser.ConfigParser) -> np.ndarray:
    """
    集成所有步骤：解释 obs -> 构造 prompt -> 调用 LLM -> 解析 action。
    """
    # 1. 解释 obs
    status_text = interpret_obs(obs)

    # 2. 构造完整 Prompt
    full_prompt = prompt_template.replace("[Status Need Replacement]", status_text)

    # 3. 准备调用 LLM
    llm_api_key = config['llm_api']['api_key']
    llm_base_url = config['llm_api']['url']
    llm_model = config['llm_api']['model']

    messages = [
        # prompt.txt 的内容更适合作为 user prompt，因为它直接给出了任务指令
        {"role": "user", "content": full_prompt}
    ]

    # 4. 调用 LLM 并获取响应
    print("===================== Sending Prompt to LLM =====================")
    # print(full_prompt)
    print("=================================================================")

    response_str = llm_chat(llm_api_key, llm_model, llm_base_url, messages)

    # print(f"LLM Response (raw): {response_str}")

    # 5. 解析 LLM 响应
    try:
        action_dict = json_repair.loads(response_str)
        confidence_scores = action_dict["confidence_scores"]
        # 按照 room_1 到 room_7 的顺序提取动作值
        action_list = [confidence_scores[i] for i in range(1, 8)]
        confidence_scores = np.array(action_list)
    except Exception as e:
        print(f"Error parsing LLM response: {e}. Using default safe action [0,0,0,0,0,0,0].")
        confidence_scores = np.zeros(7)

    try:
        action_dict = json_repair.loads(response_str)
        reason = action_dict["explanation_and_thought_process"]
    except Exception as e:
        print(f"Error parsing LLM response: {e}. Using default safe reason 'No reason provided'.")
        reason = "No reason provided"

    return confidence_scores,reason


test_episode_num = 1
config = configparser.ConfigParser()
config.read('llm_baseline_prompt/config/config.ini', encoding='utf-8')

test_model_key = config['llm_api']['model'].split('/')[-1]

# 读取 Prompt 模板
try:
    with open('llm_baseline_prompt/zero_shot_prompt_action_reduce.txt', 'r', encoding='utf-8') as f:
        zero_shot_prompt_template = f.read()
except FileNotFoundError:
    print("错误：找不到 'llm_baseline_prompt/zero_shot_prompt_action_reduce.txt' 文件。")
    exit()


save_folder = 'llm_rl_baseline_251113'

print("results saved in: " + save_folder)
print("test model key: " + test_model_key)

for test_num in range(test_episode_num):



    reward_mode_list = ["Baseline_without_energy",
                        "Baseline_with_energy",
                        "Baseline_OCC_PPD_without_energy",
                        "Baseline_OCC_PPD_with_energy",]

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

    for _ in range(1):
        action_list = []
        obs = env1.reset()
        rewards = 0
        done = False
        i = 0
        while not done:
            i += 1

            old_action = get_current_fan_speed_from_obs(obs)
            old_action_index = array_to_action_index(old_action, [4, 4, 4, 4, 4, 4, 4])

            # 从 LLM 获取confidence_scores
            confidence_scores, reason = get_action_from_llm(obs, zero_shot_prompt_template, config)
            # 从confidence_scores挑出最大的3个confidence score，该处是controllable，其他位置不controllable
            max_indices = np.argsort(confidence_scores)[-3:]

            # controllable_rooms = np.array([1, 0, 1, 0, 1, 0, 0])
            controllable_rooms = np.zeros_like(old_action, dtype=int)
            controllable_rooms[max_indices] = 1

            # 得到 action mask
            action_mask = get_action_mask_fast(controllable_rooms, old_action_index, env1.action_space)

            random_action = env1.action_space.sample(mask=action_mask)


            # print(interpret_obs(obs))
            # print(get_current_fan_speed_from_obs(obs))
            # action[max_indices] = confidence_scores[max_indices]

            # action = np.array(action)
            # action_list.append(action)
            # print("Step " + str(i) + " action:" + str(action))
            obs, r, done, info = env1.step(action)



            rewards += r
            if True:
                print("Step " + str(i) + " action:" + str(action))
                print("confidence_scores:", confidence_scores)
                print("reason:", reason)
                print("obs shape:", obs.shape)
                print("r:", r)
                print("done:", done)
                print("info:", info)
        print("rewards:" + str(rewards))

    binary_data = np.array(action_list)

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))  # 3 rows, 4 columns
    # Set the title for the entire figure
    fig.suptitle("test")

    axes = axes.flatten()  # Flatten the 2D array of axes to easily index each subplot


    data_recorder = env1.data_recorder
    outdoor_temp = data_recorder["sensor_outdoor"]["outdoor_temp"]


    for i in range(7):
        ax = axes[i]
        room_str = "room" + str(i+1)

        room_temp = data_recorder[room_str]["room_temp"]

        # on_times = np.where(binary_data[:, 7 - i - 1] == 1)[0]

        occupancy = data_recorder[room_str]["occupant_num"]
        # occupancy_sitting = [ occupancy[t]["sitting"] for t in range(len(occupancy))]
        # occupancy_standing = [ occupancy[t]["standing"] for t in range(len(occupancy))]
        # occupancy_walking = [ occupancy[t]["walking"] for t in range(len(occupancy))]
        # occupancy_total = [ occupancy_sitting[t] + occupancy_standing[t] + occupancy_walking[t] for t in range(len(occupancy))]

        occupancy_total = [occupancy[t] for t in range(len(occupancy))]
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
        # ax_twin.plot(occupancy_sitting,  linestyle='-', color='m', label='sitting', alpha=0.7)
        # ax_twin.plot(occupancy_standing,  linestyle='-', color='c', label='standing', alpha=0.7)
        # ax_twin.plot(occupancy_walking,  linestyle='-', color='y', label='walking', alpha=0.7)
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
    pmv_mean_avarege = np.mean(pmv_mean)
    ax4.plot(pmv_mean, marker='o', linestyle='-', color='g', label='PMV Mean')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('ABS PMV Mean')
    ax4.set_title('PMV Mean: '+ str(round(pmv_mean_avarege, 2)))
    ax4.legend()
    ax4.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # 子图：PPD的均值
    ax5 = axes[11]
    ppd_mean = data_recorder["training"]["mean_ppd"]
    ppd_mean_avarege = np.mean(ppd_mean)
    ax5.plot(ppd_mean, marker='o', linestyle='-', color='g', label='PPD Mean')
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('PPD Mean')
    ax5.set_title('PPD Mean: '+str(round(ppd_mean_avarege, 2)))
    ax5.legend()
    ax5.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # 调整子图间距
    plt.tight_layout()


    # 获取当前时间并格式化为字符串，例如 "2025-09-11_08-31-00"
    model_key = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 定义保存文件夹

    model_key = save_folder + "_" + test_model_key + "_" + model_key

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 保存图片
    plt.savefig(os.path.join(save_folder, model_key + '.png'))



    env1.close()
