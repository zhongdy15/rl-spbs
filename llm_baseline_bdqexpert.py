import numpy as np
import gym
import json_repair
import configparser
import os
import datetime
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple


# ==========================================
# 1. 核心工具函数定义
# ==========================================

def sample_multidiscrete_with_mask(action_space, mask: Optional[Union[np.ndarray, List[np.ndarray], tuple]] = None):
    """
    针对 MultiDiscrete 动作空间的带掩码采样函数。
    """
    if not hasattr(action_space, 'nvec'):
        raise AttributeError("action_space 对象缺少 'nvec' 属性")

    nvec = action_space.nvec
    num_dims = len(nvec)

    if mask is None:
        return action_space.sample()

    if len(mask) != num_dims:
        raise ValueError(f"mask 的长度 ({len(mask)}) 必须等于 action_space 的维度数 ({num_dims})")

    sampled_action = []

    for i in range(num_dims):
        dim_mask = mask[i]
        dim_n = nvec[i]

        if not isinstance(dim_mask, np.ndarray):
            dim_mask = np.array(dim_mask, dtype=np.int8)

        valid_indices = np.where(dim_mask == 1)[0]

        if valid_indices.size > 0:
            chosen = action_space.np_random.choice(valid_indices)
            sampled_action.append(chosen)
        else:
            # 如果某维度没有有效动作，默认返回 0
            sampled_action.append(0)

    return np.array(sampled_action, dtype=action_space.dtype)


def get_action_mask_from_llm(obs: np.ndarray, prompt_template: str, config: configparser.ConfigParser) -> Tuple[
    np.ndarray, str]:
    """
    调用 LLM 并解析出 Action Mask。
    如果 LLM 推荐动作，则 mask 对应位为 1；否则为 0。
    兜底策略：如果解析失败或房间未提及，对应维度全开（全1）。
    """
    # 1. 解释 obs (请确保您的环境中定义了 interpret_obs 和 llm_chat)
    try:
        status_text = interpret_obs(obs)  # 假设这是您外部定义的函数
    except NameError:
        status_text = str(obs)  # 临时 fallback

    # 2. 构造 Prompt
    full_prompt = prompt_template.replace("[Status Need Replacement]", status_text)

    # 3. 准备配置
    llm_api_key = config['llm_api']['api_key']
    llm_base_url = config['llm_api']['url']
    llm_model = config['llm_api']['model']
    messages = [{"role": "user", "content": full_prompt}]

    # 4. 调用 LLM
    print(">>> Sending Prompt to LLM...")
    try:
        # 假设 llm_chat 是您外部定义的函数
        response_str = llm_chat(llm_api_key, llm_model, llm_base_url, messages)
    except NameError:
        print("Error: llm_chat function not defined.")
        return np.ones((7, 4), dtype=np.int8), "LLM Call Failed"

    # print(f"LLM Response: {response_str}")

    # 5. 解析并生成 Mask
    try:
        response_dict = json_repair.loads(response_str)
        recommendations = response_dict.get("recommendations", {})
        reason = response_dict.get("analysis", "No analysis provided")

        if not isinstance(recommendations, dict):
            raise ValueError("Recommendations is not a dict")

        # 初始化 Mask: (7个房间, 4个动作)
        action_mask = np.zeros((7, 4), dtype=np.int8)

        for i in range(1, 8):
            room_key = f"room_{i}"
            allowed_indices = recommendations.get(room_key, [])
            valid_indices = [idx for idx in allowed_indices if isinstance(idx, int) and 0 <= idx < 4]

            if valid_indices:
                action_mask[i - 1, valid_indices] = 1
            else:
                # 单房间兜底：如果 LLM 没提该房间，允许该房间所有动作
                action_mask[i - 1, :] = 1

    except Exception as e:
        print(f"Error parsing LLM response: {e}. Defaulting to ALLOW ALL.")
        reason = f"Parsing Error: {str(e)}"
        # 全局兜底：允许所有动作
        action_mask = np.ones((7, 4), dtype=np.int8)

    return action_mask, reason


# ==========================================
# 2. 主测试循环
# ==========================================

test_episode_num = 1
config = configparser.ConfigParser()
# 请确保路径正确
config.read('llm_baseline_prompt/config/config.ini', encoding='utf-8')

test_model_key = config['llm_api']['model'].split('/')[-1]

# 读取 Prompt 模板
try:
    with open('llm_baseline_prompt/zero_shot_prompt_action_candidates.txt', 'r', encoding='utf-8') as f:
        zero_shot_prompt_template = f.read()
except FileNotFoundError:
    print("错误：找不到 prompt 模板文件，使用默认空模板。")
    zero_shot_prompt_template = "[Status Need Replacement]"

save_folder = 'llm_rl_baseline_260127'
print("results saved in: " + save_folder)
print("test model key: " + test_model_key)

for test_num in range(test_episode_num):

    reward_mode = "Baseline_OCC_PPD_with_energy"
    tradeoff_constant = 10
    frame_skip = 5

    # 注意：请确保您已导入 ObsHistoryWrapper, FrameSkip 以及 gym 环境注册
    env1 = gym.make("SemiPhysBuildingSim-v0",
                    reward_mode=reward_mode,
                    tradeoff_constant=tradeoff_constant,
                    eval_mode=True)

    # 假设 wrappers 已经导入
    try:
        env1 = ObsHistoryWrapper(env1, horizon=frame_skip)
        env1 = FrameSkip(env1, skip=frame_skip)
    except NameError:
        print("Warning: Wrappers not found, running without wrappers.")

    print("Frame skip: " + str(frame_skip))

    for _ in range(1):
        action_list = []
        obs = env1.reset()
        rewards = 0
        done = False
        i = 0

        while not done:
            i += 1

            # ---------------------------------------------------------
            # 修改部分开始：使用 LLM 获取 Mask 并采样动作
            # ---------------------------------------------------------

            # 1. 获取 Mask 和理由
            # action_mask shape: (7, 4), 0=禁止, 1=允许
            action_mask, reason = get_action_mask_from_llm(obs, zero_shot_prompt_template, config)

            # 2. 使用 Mask 进行 MultiDiscrete 采样
            # 这里的 action 将是一个长度为 7 的数组，例如 [0, 1, 3, 0, 2, 1, 0]
            action = sample_multidiscrete_with_mask(env1.action_space, mask=action_mask)

            # ---------------------------------------------------------
            # 修改部分结束
            # ---------------------------------------------------------

            # 记录动作
            action_list.append(action)

            # 执行环境步进
            obs, r, done, info = env1.step(action)
            rewards += r

            if True:  # 这里的 True 可以改为 if i % 10 == 0: 减少打印频率
                print(f"Step {i} | Action: {action} | R: {r:.2f}")
                # print(f"Reason: {reason}")
                # print(f"Mask row 0: {action_mask[0]}") # 调试用：查看第一个房间的 mask

        print("Total rewards:" + str(rewards))

    # ==========================================
    # 3. 绘图逻辑 (保持原有逻辑，稍作数据处理适配)
    # ==========================================

    # 转换为 numpy 数组，shape: (TimeSteps, 7)
    binary_data = np.array(action_list)

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle(f"Test Model: {test_model_key}")
    axes = axes.flatten()

    data_recorder = env1.data_recorder
    outdoor_temp = data_recorder["sensor_outdoor"]["outdoor_temp"]

    for i in range(7):
        ax = axes[i]
        room_str = "room" + str(i + 1)
        room_temp = data_recorder[room_str]["room_temp"]

        # 注意：这里原代码检查 == 1。如果您的动作含义是：
        # 0: Off, 1: Low, 2: Med, 3: High
        # 那么下面的 scatter 只会标记出 Low 的时刻。
        # 如果您想标记所有开启时刻（非0），可以改为 > 0
        on_times = np.where(binary_data[:, i] == 1)[0]  # 注意索引: binary_data[:, i] 对应 room_i+1

        occupancy = data_recorder[room_str]["occupant_num"]
        occupancy_total = [occupancy[t] for t in range(len(occupancy))]

        ax.plot(room_temp, marker='o', linestyle='-', color='b', label='Temperature')
        ax.plot(outdoor_temp, marker='o', linestyle='-', color='r', label='Outdoor Temp')

        # 可以在图上画出动作点，这里保持原样
        # ax.scatter(on_times, [20] * len(on_times), color='black', s=10, label='Action=1')

        ax.set_title(room_str)
        ax.set_ylim(19, 31)
        ax.set_xlim(0, len(room_temp))  # 自动适配长度
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        ax_twin = ax.twinx()
        ax_twin.plot(occupancy_total, linestyle='-', color='k', label='Occupancy', alpha=0.5)
        ax_twin.set_ylim(0, 11)

        if i == 6:
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

    # Reward Plot
    ax2 = axes[8]
    reward_hist = data_recorder["training"]["reward"]
    ax2.plot(reward_hist, color='g')
    ax2.set_title(f'Total Reward: {np.sum(reward_hist):.1f}')

    # Energy Plot
    ax3 = axes[9]
    energy_hist = data_recorder["training"]["energy_consumption"]
    ax3.plot(energy_hist, color='orange')
    ax3.set_title(f'Total Energy: {np.sum(energy_hist):.1f}')

    # PMV Plot
    ax4 = axes[10]
    pmv_hist = data_recorder["training"]["mean_pmv"]
    ax4.plot(pmv_hist, color='purple')
    ax4.set_title(f'Mean PMV: {np.mean(pmv_hist):.2f}')

    # PPD Plot
    ax5 = axes[11]
    ppd_hist = data_recorder["training"]["mean_ppd"]
    ax5.plot(ppd_hist, color='brown')
    ax5.set_title(f'Mean PPD: {np.mean(ppd_hist):.2f}')

    plt.tight_layout()

    # 保存文件
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{save_folder}_{test_model_key}_{timestamp}.png"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.savefig(os.path.join(save_folder, file_name))
    print(f"Figure saved to {file_name}")

    env1.close()
