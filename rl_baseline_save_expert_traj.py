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
# 导入 os 模块来创建文件夹
from interpret_obs import interpret_obs

algo_classes = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "bdq": BDQ}

save_folder = 'D:\\research\\rl-spbs\\rl-spbs\\rl_baseline_0121_bdq'
rl_baseline_251105 = {
    # "ppo": "D:\\research\\remote_logs\\260119_llmrl_a2cppo复现\\ppo_Baseline_OCC_PPD_with_energy_10_2025-12-25-15-05-41\\ppo\\SemiPhysBuildingSim-v0_1",
    # "a2c": "D:\\research\\remote_logs\\260119_llmrl_a2cppo复现\\a2c_Baseline_OCC_PPD_with_energy_10_2025-12-25-15-03-44\\a2c\\SemiPhysBuildingSim-v0_1",
    "bdq": "D:\\research\\remote_logs\\260119_llmrl_a2cppo复现\\bdq_Baseline_OCC_PPD_with_energy_10_2026-01-19-10-51-25\\bdq\\SemiPhysBuildingSim-v0_1",
    # "dqn": "D:\\research\\remote_logs\\260119_llmrl_a2cppo复现\dqn_Baseline_OCC_PPD_with_energy_10_2026-01-19-10-47-55\dqn\SemiPhysBuildingSim-v0_1",
}

seed_num = 5

# ==========================================
# 1. 定义用于收集数据的列表
# ==========================================
collected_observations = []
collected_actions = []

for test_model_key_base in rl_baseline_251105.keys():
    for i in range(seed_num):
        model_dir_base = rl_baseline_251105[test_model_key_base]
        model_dir = model_dir_base[:-2] + f"_{i + 1}"

        test_model_key = f"{test_model_key_base}_{i + 1}"
        # 加载模型
        model = algo_classes[test_model_key_base].load(model_dir + "/best_model.zip")

        print("Loading model Successfully: " + model_dir)
        print("results saved in: " + save_folder)

        reward_mode = "Baseline_OCC_PPD_with_energy"
        tradeoff_constant = 10
        frame_skip = 5

        # 创建环境
        env1 = gym.make("SemiPhysBuildingSim-v0",
                        reward_mode=reward_mode,
                        tradeoff_constant=tradeoff_constant,
                        eval_mode=True)

        # 应用 Wrapper，确保 obs 是 T 到 T-4 的堆叠
        env1 = ObsHistoryWrapper(env1, horizon=frame_skip)
        env1 = FrameSkip(env1, skip=frame_skip)

        print("Frame skip: " + str(frame_skip))

        # 运行 N 个 Episode (这里保持你原代码的 1 次)
        for _ in range(1):
            obs = env1.reset()
            rewards = 0
            done = False
            step_i = 0

            while not done:
                step_i += 1

                # 模型预测动作 (deterministic=True 确保是专家确定的行为)
                action, _state = model.predict(obs, deterministic=True)

                # 转换为 numpy 格式以便保存
                action = np.array(action)

                # ==========================================
                # 2. 核心收集逻辑：保存 当前Obs 和 下一步Action
                # ==========================================
                # obs 已经是通过 Wrapper 处理过的堆叠状态
                # 为了保险起见，使用 flatten() 确保它是 180 维的一维向量
                # 这样堆叠后就是 (N, 180)
                collected_observations.append(obs.flatten())
                collected_actions.append(action)

                # 执行环境步进
                obs, r, done, info = env1.step(action)

                rewards += r

                # 打印第一步的信息用于调试
                if step_i == 1:
                    print(f"Debug - Step 1 Obs Shape: {obs.shape}")  # 应该是 (180,) 或类似
                    print(f"Debug - Step 1 Action: {action}")
                    print("interpret_obs:", interpret_obs(obs))

            print("rewards:" + str(rewards))

        env1.close()

# ==========================================
# 3. 数据转换与保存
# ==========================================
# 将列表转换为 numpy 数组
final_obs = np.array(collected_observations)
final_actions = np.array(collected_actions)

# 打印最终形状以供核对
print("\n" + "=" * 30)
print("Data Collection Finished")
print("=" * 30)
print(f"Final Obs Shape:    {final_obs.shape}")  # 期望: (N, 180)
print(f"Final Actions Shape: {final_actions.shape}")  # 期望: (N, 7)

# 验证维度是否符合你的严格定义
if final_obs.shape[1] != 180:
    print(f"Warning: Obs dimension is {final_obs.shape[1]}, expected 180.")
if final_actions.shape[1] != 7:
    print(f"Warning: Action dimension is {final_actions.shape[1]}, expected 7.")

# 保存为 .npz
output_filename = "expert_trajectories.npz"
np.savez(os.path.join(save_folder, output_filename), obs=final_obs, actions=final_actions)

print(f"\nSaved successfully to: {os.path.abspath(os.path.join(save_folder, output_filename))}")
print("Structure inside npz:")
print(" - 'obs': (N, 180) -> History stacked observations")
print(" - 'actions': (N, 7) -> Discrete actions per room")
