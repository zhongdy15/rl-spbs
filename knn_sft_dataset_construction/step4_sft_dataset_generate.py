import numpy as np
import json
import os
from typing import Dict
from tqdm import tqdm  # 如果没有安装，可以使用 pip install tqdm，或者删除相关代码
import sys

# ==========================================
# 1. 辅助函数: interpret_obs
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

from interpret_obs import interpret_obs, extract_current_key_features


def build_sft_dataset(
        npz_path: str,
        prompt_path: str,
        output_json_path: str
):
    # --- A. 加载数据 ---
    print(f"Loading data from {npz_path}...")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Data file not found: {npz_path}")

    data = np.load(npz_path)
    obs_data = data['obs']  # Shape: (N, 180)
    valid_mask = data['valid_mask']  # Shape: (N, 7, 4)

    print(f"Loaded {len(obs_data)} samples.")

    # --- B. 加载 Prompt 模板 ---
    print(f"Loading prompt template from {prompt_path}...")
    if not os.path.exists(prompt_path):
        # 如果文件不存在，这里创建一个简单的模拟模板用于测试，实际使用请确保文件存在
        print("Warning: Prompt file not found. Using dummy template for testing.")
        template = "Prompt Header...\n[Status Need Replacement]\nPrompt Footer..."
    else:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

    sft_dataset = []

    # --- C. 遍历处理 ---
    print("Processing samples...")
    # 使用 tqdm 显示进度条
    for i in tqdm(range(len(obs_data)), desc="Building Dataset"):

        # 1. 构建 instruction
        # 将 numpy obs 转换为 描述文本
        try:
            obs_str = interpret_obs(obs_data[i])
        except ValueError as e:
            print(f"Skipping sample {i} due to error: {e}")
            continue

        # 替换模板中的占位符
        instruction = template.replace("[Status Need Replacement]", obs_str)

        # 2. 构建 output
        # 解析 mask 得到推荐动作列表
        # valid_mask[i] shape is (7, 4)
        recommendations = {}
        for room_idx in range(7):
            # 获取第 room_idx 个房间的 mask (长度为4的bool数组)
            room_mask = valid_mask[i][room_idx]
            # 找到 True 的索引位置，即为推荐动作 [0, 1, 2, 3] 中的子集
            # np.where 返回 tuple, 取 [0] 得到索引数组
            allowed_actions = np.where(room_mask)[0].tolist()

            # 如果列表为空（虽然 KNN TopN 不太可能出现全空），给一个默认安全值 [0]
            if not allowed_actions:
                allowed_actions = [0]

            recommendations[f"room_{room_idx + 1}"] = allowed_actions

        # 构造符合 Prompt 要求的 JSON 对象
        # 注意：由于我们是从数据中统计得出的结果，没有专家的文本推理过程。
        # 为了满足 JSON 格式要求，我们生成一个通用的 analysis 字段。
        json_response = {
            "analysis": "Based on the historical room temperature trends, occupancy levels, and outdoor conditions provided in the input, a statistical pattern analysis matches the current state to optimal historical control strategies. The recommended fan speeds listed below aim to maintain the target temperature range (25-27°C) while optimizing energy efficiency.",
            "recommendations": recommendations
        }

        # 将 JSON 对象转为字符串，作为模型的 expected output
        output_str = json.dumps(json_response)  # 默认不加缩进，节省 token

        # 3. 添加到数据集
        sft_dataset.append({
            "instruction": instruction,
            "input": "",
            "output": output_str
        })

    # --- D. 保存结果 ---
    print(f"Saving {len(sft_dataset)} samples to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=2)

    print("Done!")


# ==========================================
# 3. 运行配置
# ==========================================
if __name__ == "__main__":
    # 配置输入输出路径
    NPZ_INPUT = "output3_obs_with_valid_mask_N50_ep0.00.npz"
    PROMPT_INPUT = "../llm_baseline_prompt/zero_shot_prompt_action_candidates.txt"
    JSON_OUTPUT = "output4_sft_training_data.json"

    # 运行构建
    try:
        build_sft_dataset(NPZ_INPUT, PROMPT_INPUT, JSON_OUTPUT)

        # 打印一条数据验证格式
        print("\n--- Sample Data Preview ---")
        with open(JSON_OUTPUT, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data:
                print("Instruction snippet:")
                print(data[0]['instruction'][:200] + "...")
                print("\nOutput snippet:")
                print(data[0]['output'])
    except Exception as e:
        print(f"\nExecution failed: {e}")
