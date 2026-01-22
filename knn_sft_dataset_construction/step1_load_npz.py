# 文件名: test_npz.py
import numpy as np
import sys
import os

# 1. 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

from interpret_obs import interpret_obs, extract_current_key_features

# 设置 npz 文件路径
NPZ_FILE = "expert_trajectories.npz"


def main():
    if not os.path.exists(NPZ_FILE):
        print(f"错误: 找不到文件 {NPZ_FILE}")
        return

    print(f"正在加载 {NPZ_FILE} ...")
    data = np.load(NPZ_FILE)

    # 查看 npz 中的键
    print(f"文件包含的 Keys: {list(data.keys())}")

    # 读取 obs 和 actions
    # 假设之前的保存代码使用了 'obs' 和 'actions' 作为键名
    # 如果你没指定键名，默认可能是 'arr_0', 'arr_1'
    if 'obs' in data and 'actions' in data:
        observations = data['obs']
        actions = data['actions']
    elif 'arr_0' in data and 'arr_1' in data:
        observations = data['arr_0']
        actions = data['arr_1']
        print("警告: 使用了默认键名 arr_0 (obs) 和 arr_1 (actions)")
    else:
        print("错误: 无法识别 npz 中的数组键名")
        return

    print(f"Observations Shape: {observations.shape}")
    print(f"Actions Shape: {actions.shape}")
    print("-" * 50)

    # === 测试第一条数据 ===
    idx = 1
    obs_sample = observations[idx]
    action_sample = actions[idx]

    print(f"正在解释第 {idx} 条 Obs (Shape: {obs_sample.shape})...\n")

    try:
        # 调用 interpret_obs 函数
        text_description = interpret_obs(obs_sample)

        print("【生成的文本描述】:")
        print(text_description)

        print("\n" + "-" * 20 + " 对应的 Action (T+1) " + "-" * 20)
        print(f"Action (Room 1-7 Next FCU State): {action_sample}")

    except Exception as e:
        print(f"解释 Obs 时发生错误: {e}")


if __name__ == "__main__":
    main()
