import numpy as np
import os
import sys

# 1. 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

from interpret_obs import interpret_obs, extract_current_key_features

def process_candidate_frequencies(npz_path, top_n=50, epsilon=0.0):
    """
    读取 KNN 结果，计算动作频率，并根据 epsilon 筛选可能的动作集合。

    Args:
        npz_path (str): .npz 文件路径
        top_n (int): 只取前 N 个最近邻 (包含自身)
        epsilon (float): 概率阈值，低于此频率的动作被视为不可能

    Returns:
        action_freq (np.ndarray): 形状 (N, 7, 4)，每个样本每个房间的动作频率分布
        valid_action_mask (np.ndarray): 形状 (N, 7, 4)，布尔值，表示动作是否有效
    """
    # 1. 加载数据
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件未找到: {npz_path}")

    print(f"正在加载 {npz_path} ...")
    data = np.load(npz_path)

    # 获取 candidate_actions (N, K, 7)
    # 注意：确保加载的是 candidate_actions
    if 'candidate_actions' in data:
        candidates = data['candidate_actions']
    else:
        raise ValueError("npz 文件中未找到 'candidate_actions'")

    N_samples, K_neighbors, n_rooms = candidates.shape
    print(f"原始候选动作形状: {candidates.shape}")

    # 2. 截取前 top_n 个近邻
    # 如果 top_n 大于实际拥有的 K，则取全部
    real_top_n = min(top_n, K_neighbors)
    candidates_top = candidates[:, :real_top_n, :]  # Shape: (N, top_n, 7)
    print(f"截取前 {real_top_n} 个近邻。形状变为: {candidates_top.shape}")

    # 3. 计算频率 (核心步骤)
    # 动作空间是离散的: 0, 1, 2, 3。总共 4 个动作。
    n_actions = 4

    # 3.1 确保是整数类型 (作为索引使用)
    candidates_top = candidates_top.astype(int)

    # 3.2 使用 One-Hot 技巧进行向量化统计
    # np.eye(4) 生成单位矩阵
    # np.eye(4)[candidates_top] 会利用索引广播机制生成 One-Hot 编码
    # 结果形状变为: (N, top_n, 7, 4)
    candidates_one_hot = np.eye(n_actions)[candidates_top]

    # 3.3 沿 axis=1 (近邻维度) 求平均值，即得到频率
    # 结果形状: (N, 7, 4)
    action_freq = np.mean(candidates_one_hot, axis=1)

    print(f"频率计算完成。结果形状: {action_freq.shape}")

    # 4. 根据 epsilon 获取动作集合 (Mask)
    # 结果形状: (N, 7, 4)，类型为 bool
    valid_action_mask = action_freq > epsilon

    return action_freq, valid_action_mask, candidates_top, data["obs"], data["actions"]


# ==========================================
# 测试与验证
# ==========================================
if __name__ == "__main__":
    NPZ_FILE = "output2_expert_data_with_candidates.npz"

    # 超参数设置
    TOP_N = 50  # 只看前50个最相似的状态
    EPSILON = 0.0  # 只有出现过 的动作才被认为是“候选动作”

    try:
        freq_array, mask_array, raw_top_candidates, obs, actions = process_candidate_frequencies(
            NPZ_FILE, top_n=TOP_N, epsilon=EPSILON
        )

        # --- 结果展示 ---
        print("\n" + "=" * 40)
        print(f"结果分析 (Epsilon = {EPSILON})")
        print("=" * 40)

        # 随机抽取一个样本查看详情
        sample_idx = 0
        print(f"样本 {sample_idx} 分析:")

        # 打印该样本前5个邻居的动作，有个直观感受
        print(f"该样本前 5 个近邻的动作 (用于直观对比):")
        print(raw_top_candidates[sample_idx, :5, :])

        print("\n各房间动作频率分布 (7个房间 x 4个动作):")
        # 格式化打印，保留2位小数
        np.set_printoptions(precision=2, suppress=True)
        print(freq_array[sample_idx])

        print("\n筛选后的候选动作集合 (概率 > Epsilon):")
        # 将 Mask 转换为人类可读的列表格式
        for room_i in range(7):
            # np.where 返回满足条件的索引
            valid_acts = np.where(mask_array[sample_idx, room_i, :])[0]
            print(
                f"Room {room_i + 1}: 可选动作 {valid_acts.tolist()} \t(对应概率: {freq_array[sample_idx, room_i, valid_acts]})")

        # --- 保存最终结果 ---
        # 我们可以保存频率分布，训练时可以作为 Soft Label 或者 Mask 使用
        save_path = f"output3_obs_with_valid_mask_N{TOP_N}_ep{EPSILON:.2f}.npz"
        np.savez_compressed(save_path,
                            obs=obs,
                            actions=actions,
                            action_freq=freq_array,
                            valid_mask=mask_array)
        print(f"\n处理结果已保存至: {save_path}")

    except Exception as e:
        print(f"发生错误: {e}")
