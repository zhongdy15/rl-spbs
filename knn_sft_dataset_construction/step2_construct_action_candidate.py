import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import sys

# 1. 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

from interpret_obs import interpret_obs, extract_current_key_features


def generate_candidate_actions(npz_path: str, k_neighbors: int = 5):
    print(f"[{npz_path}] 开始处理...")

    # --- A. 加载数据 ---
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到文件: {npz_path}")

    data = np.load(npz_path)
    # 兼容常见的键名
    if 'obs' in data and 'actions' in data:
        raw_obs = data['obs']
        raw_actions = data['actions']
    elif 'arr_0' in data and 'arr_1' in data:
        raw_obs = data['arr_0']
        raw_actions = data['arr_1']
    else:
        raise ValueError("NPZ文件中未找到预期的 obs/actions 键值。")

    N_samples = raw_obs.shape[0]
    print(f"已加载数据。样本数 (N): {N_samples}")
    print(f"原始 Obs 形状: {raw_obs.shape}, 原始 Actions 形状: {raw_actions.shape}")

    # --- B. 特征提取 ---
    print("\n开始提取关键特征 (Outdoor + Room Temps + Occupants)...")
    # 直接传入二维数组进行批量处理
    feature_obs = extract_current_key_features(raw_obs)
    print(f"特征提取完成。Feature Obs 形状: {feature_obs.shape}")  # 期望 (N, 15)
    assert feature_obs.shape == (N_samples, 15)

    # --- C. 数据标准化 (Standardization) ---
    print("\n开始数据标准化 (StandardScaler)...")
    # KNN 对特征尺度敏感，必须进行标准化，使不同单位的特征具有相同的权重
    scaler = StandardScaler()
    feature_obs_norm = scaler.fit_transform(feature_obs)
    print("数据标准化完成。")

    # --- D. K近邻搜索 (KNN Search) ---
    print(f"\n开始执行 KNN 搜索 (K={k_neighbors}, Metric=Euclidean)...")
    # 使用欧氏距离查找最近邻
    knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', n_jobs=-1)
    knn_model.fit(feature_obs_norm)

    # 返回每个样本的 K 个近邻的索引 (indices)
    # 注意：在 sklearn 中，查询数据集自身时，最近的邻居（距离为0）总是它自己。
    # 因此 indices[:, 0] 将会是样本自身的索引。
    distances, neighbors_indices = knn_model.kneighbors(feature_obs_norm)
    print(f"KNN 搜索完成。索引矩阵形状: {neighbors_indices.shape}")  # 期望 (N, K)

    # --- E. 构建候选动作集合 ---
    print("\n根据近邻索引检索候选动作...")
    # 使用 numpy 的高级索引功能，利用索引矩阵直接从 raw_actions 中提取动作
    # raw_actions 形状 (N, 7), indices 形状 (N, K) -> 结果形状 (N, K, 7)
    candidate_actions_array = raw_actions[neighbors_indices]

    print("=" * 40)
    print("候选动作生成完毕")
    print(f"最终候选动作数组形状 (N, K, ActionDim): {candidate_actions_array.shape}")
    print("=" * 40)

    return raw_obs, raw_actions, candidate_actions_array, neighbors_indices


# ==============================================================================
# 3. 运行脚本
# ==============================================================================
if __name__ == "__main__":
    NPZ_FILE = "expert_trajectories.npz"  # 请确保此文件存在于当前目录
    K = 300  # 定义近邻数量（包含自身）

    try:
        obs, acts, candidates, nn_idx = generate_candidate_actions(NPZ_FILE, k_neighbors=K)

        # --- 验证环节 ---
        # print("\n--- 验证结果 (Sample 0) ---")
        # sample_idx = 0
        #
        # # 1. 获取样本自身的真实动作 (Ground Truth)
        # self_action_gt = acts[sample_idx]
        # print(f"样本 {sample_idx} 的真实动作 (GT): \n{self_action_gt}")
        #
        # # 2. 获取该样本的候选动作集合 (K个)
        # sample_candidates = candidates[sample_idx]
        # print(f"\n样本 {sample_idx} 的候选动作集合 (Shape: {sample_candidates.shape}):")
        # print(sample_candidates)
        #
        # # 3. 验证要求：“包含自己的动作和K近邻状态的动作”
        # # 由于 sklearn KNN 搜索自身时会包含自身在第一个位置 (index 0)，
        # # 我们检查候选集里的第一个动作是否等于真实动作。
        # is_self_included = np.array_equal(sample_candidates[0], self_action_gt)
        # print(f"\n验证: 候选集中是否包含自身动作 (通常在第一个位置)? -> {'✅ 是' if is_self_included else '❌ 否'}")
        #
        # # 显示其最近邻的索引
        # print(f"该样本的 K 个近邻索引为: {nn_idx[sample_idx]}")
        # print("(第一个索引通常是它自己)")

        # --- 保存结果 (可选) ---
        output_path = "output2_expert_data_with_candidates.npz"
        np.savez_compressed(output_path,
                            obs=obs,
                            actions=acts,
                            candidate_actions=candidates,
                            neighbors_indices=nn_idx)
        print(f"\n处理后的数据已保存至: {output_path}")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保 'expert_trajectories.npz' 文件存在，或修改代码中的文件路径。")
    except Exception as e:
        print(f"\n发生其他错误: {e}")