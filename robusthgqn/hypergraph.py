import numpy as np
import torch as th


def get_hypergraph_nvec(action_nvec, hypergraph):
    hypergraph_nvec = []
    for hyperedge in hypergraph:
        hyperedge_dim = np.prod(action_nvec[hyperedge])
        hypergraph_nvec.append(hyperedge_dim)
    return hypergraph_nvec


def get_argmax_from_q_values(next_q_values):
    """
    计算每个 tensor 中最大值的索引，对于形状不一致的列表（例如 64x2 和 64x8 的 tensor），
    分别沿列维度计算 argmax。

    参数:
    - next_q_values (list): 一个包含不同形状 tensor 的列表，每个 tensor 的形状可以是 (64, 2) 或 (64, 8)。

    返回:
    - argmax_tensor (Tensor): 包含最大值索引的 2D 张量，形状为 (64, len(next_q_values))。
    """
    argmax_values = []

    # 对每个 tensor 分别使用 argmax
    for tensor in next_q_values:
        # 获取最大值的索引，沿着 dim=1 (即列) 维度获取最大值的索引
        argmax_values.append(th.argmax(tensor, dim=1))

    # 将每个 tensor 的 argmax 结果堆叠成一个 2D 张量，形状为 (64, len(next_q_values))
    argmax_tensor = th.stack(argmax_values, dim=1)

    return argmax_tensor


def get_values_from_idx(next_q_values, argmax):
    """
    根据给定的 argmax 索引，从 next_q_values_target 中取出对应的值。

    参数:
    - next_q_values (list): 一个包含不同形状 tensor 的列表，每个 tensor 的形状可以是 (64, 2) 或 (64, 8)。
    - argmax (Tensor): 包含最大值索引的 2D 张量，形状为 (64, len(next_q_values))，
                       表示每个 tensor 中每一行的最大值的索引。

    返回:
    - selected_values (Tensor): 从 next_q_values 中提取的对应值，形状为 (64, len(next_q_values))。
    """
    selected_values = []

    # 遍历 next_q_values 和 argmax 同时获取每个 tensor 和对应的最大值索引
    for i, tensor in enumerate(next_q_values):
        # 从每个 tensor 中取出相应的最大值
        selected_values.append(tensor[th.arange(tensor.size(0)), argmax[:, i]])

    # 将结果堆叠成一个 2D 张量，形状为 (64, len(next_q_values))
    selected_values_tensor = th.stack(selected_values, dim=1)

    return selected_values_tensor


def get_index(action, action_nvec):
    # 获取 Tensor 的形状
    batch_size, dim = action.shape
    action = action.long()  # 确保是整数类型

    index = th.zeros(batch_size, dtype=th.long, device=action.device)  # 初始化一个 Tensor 来保存每个 [a, b, c, ...] 的序号

    # 对每一维进行循环计算
    for i in range(dim):
        # 对应每个维度的乘数（只乘当前维度之后的最大值）
        multiplier = action_nvec[i + 1:].prod() if i + 1 < dim else 1
        index += action[:, i] * multiplier

    return index


def get_indices(index, action_nvec):
    # 初始化一个空的 tensor 来存储 indices
    dim = len(action_nvec)
    indices = th.zeros(len(index), dim, dtype=th.long, device=index.device)  # 假设 index 是一个 tensor

    for i in range(dim - 1, -1, -1):
        # 对于每个维度，使用除法和取余操作反向计算每个维度的值
        divisor = th.tensor(action_nvec[i], dtype=th.long)
        indices[:, i] = index % divisor
        index //= divisor  # 更新 index 为剩余的部分

    return indices


def convert_action(actions, hypergraph, action_nvec):
    # B * 7 的原始动作，假设 actions 是一个 [B, 7] 的 tensor
    B, _ = actions.shape

    # 创建一个空的结果 tensor 用于存储转换后的索引
    converted_actions = th.zeros(B, len(hypergraph), dtype=th.long, device=actions.device)

    for i, group in enumerate(hypergraph):
        if len(group) == 1:
            # 如果组中只有一个元素，直接赋值
            converted_actions[:, i] = actions[:, group[0]]
        else:
            # 否则，将动作合并,
            action_value = get_index(actions[:, group], action_nvec[group])
            converted_actions[:, i] = action_value

    return converted_actions


def revert_action(converted_actions, hypergraph, action_nvec):
    B, _ = converted_actions.shape
    action_dim = len(action_nvec)

    # 初始化恢复后的动作 tensor
    original_actions = th.zeros(B, action_dim, dtype=th.long, device=converted_actions.device)

    # 对于每一组 hypergraph
    for i, group in enumerate(hypergraph):
        if len(group) == 1:
            # 如果组中只有一个元素，直接恢复
            original_actions[:, group[0]] = converted_actions[:, i]
        else:
            # 否则，恢复合并的动作
            action_value = get_indices(converted_actions[:, i], action_nvec[group])
            original_actions[:, group] = action_value

    return original_actions