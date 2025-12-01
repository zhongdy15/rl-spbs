from gym.spaces import Dict, Discrete, Box
import gym
import numpy as np
# Action is Discrete Now

def available_action_set():
    action = {}

    # 初始化每个房间的控制参数
    for r in range(1, 8):
        room_control_key = f"room{r}_control"
        action[room_control_key] = {
            "roomtemp_setpoint": {24}, #
            "roomRH_setpoint": {70},# 固定70
            "FCU_onoff_setpoint": {1}, # 选{0，1}最简单控制
            "FCU_fan_setpoint": {0,1,2,3}, #{0,3}, #{0,1,2,3}, # 选{1，2，3}低中高档 {0,1,3},#
            "FCU_workingmode_setpoint": {1},
            "valve_setpoint": {100},# 固定100
        }

    print("FCU_fan_setpoint: " + str(action["room1_control"]["FCU_fan_setpoint"]))
    # 初始化泵的控制参数[暂时固定]
    action["pump1_control"] = {
        "pump_onoff_setpoint": {1},
        "pump_frequency_setpoint": {50},
        "pump_valve_setpoint": {1},
    }

    # 初始化热泵的控制参数【暂时固定】
    action["heatpump_control"] = {
        "heatpump_onoff_setpoint": {1},
        "heatpump_supplytemp_setpoint": {7},
        "heatpump_workingmode_setpoint": {2},
    }
    return action

def create_action_space(action_set):
    action_space = []
    for room in action_set.keys():
        params = action_set[room]
        # 获取每个控制参数的取值个数
        sizes = [len(v) for v in params.values()]
        action_space.append(sizes)

    # 计算总的离散动作组合数量
    total_actions = np.prod([np.prod(size) for size in action_space])
    return gym.spaces.Discrete(total_actions)


def map_action_to_controls(action_set, action_index):
    controls = {}
    # 从action_set中获取控制参数
    for room, params in action_set.items():
        controls[room] = {}
        param_indices = {key: 0 for key in params.keys()}

        # 根据索引计算出每个控制参数的取值
        for key, value_set in params.items():
            value_list = list(value_set)
            size = len(value_list)
            param_index = (action_index % size)
            param_indices[key] = value_list[param_index]
            action_index //= size

        controls[room] = param_indices

    return controls


def map_controls_to_action(action_set, controls):
    action_index = 0
    multiplier = 1
    for room, params in action_set.items():
        room_control = controls[room]

        for key, value_set in params.items():
            value_list = list(value_set)
            value_index = value_list.index(room_control[key])  # 找到控制参数的索引
            action_index += value_index * multiplier
            multiplier *= len(value_list)  # 更新乘数以支持多维组合

    return action_index

def action_index_to_array(action_index, dims):
    """
    参数:
        action_index : int
        dims : list或tuple，长度为7，每一维的取值数量，如 [3, 2, 4, 5, 3, 2, 2]

    返回:
        np.array(size=7)
    """
    assert isinstance(action_index, int), "action_index must be an integer"
    result = []
    for dim_size in dims:
        result.append(action_index % dim_size)
        action_index //= dim_size
    return np.array(result, dtype=int)


def array_to_action_index(arr, dims):
    """
    参数:
        arr : np.array(size=7)
        dims: 每一维的取值数量

    返回:
        对应的 action_index (int)
    """
    action_index = 0
    multiplier = 1

    for value, dim_size in zip(arr, dims):
        action_index += value * multiplier
        multiplier *= dim_size

    return action_index


def get_action_mask_fast(controllable_rooms, old_action_index, action_space):
    """
    高效地生成动作掩码(action mask)，避免遍历整个动作空间。

    Args:
        controllable_rooms (np.array): 长度为7的数组，1表示可控，0表示不可控。
        old_action_index (int): 上一个时间步的动作索引。
        action_space (gym.spaces.Discrete): 智能体的动作空间。

    Returns:
        np.array: 一个长度与动作空间大小相同的0/1数组，
                  值为1表示该动作可选，值为0表示该动作被屏蔽。
    """
    n_actions = action_space.n # 16384
    n_rooms = 7
    # 动作空间的基数，即每个房间风机档位的选项数 (4)
    base = 4 #int(round(n_actions ** (1 / n_rooms)))  # 4 = 16384^(1/7)

    # 1. 初始化一个全为1的掩码数组，表示所有动作初始都可选
    action_mask = np.ones(n_actions, dtype=np.int8)

    # 2. 找到不可控房间的索引
    uncontrollable_room_indices = np.where(controllable_rooms == 0)[0]

    # 3. 遍历每一个不可控的房间，并排除无效动作
    temp_index = old_action_index
    for room_idx in range(n_rooms):
        # 计算当前房间对应的乘数（权重），即 4^0, 4^1, 4^2, ...
        multiplier = base ** room_idx

        # a. 提取旧动作中，当前房间的动作值 (0, 1, 2, or 3)
        old_room_action = (temp_index // multiplier) % base

        # b. 如果当前房间是不可控的
        if room_idx in uncontrollable_room_indices:
            # 我们需要将所有动作中，该房间动作值不等于 old_room_action 的动作屏蔽掉

            # 使用向量化操作，高效地找到所有需要被屏蔽的动作索引
            # 核心思想：(indices // multiplier) % base != old_room_action
            # 这会找所有在 room_idx "位" 上值不等于 old_room_action 的索引
            all_indices = np.arange(n_actions)
            # 计算所有索引在当前房间“位”上的值
            room_actions_for_all_indices = (all_indices // multiplier) % base
            # 找到那些值不等于旧动作值的索引
            indices_to_mask = np.where(room_actions_for_all_indices != old_room_action)[0]

            # 将这些索引在mask中置为0
            action_mask[indices_to_mask] = 0

    return action_mask


def build_mask_from_recommendations(
        recommendations,
        action_space_n: int = 16384,
        n_rooms: int = 7,
        base: int = 4
) -> np.ndarray:
    """
    根据每个房间的推荐动作列表构建全局动作掩码。

    逻辑：
    全局动作 Index 被视为一个 7 位的 4 进制数。
    只有当第 i 位的数值存在于 recommendations['room_{i+1}'] 中时，该全局动作才保留。

    Args:
        recommendations (Dict): 格式如 {'room_1': [0], 'room_2': [0, 1], ...}
        action_space_n (int): 动作空间总大小 (4^7 = 16384)。
        n_rooms (int): 房间数量。
        base (int): 每个房间的档位数量。

    Returns:
        np.array: 0/1 掩码数组，1 表示该全局动作符合所有房间的推荐。
    """
    # 1. 初始化全 1 掩码
    action_mask = np.ones(action_space_n, dtype=np.int8)

    # 创建所有动作的索引数组 [0, 1, ..., 16383]
    all_indices = np.arange(action_space_n)

    # 2. 逐个房间应用约束
    # 假设 recommendations 的键是 "room_1", "room_2" ... "room_7"
    for i in range(n_rooms):
        room_key = f"room_{i + 1}"

        # 获取该房间允许的动作列表，例如 [1, 2]
        # 如果 LLM 漏掉了某个房间，默认允许所有动作（或根据需求报错/全禁）
        allowed_actions = recommendations.get(room_key, list(range(base)))

        if len(allowed_actions) == base:
            # 如果允许该房间的所有动作，则不需要过滤
            continue

        # 计算当前房间对应的权重 (4^0, 4^1, ...)
        # 注意：这里假设 room_1 是最低位 (Little-Endian)。
        # 如果你的编码方式是 room_1 是最高位，请修改此处逻辑。
        # 参照你之前的代码: multiplier = base ** room_idx，这通常意味着 room_0(即room_1) 是最低位。
        multiplier = base ** i

        # 3. 向量化计算：提取所有全局动作在该房间这一"位"上的值
        # 公式: action_val = (global_index // 4^i) % 4
        current_room_values = (all_indices // multiplier) % base

        # 4. 判断这些值是否在允许列表中
        # np.isin 返回一个布尔数组
        is_valid = np.isin(current_room_values, allowed_actions)

        # 5. 更新全局掩码 (逻辑与)
        # 只有之前合法 且 当前房间也合法的动作才保留
        action_mask &= is_valid.astype(np.int8)

        # 提早退出优化：如果掩码全为0，说明没有满足条件的动作，无需继续计算
        if not np.any(action_mask):
            break

    return action_mask

if __name__ == "__main__":
    import json
    # 使用示例
    action_set = available_action_set()
    action_space = create_action_space(action_set)

    # 从动作空间中采样
    sampled_action_index = action_space.sample()
    print(f"Sampled Action Index: {sampled_action_index}")

    controls = map_action_to_controls(action_set, sampled_action_index)
    # 将控制指令映射到动作索引
    mapped_action_index = map_controls_to_action(action_set, controls)
    print(f"Mapped Action Index: {mapped_action_index}")
    print("Mapped Controls:", controls)

    # --- 场景设定 ---
    controllable_rooms = np.array([1, 0, 1, 0, 1, 0, 0])
    old_action_index = 500 # [0,1,3,3,1,0,0]d
    controls = map_action_to_controls(action_set, old_action_index)
    # print(f"Old Action: {controls}")
    print("Old Action (JSON format):")
    print(json.dumps(controls, indent=4, ensure_ascii=False))

    action_mask_fast = get_action_mask_fast(controllable_rooms, old_action_index, action_space)
    print("\n--- 结果验证 ---")
    action_mask = action_mask_fast  # 使用快速方法的结果进行验证

    valid_action_indices = np.where(action_mask == 1)[0]
    num_valid_actions = len(valid_action_indices)

    print(f"可控房间: {controllable_rooms}")
    print(f"旧动作索引: {old_action_index}")
    print(f"Mask中有效动作的数量: {num_valid_actions}")

    # 理论上，可行动作数量应该是：(风机档位数)^(可控房间数) = 4^3 = 64
    num_controllable = np.sum(controllable_rooms)
    base = int(round(action_space.n ** (1 / 7)))
    theoretical_valid_actions = base ** num_controllable
    print(f"理论上有效动作的数量应为 {base}^{num_controllable} = {theoretical_valid_actions}")

    assert num_valid_actions == theoretical_valid_actions
    print("[成功] 实际有效动作数与理论值相符！")

    # 验证逻辑：所有有效动作，其不可控房间的设定都应与old_action_index一致
    base = 4
    is_check_passed = True
    print("old action:", action_index_to_array(old_action_index,[4, 4, 4, 4, 4, 4, 4]))
    for valid_idx in valid_action_indices:
        # 检查每个不可控房间
        print(f"检查有效动作索引 {action_index_to_array(valid_idx,[4, 4, 4, 4, 4, 4, 4])}:")
        for room_idx, is_controllable in enumerate(controllable_rooms):
            if not is_controllable:
                multiplier = base ** room_idx
                # 获取有效动作中，该房间的设定
                new_room_action = (valid_idx // multiplier) % base
                # 获取旧动作中，该房间的设定
                old_room_action = (old_action_index // multiplier) % base

                if new_room_action != old_room_action:
                    print(f"[失败!] 在有效动作索引 {valid_idx} 中, 不可控房间 {room_idx + 1} 的设定错误。")
                    is_check_passed = False
                    break
        if not is_check_passed:
            break

    if is_check_passed:
        print("[成功] 所有有效动作均满足不可控房间设定不变的约束。")

