import numpy as np
from typing import List, Dict


def interpret_obs(obs: np.ndarray, frame_skip: int = 5) -> str:
    """
    解释环境观测数组 (obs)，并将其转换为人类可读的英文文本描述。

    Args:
        obs (np.ndarray): 形状为 (180,) 的观测数组。
        frame_skip (int): 帧跳数，表示 obs 中包含多少个时间步的观测。默认为 5。

    Returns:
        str: 描述环境状态的格式化英文文本。
    """
    if obs.shape[0] != 180:
        raise ValueError(f"输入数组 obs 的形状应为 (180,)，但实际为 {obs.shape}")

    # 定义观测值的键和结构
    # 这个结构必须与环境生成 obs 的方式严格匹配
    room_state_keys = [
        "room_temp",  # 房间温度
        "FCU_fan_feedback",  # FCU 挡位
        "supply_temp",  # FCU 供水温度
        "return_temp",  # FCU 回水温度
        "occupant_num",  # 房间人数
    ]

    observation_key_dict = {
        "sensor_outdoor": ["outdoor_temp"],  # 室外温度
        "room1": room_state_keys,
        "room2": room_state_keys,
        "room3": room_state_keys,
        "room4": room_state_keys,
        "room5": room_state_keys,
        "room6": room_state_keys,
        "room7": room_state_keys,
    }

    # 计算每个时间步观测值的长度
    single_frame_length = obs.shape[0] // frame_skip
    if single_frame_length != 36:
        # 进行一次校验，确保切分是正确的
        raise ValueError(f"每个时间步的观测长度应为 36 (180/5)，但计算结果为 {single_frame_length}")

    # 将 (180,) 的数组切分为 5 个 (36,) 的数组
    # obs 的结构是 [最旧, ..., 最新]，所以切分后 frames[0] 是最旧的
    frames = np.split(obs, frame_skip)

    # 我们需要从最新到最旧进行描述，所以将 frames 列表反转
    frames.reverse()  # 现在 frames[0] 是最新的观测

    final_text_parts = []

    # 遍历每一个时间步的观测 (从最新到最旧)
    for i, frame_data in enumerate(frames):
        # 构造时间描述文本
        if i == 0:
            time_desc = "Current Status (at Time T, FCU Fan Speed was active from T-1min to T):"
        else:
            time_desc = f"Status {i} minute(s) ago (at Time T-{i}min, FCU Fan Speed was active from T-{i + 1}min to T-{i}min):"

        frame_text_parts = [time_desc]
        current_index = 0

        # 解析单个时间步 (36,) 的数据
        parsed_frame: Dict[str, Dict[str, float]] = {}

        # 按照 key_dict 的顺序解析数据
        for main_key, sub_keys in observation_key_dict.items():
            parsed_frame[main_key] = {}
            for sub_key in sub_keys:
                value = frame_data[current_index]
                parsed_frame[main_key][sub_key] = value
                current_index += 1

        # 生成房间状态描述
        for room_num in range(1, 8):
            room_key = f"room{room_num}"
            room_data = parsed_frame[room_key]

            # 从解析的数据中提取需要展示的信息
            temp = room_data["room_temp"]
            occupants = int(room_data["occupant_num"])  # 人数通常是整数
            fan_speed = int(room_data["FCU_fan_feedback"])  # 挡位也是整数

            room_str = (f"Room {room_num} — Temperature: {temp:.2f}°C, "
                        f"Occupant Num: {occupants}, "
                        f"Fan Speed: {fan_speed};")
            frame_text_parts.append(room_str)

        # 生成室外温度描述
        outdoor_temp = parsed_frame["sensor_outdoor"]["outdoor_temp"]
        outdoor_str = f"Outdoor - Temperature: {outdoor_temp:.2f}°C"
        frame_text_parts.append(outdoor_str)

        # 将当前时间步的所有文本合并
        final_text_parts.append("\n".join(frame_text_parts))

    # 将所有时间步的文本用两个换行符隔开，形成最终输出
    return "\n\n".join(final_text_parts)


def get_current_fan_speed_from_obs(obs: np.ndarray, frame_skip: int = 5) -> np.ndarray:
    """
    从完整的观测数组 (obs) 中提取最新时间步的7个房间的风扇速度。

    Args:
        obs (np.ndarray): 形状为 (180,) 的观测数组。
        frame_skip (int): 帧跳数，表示 obs 中包含多少个时间步的观测。默认为 5。

    Returns:
        np.ndarray: 一个长度为 7 的 numpy 数组，包含从 room1 到 room7 的最新风扇速度。
    """
    # 1. 校验输入数组的形状
    if obs.shape[0] != 180:
        raise ValueError(f"输入数组 obs 的形状应为 (180,)，但实际为 {obs.shape}")

    # 2. 计算单个时间步（frame）的数据长度
    single_frame_length = obs.shape[0] // frame_skip
    if single_frame_length != 36:
        raise ValueError(f"每个时间步的观测长度应为 36 (180/5)，但计算结果为 {single_frame_length}")

    # 3. 提取最新时间步的数据
    # obs 数组的结构是 [最旧, ..., 最新]，所以最后一个切片是最新数据。
    # 最后一个切片的起始索引是 (frame_skip - 1) * single_frame_length
    latest_frame_start_index = (frame_skip - 1) * single_frame_length
    latest_frame = obs[latest_frame_start_index:]

    # 4. 从最新时间步中提取风扇速度
    # 根据 `interpret_obs` 函数中的数据结构，我们可以确定每个房间风扇速度的索引。
    # - 室外温度: 1个值 (索引 0)
    # - 每个房间: 5个值 (temp, fan_speed, supply, return, occupants)
    # - `FCU_fan_feedback` 是每个房间数据块中的第2个值（索引为1）。

    # room1 的风扇速度索引: 1 (室外) + 1 (room1 temp) = 2
    # room2 的风扇速度索引: 1 (室外) + 5 (room1) + 1 (room2 temp) = 7
    # room_N 的风扇速度索引: 1 + (N-1)*5 + 1
    # 公式为： 2 + (N-1)*5

    fan_speed_indices = [2 + (i * 5) for i in range(7)]

    # 直接使用高级索引从最新帧中提取所有风扇速度
    fan_speeds = latest_frame[fan_speed_indices]

    return fan_speeds

if __name__ == "__main__":
    # 创建一个随机的模拟 obs 数组用于测试
    # 实际使用时，这个 obs 会由您的环境提供
    mock_obs = np.array([24.21     , 26.043118 ,  3.       ,  7.       ,  7.5617833,
        0.       , 26.820894 ,  0.       ,  7.       ,  7.       ,
        1.       , 26.255205 ,  2.       ,  7.       , 10.011565 ,
        1.       , 26.642696 ,  0.       ,  7.       ,  7.       ,
        2.       , 26.472988 ,  0.       ,  7.       ,  7.       ,
        2.       , 26.406898 ,  2.       ,  7.       ,  8.505783 ,
        4.       , 26.425941 ,  0.       ,  7.       ,  7.       ,
        0.       , 24.21     , 25.624416 ,  3.       ,  7.       ,
        7.5412498,  0.       , 27.136099 ,  0.       ,  7.       ,
        7.       ,  1.       , 26.051714 ,  2.       ,  7.       ,
        9.952588 ,  2.       , 26.806084 ,  0.       ,  7.       ,
        7.       ,  3.       , 26.423903 ,  0.       ,  7.       ,
        7.       ,  1.       , 26.318666 ,  2.       ,  7.       ,
        8.494567 ,  4.       , 26.354914 ,  0.       ,  7.       ,
        7.       ,  0.       , 24.21     , 25.240803 ,  3.       ,
        7.       ,  7.5224323,  0.       , 27.445732 ,  0.       ,
        7.       ,  7.       ,  1.       , 25.839266 ,  2.       ,
        7.       ,  9.903562 ,  1.       , 26.966154 ,  0.       ,
        7.       ,  7.       ,  3.       , 26.377039 ,  0.       ,
        7.       ,  7.       ,  1.       , 26.211382 ,  2.       ,
        7.       ,  8.483939 ,  3.       , 26.286812 ,  0.       ,
        7.       ,  7.       ,  0.       , 24.21     , 24.889431 ,
        3.       ,  7.       ,  7.5051913,  0.       , 27.749908 ,
        0.       ,  7.       ,  7.       ,  1.       , 25.666338 ,
        2.       ,  7.       ,  9.852378 ,  2.       , 27.099894 ,
        0.       ,  7.       ,  7.       ,  2.       , 26.355957 ,
        0.       ,  7.       ,  7.       ,  2.       , 26.10953  ,
        2.       ,  7.       ,  8.471015 ,  3.       , 26.221535 ,
        0.       ,  7.       ,  7.       ,  0.       , 24.21     ,
       24.567688 ,  3.       ,  7.       ,  7.4894   ,  0.       ,
       28.048738 ,  0.       ,  7.       ,  7.       ,  1.       ,
       25.50631  ,  2.       ,  7.       ,  9.810715 ,  2.       ,
       27.254095 ,  0.       ,  7.       ,  7.       ,  3.       ,
       26.31247  ,  0.       ,  7.       ,  7.       ,  1.       ,
       26.012857 ,  2.       ,  7.       ,  8.458746 ,  3.       ,
       26.158981 ,  0.       ,  7.       ,  7.       ,  0.       ])

    # 您可以替换为实际的 obs 数据
    # obs = ... (从您的环境中获取的实际 numpy array)

    # 调用函数并打印结果
    explanation = interpret_obs(mock_obs, frame_skip=5)
    print(explanation)

    current_fan_speeds = get_current_fan_speed_from_obs(mock_obs, frame_skip=5)

    print("提取到的最新风扇速度:")
    print(current_fan_speeds)
    print(f"类型: {type(current_fan_speeds)}")
    print(f"形状: {current_fan_speeds.shape}")

    # # 作为验证，我们可以检查 `interpret_obs` 的输出
    # print("\n----------------------------------\n")
    # print("使用 interpret_obs 函数的完整输出（仅用于验证）:")
    # explanation = interpret_obs(mock_obs, frame_skip=5)
    # print(explanation)
