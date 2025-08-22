import pandas as pd
import json
import os

# --- 配置参数 ---
DATA_DIR = '../../expert_data/expert_data_downsampling'  # 数据文件存放目录
ROOM_IDS = list(range(1, 8))  # 房间ID列表，从 room1 到 room7
DATES = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]  # 8月份的日期
YEAR = 2021
MONTH = 8
TIME_RANGE = "0900_2100"  # 每个文件的有效时间范围

# 舒适度目标范围
TARGET_TEMP_MIN = 22
TARGET_TEMP_MAX = 26
TARGET_RH_MIN = 40
TARGET_RH_MAX = 60

# 用于计算不适度分数的权重 (可调)
WEIGHTS_FOR_SCORE = {
    'temp': 2.0,  # 温度偏差权重较高
    'rh': 1.0,  # 湿度偏差权重适中
    'occupant': 0.5  # 人员数量作为不适度加剧因子
}
NUM_ROOMS_TO_OUTPUT = 3  # LLM输出的优先级房间数量 K


# --- 辅助函数：计算房间信息 (分数和原因) ---
def calculate_room_info(room_id, room_data, prev_fcu_state, current_fcu_state):
    """
    计算给定房间当前状态的不适度分数和需要改变的原因。
    参数:
        room_id (int): 房间ID。
        room_data (dict): 包含 'room_temp1', 'room_RH1', 'occupant_num' 的当前房间状态数据。
        prev_fcu_state (float): 上一个时间步的FCU档位。
        current_fcu_state (float): 当前时间步的FCU档位。
    返回:
        dict: 包含 'room_id', 'score', 'reason', 'is_fcu_changed' 等信息的字典。
    """
    temp = room_data['room_temp1']
    rh = room_data['room_RH1']
    occupant = room_data['occupant_num']

    temp_dev = 0
    reason_temp = ""
    if temp > TARGET_TEMP_MAX:
        temp_dev = temp - TARGET_TEMP_MAX
        reason_temp = "温度过高"
    elif temp < TARGET_TEMP_MIN:
        temp_dev = TARGET_TEMP_MIN - temp
        reason_temp = "温度过低"

    rh_dev = 0
    reason_rh = ""
    if rh > TARGET_RH_MAX:
        rh_dev = rh - TARGET_RH_MAX
        reason_rh = "湿度过高"
    elif rh < TARGET_RH_MIN:
        rh_dev = TARGET_RH_MIN - rh
        reason_rh = "湿度过低"

    # 不适度分数计算：分数越高表示越不舒适或越需要调整
    score = (temp_dev * WEIGHTS_FOR_SCORE['temp']) + \
            (rh_dev * WEIGHTS_FOR_SCORE['rh']) + \
            (occupant * WEIGHTS_FOR_SCORE['occupant'])

    reasons_list = []
    if reason_temp:
        reasons_list.append(reason_temp)
    if reason_rh:
        reasons_list.append(reason_rh)

    # 如果有不适度且有人员，则加入人员因素
    if occupant > 0 and (reason_temp or reason_rh):
        reasons_list.append("人员较多")

    # 判断FCU档位是否发生变化
    is_fcu_changed = (current_fcu_state != prev_fcu_state)

    # 如果FCU档位被调低，且房间处于舒适区，则可能是能耗优化
    if is_fcu_changed and current_fcu_state < prev_fcu_state and not (reason_temp or reason_rh):
        reasons_list.append("能耗优化潜力")

    # 如果没有任何明确原因，提供一个通用理由 (通常不会发生，因为我们只关注专家调整过的房间)
    final_reason = "，".join(reasons_list) if reasons_list else "专家进行了调整"

    return {
        'room_id': room_id,
        'score': score,
        'reason': final_reason,
        'is_fcu_changed': is_fcu_changed,
        'current_fcu_state': current_fcu_state
    }


# --- 主要数据集生成逻辑 ---
sft_dataset = []  # 存储所有SFT样本

for day in DATES:
    print(f"正在处理日期: {YEAR}-{MONTH:02d}-{day:02d}")

    # 1. 加载当前日期所有房间的数据
    room_data_dfs = {}
    try:
        for room_id in ROOM_IDS:
            file_path = os.path.join(DATA_DIR, f"room_{room_id}_{YEAR}_{MONTH}_{day}_{TIME_RANGE}.csv")
            df = pd.read_csv(file_path)
            # 将 'time' 列转换为 datetime.time 对象以便排序和对齐
            df['time_dt'] = pd.to_datetime(df['time'], format='%H:%M').dt.time
            df = df.sort_values(by='time_dt').reset_index(drop=True)  # 确保按时间排序
            room_data_dfs[room_id] = df
    except FileNotFoundError as e:
        print(f"警告: 找不到日期 {day} 的部分或全部房间数据文件。跳过此日期。错误: {e}")
        continue
    except Exception as e:
        print(f"加载日期 {day} 数据时发生错误: {e}。跳过此日期。")
        continue

    # 获取共同的时间步（假设所有房间文件的时间步是同步的）
    time_steps_dt = room_data_dfs[1]['time_dt'].tolist()

    # 2. 遍历每个时间步，从第二个时间步开始 (需要前一个时间步的数据)
    for i in range(1, len(time_steps_dt)):
        current_time_str = time_steps_dt[i].strftime('%H:%M')
        previous_time_str = time_steps_dt[i - 1].strftime('%H:%M')

        # 提取户外温度 (假设同一时间点所有房间的户外温度一致)
        outdoor_temp = room_data_dfs[1].loc[i, 'outdoor_temp']

        current_states_all_rooms = {}
        previous_fcu_states_all_rooms = {}

        # 收集所有房间在当前和前一个时间步的状态
        for room_id in ROOM_IDS:
            current_row = room_data_dfs[room_id].loc[i]
            previous_row = room_data_dfs[room_id].loc[i - 1]

            current_states_all_rooms[room_id] = {
                'room_temp1': current_row['room_temp1'],
                'room_RH1': current_row['room_RH1'],
                'occupant_num': current_row['occupant_num'],
                'FCU_combined_state': current_row['FCU_combined_state']
            }
            previous_fcu_states_all_rooms[room_id] = previous_row['FCU_combined_state']

        # 3. 识别专家调整过的房间并计算其信息
        expert_changed_room_info = []  # 存储发生变化的房间信息
        for room_id in ROOM_IDS:
            info = calculate_room_info(
                room_id=room_id,
                room_data=current_states_all_rooms[room_id],
                prev_fcu_state=previous_fcu_states_all_rooms[room_id],
                current_fcu_state=current_states_all_rooms[room_id]['FCU_combined_state']
            )
            if info['is_fcu_changed']:
                expert_changed_room_info.append(info)

        # 如果当前时间步没有任何房间被专家调整，则不生成SFT样本
        if not expert_changed_room_info:
            continue

        # 4. 排序并选择优先级最高的 K 个房间
        expert_changed_room_info.sort(key=lambda x: x['score'], reverse=True)  # 按不适度分数降序排序
        selected_rooms_for_output = expert_changed_room_info[:NUM_ROOMS_TO_OUTPUT]

        # 5. 构建SFT样本的 'instruction', 'input', 'output'
        instruction = "根据当前环境和房间状态，预测最需要改变控制档位的几个房间，按优先级排序，并简要解释原因。"

        # 构建 'input' 字符串
        input_string = f"户外温度: {outdoor_temp:.1f}°C。\n当前7个房间的状态如下：\n"
        for room_id in ROOM_IDS:
            room_data = current_states_all_rooms[room_id]
            input_string += (
                f"房间{room_id} - 温度: {room_data['room_temp1']:.1f}°C, "
                f"湿度: {room_data['room_RH1']:.1f}%, "
                f"人员: {room_data['occupant_num']:.0f}人, "
                f"当前档位: {room_data['FCU_combined_state']:.0f}。\n"
            )

        # 构建 'output' JSON数据结构
        output_rooms_list = [info['room_id'] for info in selected_rooms_for_output]
        output_reasons_list = [f"房间{info['room_id']}{info['reason']}" for info in selected_rooms_for_output]

        output_data = [{"rooms": output_rooms_list, "reasons": output_reasons_list}]
        # 将 output_data 转换为JSON字符串
        output_json_string = json.dumps(output_data, ensure_ascii=False)

        # 按照SFT要求格式添加到数据集
        sft_dataset.append([
            {
                "instruction": instruction,
                "input": input_string,
                "output": output_json_string,
                "system": None,  # 系统提示词，此处可为空
                "history": []  # 历史对话记录，此处为空
            }
        ])

# --- 保存数据集到JSON文件 ---
output_file_path = 'sft_priority_room_dataset.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(sft_dataset, f, ensure_ascii=False, indent=2)

print(f"\n数据集生成完成。总样本数: {len(sft_dataset)}")
print(f"数据集已保存至: {output_file_path}")
