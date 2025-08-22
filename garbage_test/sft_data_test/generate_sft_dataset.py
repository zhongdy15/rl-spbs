import pandas as pd
import json
import os
import glob

# --- 配置参数 ---
ROOM_IDS = [1, 2, 4, 5, 6, 7] # 房间ID从1到7
DATA_DIR = '../expert_data_test/expert_data_clean'  # 历史数据文件目录
OUTPUT_DIR = 'sft_data'  # SFT数据集输出目录

# 舒适度阈值
COMFORT_TEMP_MIN = 22.0
COMFORT_TEMP_MAX = 26.0
COMFORT_RH_MIN = 40.0
COMFORT_RH_MAX = 60.0

MAX_OUTPUT_ROOMS = 4  # LLM输出中最多包含的房间数量


# --- 辅助函数 ---

def calculate_room_priority_score_and_reasons(room_data):
    """
    计算单个房间的优先级分数和需要改变的理由。
    分数越高，表示房间越需要关注。
    """
    avg_temp = (room_data['room_temp1'] + room_data['room_temp2']) / 2
    avg_rh = (room_data['room_RH1'] + room_data['room_RH2']) / 2
    occupant_num = room_data['occupant_num']

    score = 0.0
    reasons = []

    # 温度相关的理由和分数
    if avg_temp > COMFORT_TEMP_MAX:
        score += (avg_temp - COMFORT_TEMP_MAX) * 5.0  # 温度过高，高优先级
        reasons.append("温度过高")
    elif avg_temp < COMFORT_TEMP_MIN:
        score += (COMFORT_TEMP_MIN - avg_temp) * 3.0  # 温度过低，中等优先级
        reasons.append("温度过低")

    # 湿度相关的理由和分数
    if avg_rh > COMFORT_RH_MAX:
        score += (avg_rh - COMFORT_RH_MAX) * 2.0  # 湿度过高，中等优先级
        reasons.append("湿度过高")
    elif avg_rh < COMFORT_RH_MIN:
        score += (COMFORT_RH_MIN - avg_rh) * 1.0  # 湿度过低，较低优先级
        reasons.append("湿度过低")

    # 人员数量相关的分数和理由
    if occupant_num > 0:
        score += occupant_num * 10.0  # 有人存在，显著增加优先级
        reasons.append("人员较多")

    # 如果没有明确的温湿度或人员理由，但FCU实际改变了，可能存在其他未捕捉到的专家判断，
    # 此时我们仍然会根据FCU变化本身作为理由。
    if not reasons and score == 0:
        reasons.append("当前状态良好")  # 作为默认值，通常会被FCU变化理由覆盖

    # 返回分数和理由列表
    return score, reasons


def create_sft_dataset():
    """
    创建LLM的SFT数据集。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_room_data = {}
    # 加载所有房间的历史数据
    for room_id in ROOM_IDS:
        # 假设文件名为 room_X_2021_8_21_0900_2100.csv
        specific_file = os.path.join(DATA_DIR, f'room_{room_id}_2021_8_21_0900_2100.csv')

        if not os.path.exists(specific_file):
            print(f"错误: 未找到文件 {specific_file}。请检查路径或文件名。")
            continue  # 跳过此房间，如果文件不存在

        df = pd.read_csv(specific_file)
        # 将日期和时间列合并为datetime对象，并设置为索引
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.set_index('datetime').sort_index()
        all_room_data[f'room_{room_id}'] = df
        print(f"已加载房间 {room_id} 的数据，形状: {df.shape}")

    if not all_room_data or len(all_room_data) < len(ROOM_IDS):
        print("未加载所有房间数据，或房间数量不足。退出。")
        return

    # 找到所有房间共同的时间戳，确保数据对齐
    common_timestamps = None
    for room_df in all_room_data.values():
        if common_timestamps is None:
            common_timestamps = room_df.index
        else:
            common_timestamps = common_timestamps.intersection(room_df.index)

    if common_timestamps is None or common_timestamps.empty:
        print("未在所有房间数据中找到共同的时间戳。退出。")
        return

    common_timestamps = common_timestamps.sort_values()
    print(f"找到 {len(common_timestamps)} 个共同时间戳。")

    sft_dataset = []

    # 遍历共同时间戳，从第二个时间步开始，以便与前一个时间步进行比较
    for i in range(1, len(common_timestamps)):
        current_ts = common_timestamps[i]
        prev_ts = common_timestamps[i - 1]

        current_states = {}
        prev_states = {}

        # 收集当前和前一个时间步所有房间的状态
        all_rooms_data_available = True
        for room_id in ROOM_IDS:
            room_key = f'room_{room_id}'

            # 确保当前和前一个时间戳的数据都存在于该房间的数据框中
            if current_ts not in all_room_data[room_key].index or \
                    prev_ts not in all_room_data[room_key].index:
                all_rooms_data_available = False
                break

            current_states[room_key] = all_room_data[room_key].loc[current_ts]
            prev_states[room_key] = all_room_data[room_key].loc[prev_ts]

        if not all_rooms_data_available:
            # 如果任何房间在此时间戳的数据不完整，则跳过
            continue

            # 识别FCU档位发生变化的房间，并计算其优先级分数和理由
        changed_rooms_info = []  # 存储 (room_id, score, reasons_list, current_fcu, prev_fcu)

        for room_id in ROOM_IDS:
            room_key = f'room_{room_id}'
            current_fcu_state = current_states[room_key]['FCU_combined_state']
            prev_fcu_state = prev_states[room_key]['FCU_combined_state']

            # 如果FCU档位发生变化
            if current_fcu_state != prev_fcu_state:
                score, reasons = calculate_room_priority_score_and_reasons(current_states[room_key])
                changed_rooms_info.append({
                    "room_id": room_id,
                    "score": score,
                    "reasons_list": reasons,  # 存储为列表，后续处理
                    "current_fcu": current_fcu_state,
                    "prev_fcu": prev_fcu_state
                })

        if not changed_rooms_info:
            # 如果此时间步没有房间的FCU档位发生变化，则不生成SFT样本
            continue

            # 根据计算出的分数对发生变化的房间进行降序排序
        changed_rooms_info.sort(key=lambda x: x['score'], reverse=True)

        # 准备LLM的输入（Prompt）字符串
        input_str_parts = []
        for room_id in ROOM_IDS:
            room_key = f'room_{room_id}'
            current_room_data = current_states[room_key]
            temp1 = current_room_data['room_temp1']
            temp2 = current_room_data['room_temp2']
            rh1 = current_room_data['room_RH1']
            rh2 = current_room_data['room_RH2']
            occupant = int(current_room_data['occupant_num'])
            fcu_state = int(current_room_data['FCU_combined_state'])

            input_str_parts.append(
                f"房间{room_id} - 温度: {temp1:.1f}°C/{temp2:.1f}°C, 湿度: {rh1:.1f}%/{rh2:.1f}%, "
                f"人员: {occupant}人, 当前档位: {fcu_state}"
            )

        input_prompt = "当前7个房间状态：\n" + "; ".join(input_str_parts) + \
                       f"。目标是保持舒适温度({COMFORT_TEMP_MIN}-{COMFORT_TEMP_MAX}°C)和适宜湿度({COMFORT_RH_MIN}-{COMFORT_RH_MAX}%)。"

        instruction_prompt = "基于历史操作数据，输出最需要改变档位的房间列表，按优先级排序，并简要解释原因。不要输出具体档位，只需房间ID和原因。"

        # 准备LLM的输出（Target）
        output_rooms = []
        output_reasons = []
        for j, room_info in enumerate(changed_rooms_info):
            if j >= MAX_OUTPUT_ROOMS:
                break  # 限制输出的房间数量

            output_rooms.append(room_info['room_id'])

            # 组合理由：先是状态不适的理由，再是FCU档位变化的理由
            final_reasons_for_output_list = []

            # 添加因状态不适而来的理由 (如果存在且不只是默认的“状态良好”)
            if room_info['reasons_list'] and "当前状态良好" not in room_info['reasons_list']:
                final_reasons_for_output_list.extend(room_info['reasons_list'])

            # 添加FCU档位变化的理由，这是专家实际操作的体现
            fcu_change_reason = f"FCU从{int(room_info['prev_fcu'])}档变为{int(room_info['current_fcu'])}档"
            final_reasons_for_output_list.append(fcu_change_reason)

            # 去重并以分号连接
            final_reasons_for_output_list = list(dict.fromkeys(final_reasons_for_output_list))  # 保持顺序去重
            output_reasons.append("; ".join(final_reasons_for_output_list))

        # 按照指定格式添加到SFT数据集
        sft_dataset.append([
            {
                "instruction": instruction_prompt,
                "input": input_prompt,
                "output": json.dumps({"rooms": output_rooms, "reasons": output_reasons}, ensure_ascii=False),
                "system": "你是一个智能HVAC系统控制助手，能基于房间状态和历史操作数据，识别并推荐最需要改变FCU档位的房间，以优化舒适度和能耗。"
            }
        ])

    # 保存数据集到JSON文件
    output_filepath = os.path.join(OUTPUT_DIR, 'fcu_priority_sft_dataset.json')
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=2)

    print(f"SFT数据集已成功创建于: {output_filepath}")
    print(f"总样本数: {len(sft_dataset)}")


# --- 执行数据集创建 ---
if __name__ == '__main__':
    create_sft_dataset()
