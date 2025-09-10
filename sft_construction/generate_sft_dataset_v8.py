import pandas as pd
import json
import os
from datetime import datetime
import random
import numpy as np  # 导入 numpy 用于 NaN

# --- Constants ---
# SYSTEM_PROMPT 和 INSTRUCTION_PROMPT 的内容将从上述更新的文件中读取。
with open("predefined_prompt/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open("predefined_prompt/instruction_prompt.txt", "r", encoding="utf-8") as f:
    INSTRUCTION_PROMPT = f.read()

OPTIMAL_TEMP_MIN = 25
OPTIMAL_TEMP_MAX = 27

DATA_DIR = "../expert_data/expert_data_remap"
# 建议为新的数据集格式更改输出目录名称
OUTPUT_DIR = "all_types_sft_data_v8_occupant_change"

ROOM_IDS = range(1, 8)
DAYS = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]  # 8月份的特定天

N_PREV_OCCUPANT_CHANGE_STATES = 2  # 设置N值为2

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and Aggregate Data ---
room_dfs = {}

print("Loading and preprocessing data for each room...")
for room_id in ROOM_IDS:
    room_data_frames = []
    for day in DAYS:
        file_path = os.path.join(DATA_DIR, f"room_{room_id}_2021_8_{day}_0900_2100.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            room_data_frames.append(df)
        else:
            pass

    if room_data_frames:
        combined_df = pd.concat(room_data_frames, ignore_index=True)
        combined_df['datetime'] = pd.to_datetime(combined_df['date'] + ' ' + combined_df['time'],
                                                 format='%Y/%m/%d %H:%M')
        combined_df = combined_df.sort_values(by='datetime').reset_index(drop=True)

        combined_df['mean_room_temp'] = (combined_df['room_temp1'] + combined_df['room_temp2']) / 2

        # --- 原始的上一时间步状态 (T-5min)，用于输出逻辑中的FCU变化判断 ---
        # 即使input_str不再包含T-5min状态，output_str的生成仍依赖于此
        combined_df['prev_FCU_combined_state'] = combined_df['FCU_combined_state'].shift(1)
        # 原始的prev_outdoor_temp也保留，如果需要用于output_str的context
        combined_df['prev_outdoor_temp'] = combined_df['outdoor_temp'].shift(1)

        # --- 新逻辑：跟踪N次人数变化时的状态 ---
        for k in range(1, N_PREV_OCCUPANT_CHANGE_STATES + 1):
            combined_df[f'prev_occupant_change_state_{k}_temp'] = np.nan
            combined_df[f'prev_occupant_change_state_{k}_occupant'] = np.nan
            combined_df[f'prev_occupant_change_state_{k}_fcu'] = np.nan
            # 存储人数变化时的室外温度和时间
            combined_df[f'prev_occupant_change_state_{k}_outdoor_temp'] = np.nan
            combined_df[f'prev_occupant_change_state_{k}_datetime'] = pd.NaT

        # 跟踪此房间人数变化历史的列表
        # 存储 (datetime, temp, occupant_num, FCU_combined_state, outdoor_temp)
        occupant_change_history = []

        for idx, row in combined_df.iterrows():
            if idx > 0:
                prev_row = combined_df.loc[idx - 1]

                # 检查当前时间步的occupant_num是否与上一个时间步不同
                if row['occupant_num'] != prev_row['occupant_num']:
                    # 如果不同，则将“上一个时间步”的状态记录为一次人数变化的状态
                    occupant_change_history.append({
                        'datetime': prev_row['datetime'],
                        'temp': prev_row['mean_room_temp'],
                        'occupant': prev_row['occupant_num'],
                        'fcu': prev_row['FCU_combined_state'],
                        'outdoor_temp': prev_row['outdoor_temp']
                    })
                    # 保持历史记录只包含最近的 N_PREV_OCCUPANT_CHANGE_STATES 个状态
                    if len(occupant_change_history) > N_PREV_OCCUPANT_CHANGE_STATES:
                        occupant_change_history.pop(0)  # 移除最旧的记录

                # 为当前行 (idx) 填充 N 个之前人数变化的状态
                # history列表是从旧到新排序，因此 -k 索引获取的是倒数第k个（即第k个最新的变化）
                for k in range(1, N_PREV_OCCUPANT_CHANGE_STATES + 1):
                    if len(occupant_change_history) >= k:
                        state_data = occupant_change_history[-k]
                        combined_df.loc[idx, f'prev_occupant_change_state_{k}_temp'] = state_data['temp']
                        combined_df.loc[idx, f'prev_occupant_change_state_{k}_occupant'] = state_data['occupant']
                        combined_df.loc[idx, f'prev_occupant_change_state_{k}_fcu'] = state_data['fcu']
                        combined_df.loc[idx, f'prev_occupant_change_state_{k}_outdoor_temp'] = state_data[
                            'outdoor_temp']
                        combined_df.loc[idx, f'prev_occupant_change_state_{k}_datetime'] = state_data['datetime']

        room_dfs[room_id] = combined_df
    else:
        print(f"No data loaded for room {room_id}. This room will be excluded from common timestamps consideration.")

# Find common timestamps across all rooms
print("Finding common timestamps across all rooms...")
if not room_dfs:
    print("No room data loaded. Exiting.")
    exit()

first_room_id_with_data = next((rid for rid in ROOM_IDS if rid in room_dfs), None)
if first_room_id_with_data is None:
    print("No data available for any room. Exiting.")
    exit()

common_timestamps_set = set(room_dfs[first_room_id_with_data]['datetime'])

for room_id in ROOM_IDS:
    if room_id in room_dfs:
        common_timestamps_set = common_timestamps_set.intersection(set(room_dfs[room_id]['datetime']))
    else:
        pass

common_timestamps = sorted(list(common_timestamps_set))

print(f"Found {len(common_timestamps)} common timestamps where all {len(ROOM_IDS)} rooms have data.")

# Initialize lists for SFT data categories
sft_data_type1 = []
sft_data_type2 = []
sft_data_type3 = []

# --- Generate SFT Data Entries ---
print("Generating SFT data entries...")
# 需要至少一个时间戳用于当前状态，并且prev_FCU_combined_state的有效性也要求至少从第二个时间戳开始
if len(common_timestamps) >= 1:
    for i in range(1, len(common_timestamps)):  # 从1开始，以确保prev_FCU_combined_state（T-5min）通常可用
        timestamp = common_timestamps[i]

        current_status_lines = []
        current_room_states = {}

        previous_fcu_states = {}  # 存储T-5min的FCU状态，用于output_str的逻辑判断
        current_fcu_states = {}

        is_valid_entry = True
        current_outdoor_temp = None

        # 获取当前时间步的室外温度（假设所有房间一致）
        outdoor_temp_data_row = room_dfs[ROOM_IDS[0]].loc[room_dfs[ROOM_IDS[0]]['datetime'] == timestamp]
        if not outdoor_temp_data_row.empty:
            current_outdoor_temp = outdoor_temp_data_row['outdoor_temp'].iloc[0]
        else:
            is_valid_entry = False

        if not is_valid_entry:
            continue

        for room_id in ROOM_IDS:
            room_data_at_ts = room_dfs[room_id].loc[room_dfs[room_id]['datetime'] == timestamp]

            if room_data_at_ts.empty:
                is_valid_entry = False
                break

            curr_temp = room_data_at_ts['mean_room_temp'].iloc[0]
            curr_occupant = room_data_at_ts['occupant_num'].iloc[0]
            curr_fan_speed = room_data_at_ts['FCU_combined_state'].iloc[0]

            # 获取T-5min的FCU状态，用于输出逻辑的FCU变化判断
            prev_fan_speed_t_minus_5min = room_data_at_ts['prev_FCU_combined_state'].iloc[0]
            if pd.isna(prev_fan_speed_t_minus_5min):
                is_valid_entry = False  # 如果T-5min FCU状态为NaN，则无法判断专家调整
                break

            current_status_lines.append(
                f"Room {room_id} — Temperature: {curr_temp:.2f}°C, Occupant Num: {int(curr_occupant)}, Fan Speed: {int(curr_fan_speed)};"
            )

            current_room_states[room_id] = {
                'temp': curr_temp,
                'occupant': curr_occupant,
                'fan_speed': curr_fan_speed
            }
            previous_fcu_states[room_id] = prev_fan_speed_t_minus_5min
            current_fcu_states[room_id] = curr_fan_speed

        if not is_valid_entry:
            continue

        # --- 构建 input_str，包含当前状态和N次人数变化状态 ---
        input_str_parts = []
        input_str_parts.append(
            f"You will be provided with real-time status at the current moment (T), and also up to N={N_PREV_OCCUPANT_CHANGE_STATES} previous states where the occupant number changed.\n\n")

        # 当前状态
        input_str_parts.append("Current Status (at Time T, FCU Fan Speed was active from T-5min to T):")
        input_str_parts.extend(current_status_lines)
        input_str_parts.append(f"Outdoor - Temperature: {current_outdoor_temp:.2f}°C")
        input_str_parts.append("\n")

        # 之前人数变化的状态
        input_str_parts.append(
            f"Previous Occupant Change States (N={N_PREV_OCCUPANT_CHANGE_STATES}, ordered from most recent change to oldest, NaN if not available):")

        for k in range(1, N_PREV_OCCUPANT_CHANGE_STATES + 1):
            state_lines = []
            has_valid_state_for_this_k = False

            # 收集所有房间的第k次人数变化状态信息
            prev_occ_state_info_for_k = []
            for room_id in ROOM_IDS:
                room_data_at_ts = room_dfs[room_id].loc[room_dfs[room_id]['datetime'] == timestamp]

                prev_occ_temp = room_data_at_ts[f'prev_occupant_change_state_{k}_temp'].iloc[0]
                prev_occ_occupant = room_data_at_ts[f'prev_occupant_change_state_{k}_occupant'].iloc[0]
                prev_occ_fcu = room_data_at_ts[f'prev_occupant_change_state_{k}_fcu'].iloc[0]
                prev_occ_dt = room_data_at_ts[f'prev_occupant_change_state_{k}_datetime'].iloc[0]
                prev_occ_outdoor_temp = room_data_at_ts[f'prev_occupant_change_state_{k}_outdoor_temp'].iloc[0]

                prev_occ_state_info_for_k.append({
                    'room_id': room_id,
                    'temp': prev_occ_temp,
                    'occupant': prev_occ_occupant,
                    'fcu': prev_occ_fcu,
                    'datetime': prev_occ_dt,
                    'outdoor_temp': prev_occ_outdoor_temp
                })

            # 检查是否有任何房间有有效的第k次人数变化状态
            for room_state_data in prev_occ_state_info_for_k:
                if pd.notna(room_state_data['temp']):
                    has_valid_state_for_this_k = True
                    break

            if has_valid_state_for_this_k:
                input_str_parts.append(f"State {k} (Time of Change: Most Recent = State 1):")

                valid_outdoor_temps_for_this_k_state = []
                for room_state_data in prev_occ_state_info_for_k:
                    room_id = room_state_data['room_id']
                    if pd.isna(room_state_data['temp']):
                        state_lines.append(f"Room {room_id} — NaN;")
                    else:
                        state_lines.append(
                            f"Room {room_id} — (at {room_state_data['datetime'].strftime('%H:%M')}) "
                            f"Temperature: {room_state_data['temp']:.2f}°C, "
                            f"Occupant Num: {int(room_state_data['occupant'])}, "
                            f"Fan Speed: {int(room_state_data['fcu'])};"
                        )
                        if pd.notna(room_state_data['outdoor_temp']):
                            valid_outdoor_temps_for_this_k_state.append(room_state_data['outdoor_temp'])

                input_str_parts.extend(state_lines)

                # 室外温度是全局状态，这里取所有有效室外温度的平均值（或第一个有效值）
                if valid_outdoor_temps_for_this_k_state:
                    input_str_parts.append(
                        f"Outdoor - Temperature: {np.mean(valid_outdoor_temps_for_this_k_state):.2f}°C")
                else:
                    input_str_parts.append(f"Outdoor - Temperature: NaN°C")
                input_str_parts.append("\n")
            else:  # 没有房间有有效的第k次人数变化状态
                input_str_parts.append(f"State {k} (Time of Change: Most Recent = State 1): NaN")
                input_str_parts.append(f"Outdoor - Temperature: NaN°C")
                input_str_parts.append("\n")

        input_str = "\n".join(input_str_parts)

        # --- output_json 生成逻辑 (保持不变，仍使用T-5min与T的FCU状态进行判断) ---
        expert_adjusted_room_ids = []
        for room_id in ROOM_IDS:
            if current_fcu_states[room_id] != previous_fcu_states[room_id]:
                expert_adjusted_room_ids.append(room_id)

        confidence_scores_array = []
        for room_id in ROOM_IDS:
            confidence = 0.0
            if room_id in expert_adjusted_room_ids:
                confidence = round(random.uniform(0.9, 0.99), 2)
            else:
                confidence = round(random.uniform(0.01, 0.1), 2)
            confidence_scores_array.append(confidence)

        explanation = ""

        output_object = {
            "confidence_scores": confidence_scores_array,
            "explanation_and_thought_process": explanation
        }
        output_str = json.dumps(output_object, indent=2, ensure_ascii=False)
        # --- output_str 生成逻辑结束 ---

        # 以下的分类逻辑 (is_type3_category, is_type1_category) 仍然有效，
        # 它们是根据输入和专家行为对样本进行分类，而不是输出格式。
        # 因此保持不变，可以继续用于分析或平衡数据集。
        is_type3_category = False
        for room_id in ROOM_IDS:
            if current_fcu_states[room_id] != previous_fcu_states[room_id]:
                is_type3_category = True
                break

        is_type1_category = True
        if not is_type3_category:
            for room_id in ROOM_IDS:
                if not (current_fcu_states[room_id] == 0 and previous_fcu_states[room_id] == 0):
                    is_type1_category = False
                    break
        else:
            is_type1_category = False

        sft_entry = [{
            "instruction": INSTRUCTION_PROMPT,
            "input": input_str,
            "output": output_str,
            "system": SYSTEM_PROMPT,
            "history": []
        }]

        if is_type3_category:
            sft_data_type3.append(sft_entry)
        elif is_type1_category:
            sft_data_type1.append(sft_entry)
        else:
            sft_data_type2.append(sft_entry)

else:
    print(
        "Not enough common timestamps to generate SFT data (need at least one for current state and previous for output logic).")

print(f"Generated {len(sft_data_type1)} Type 1 entries.")
print(f"Generated {len(sft_data_type2)} Type 2 entries.")
print(f"Generated {len(sft_data_type3)} Type 3 entries.")

# --- Save SFT Data ---
print("Saving SFT data to JSON files...")

# Unpack the single-element lists before dumping
with open(os.path.join(OUTPUT_DIR, "type1_stable_off.json"), "w", encoding="utf-8") as f:
    json.dump([item[0] for item in sft_data_type1], f, indent=2, ensure_ascii=False)
print(f"Saved Type 1 data to {os.path.join(OUTPUT_DIR, 'type1_stable_off.json')}")

with open(os.path.join(OUTPUT_DIR, "type2_stable_on.json"), "w", encoding="utf-8") as f:
    json.dump([item[0] for item in sft_data_type2], f, indent=2, ensure_ascii=False)
print(f"Saved Type 2 data to {os.path.join(OUTPUT_DIR, 'type2_stable_on.json')}")

with open(os.path.join(OUTPUT_DIR, "type3_adjusted.json"), "w", encoding="utf-8") as f:
    json.dump([item[0] for item in sft_data_type3], f, indent=2, ensure_ascii=False)
print(f"Saved Type 3 data to {os.path.join(OUTPUT_DIR, 'type3_adjusted.json')}")

print("SFT dataset construction complete.")
