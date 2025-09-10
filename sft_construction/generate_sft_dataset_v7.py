import pandas as pd
import json
import os
from datetime import datetime
import random # 导入 random 模块用于生成随机置信度

# --- Constants ---
# SYSTEM_PROMPT 和 INSTRUCTION_PROMPT 的内容将从上述更新的文件中读取。
with open("predefined_prompt/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open("predefined_prompt/instruction_prompt.txt", "r", encoding="utf-8") as f:
    INSTRUCTION_PROMPT = f.read()

OPTIMAL_TEMP_MIN = 25
OPTIMAL_TEMP_MAX = 27

DATA_DIR = "../expert_data/expert_data_remap"
OUTPUT_DIR = "all_types_sft_data_v7"

ROOM_IDS = range(1, 8)
DAYS = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]  # 8月份的特定天

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and Aggregate Data ---
room_dfs = {}

print("Loading and preprocessing data for each room...")
for room_id in ROOM_IDS:
    room_data_frames = []
    for day in DAYS:
        # Construct file path based on provided pattern
        file_path = os.path.join(DATA_DIR, f"room_{room_id}_2021_8_{day}_0900_2100.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            room_data_frames.append(df)
        else:
            # print(f"Warning: File not found for room {room_id}, day {day}: {file_path}") # 可以取消注释进行调试
            pass

    if room_data_frames:
        combined_df = pd.concat(room_data_frames, ignore_index=True)
        # Convert date and time columns to a single datetime object
        combined_df['datetime'] = pd.to_datetime(combined_df['date'] + ' ' + combined_df['time'],
                                                 format='%Y/%m/%d %H:%M')
        # Sort by datetime to ensure correct 'shift' for previous state
        combined_df = combined_df.sort_values(by='datetime').reset_index(drop=True)

        # Calculate previous states for categorization and output logic
        # 新增：获取上一时间步的温度、占用人数和室外温度
        combined_df['mean_room_temp'] = (combined_df['room_temp1'] + combined_df['room_temp2']) / 2
        combined_df['prev_FCU_combined_state'] = combined_df['FCU_combined_state'].shift(1)
        combined_df['prev_room_temp'] = combined_df['mean_room_temp'].shift(1)
        combined_df['prev_occupant_num'] = combined_df['occupant_num'].shift(1)
        # 确保 outdoor_temp 列存在且有 shift
        combined_df['prev_outdoor_temp'] = combined_df['outdoor_temp'].shift(1)
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
        # 如果某个房间没有数据，则它不会对 common_timestamps_set 产生影响，但后续生成数据时会跳过
        pass

common_timestamps = sorted(list(common_timestamps_set))

print(f"Found {len(common_timestamps)} common timestamps where all 7 rooms have data.")

# Initialize lists for SFT data categories (这些分类仍可用于数据分析或平衡，与新输出格式不冲突)
sft_data_type1 = []  # All rooms: prev FCU == 0 and curr FCU == 0
sft_data_type2 = []  # All rooms: prev FCU == curr FCU but not all are 0; OR mixed stable
sft_data_type3 = []  # At least one room: prev FCU != curr FCU

# --- Generate SFT Data Entries ---
print("Generating SFT data entries...")
if len(common_timestamps) > 1:
    for i in range(1, len(common_timestamps)):
        timestamp = common_timestamps[i]

        current_status_lines = []
        prev_status_lines = []
        current_room_states = {}
        previous_fcu_states = {} # 这是 (T-10min, T-5min] 的 FCU 档位
        current_fcu_states = {}  # 这是 (T-5min, T] 的 FCU 档位

        is_valid_entry = True
        current_outdoor_temp = None
        previous_outdoor_temp = None

        # 获取当前和上一时间步的室外温度
        outdoor_temp_data_row = room_dfs[ROOM_IDS[0]].loc[room_dfs[ROOM_IDS[0]]['datetime'] == timestamp]
        if not outdoor_temp_data_row.empty:
            current_outdoor_temp = outdoor_temp_data_row['outdoor_temp'].iloc[0]
            previous_outdoor_temp = outdoor_temp_data_row['prev_outdoor_temp'].iloc[0]
        else:
            is_valid_entry = False

        if pd.isna(previous_outdoor_temp): # 检查上一时间步的室外温度是否有效
            is_valid_entry = False

        if not is_valid_entry:
            continue # 跳过此时间戳，如果室外温度数据缺失

        for room_id in ROOM_IDS:
            room_data_at_ts = room_dfs[room_id].loc[room_dfs[room_id]['datetime'] == timestamp]

            if room_data_at_ts.empty:
                is_valid_entry = False
                break

            # 获取当前时间步的数据
            curr_temp = room_data_at_ts['mean_room_temp'].iloc[0]
            curr_occupant = room_data_at_ts['occupant_num'].iloc[0]
            curr_fan_speed = room_data_at_ts['FCU_combined_state'].iloc[0]

            # 获取上一时间步的数据
            prev_temp = room_data_at_ts['prev_room_temp'].iloc[0]
            prev_occupant = room_data_at_ts['prev_occupant_num'].iloc[0]
            prev_fan_speed = room_data_at_ts['prev_FCU_combined_state'].iloc[0] # 这是 (T-10min, T-5min] 的档位

            # 检查是否有 NaN 值，这通常发生在时间序列的开头
            if pd.isna(prev_fan_speed) or pd.isna(prev_temp) or pd.isna(prev_occupant):
                is_valid_entry = False
                break

            current_status_lines.append(
                f"Room {room_id} — Temperature: {curr_temp:.2f}°C, Occupant Num: {int(curr_occupant)}, Fan Speed: {int(curr_fan_speed)};"
            )
            prev_status_lines.append(
                f"Room {room_id} — Temperature: {prev_temp:.2f}°C, Occupant Num: {int(prev_occupant)}, Fan Speed: {int(prev_fan_speed)};"
            )

            current_room_states[room_id] = {
                'temp': curr_temp,
                'occupant': curr_occupant,
                'fan_speed': curr_fan_speed
            }
            previous_fcu_states[room_id] = prev_fan_speed
            current_fcu_states[room_id] = curr_fan_speed

        if not is_valid_entry:
            continue

        # 构建完整的 input_str，包含当前和上一时间步的状态
        input_str = (
            "You will be provided with real-time status at the current moment (T), and also the status from 5 minutes ago (T-5min).\n\n"
            "Current Status (at Time T, FCU Fan Speed was active from T-5min to T):\n" +
            "\n".join(current_status_lines) +
            f"\nOutdoor - Temperature: {current_outdoor_temp:.2f}°C\n\n"
            "Status 5 minutes ago (at Time T-5min, FCU Fan Speed was active from T-10min to T-5min):\n" +
            "\n".join(prev_status_lines) +
            f"\nOutdoor - Temperature: {previous_outdoor_temp:.2f}°C"
        )


        # --- 修改原始的 output_json 生成逻辑以适应新格式 ---
        # 确定专家在当前时间戳实际调整了哪些房间 (即 FCU 状态发生变化)
        expert_adjusted_room_ids = []
        for room_id in ROOM_IDS:
            if current_fcu_states[room_id] != previous_fcu_states[room_id]:
                expert_adjusted_room_ids.append(room_id)

        # 生成置信度分数列表（数组）
        confidence_scores_array = []
        for room_id in ROOM_IDS:
            confidence = 0.0
            if room_id in expert_adjusted_room_ids:
                # 如果专家调整了这个房间，则赋予高置信度 (表示该房间应该被考虑调整)
                confidence = round(random.uniform(0.9, 0.99), 2)
            else:
                # 如果专家没有调整这个房间，则赋予低置信度 (表示该房间应保持原有控制指令)
                confidence = round(random.uniform(0.01, 0.1), 2)
            confidence_scores_array.append(confidence)

        # 生成解释和思考过程
        explanation = ""

        # 构建最终的 output JSON 对象
        output_object = {
            "confidence_scores": confidence_scores_array,
            "explanation_and_thought_process": explanation
        }
        output_str = json.dumps(output_object, indent=2, ensure_ascii=False)
        # --- output_str 生成逻辑替换结束 ---

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
    print("Not enough common timestamps to generate SFT data (need at least two for comparison).")

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
