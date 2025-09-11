import pandas as pd
import json
import os
from datetime import datetime
import random  # 导入 random 模块用于生成随机置信度

# --- Constants ---
# SYSTEM_PROMPT 和 INSTRUCTION_PROMPT 的内容将从上述更新的文件中读取。
# 假设这些文件存在于预期的路径

with open("predefined_prompt/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open("predefined_prompt/instruction_prompt.txt", "r", encoding="utf-8") as f:
    INSTRUCTION_PROMPT = f.read()

OPTIMAL_TEMP_MIN = 25
OPTIMAL_TEMP_MAX = 27

# !!! 修改数据目录为1分钟间隔的数据集 !!!
DATA_DIR = "../expert_data/expert_data_remap_without_downsampling"
OUTPUT_DIR = "all_types_sft_data_v8_1min_interval"  # 更新输出目录名称以区分

ROOM_IDS = range(1, 8)
DAYS = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]  # 8月份的特定天

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and Aggregate Data ---
room_dfs = {}

print("Loading and preprocessing data for each room...")
for room_id in ROOM_IDS:
    all_shifted_daily_dfs = []  # 用于存储每一天处理并shift后的数据
    for day in DAYS:
        # Construct file path based on provided pattern
        file_path = os.path.join(DATA_DIR, f"room_{room_id}_2021_8_{day}_0900_2100.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Convert date and time columns to a single datetime object
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'],
                                            format='%Y/%m/%d %H:%M')
            # Sort by datetime to ensure correct 'shift' for previous state within the day
            df = df.sort_values(by='datetime').reset_index(drop=True)

            # Calculate mean room temperature
            df['mean_room_temp'] = (df['room_temp1'] + df['room_temp2']) / 2

            # 核心修改：在这里对每一天的数据进行shift操作
            df['FCU_state_T_minus_1min'] = df['FCU_combined_state'].shift(1)
            df['room_temp_T_minus_1min'] = df['mean_room_temp'].shift(1)
            df['occupant_num_T_minus_1min'] = df['occupant_num'].shift(1)
            df['outdoor_temp_T_minus_1min'] = df['outdoor_temp'].shift(1)

            df['FCU_state_T_minus_2min'] = df['FCU_combined_state'].shift(2)
            df['room_temp_T_minus_2min'] = df['mean_room_temp'].shift(2)
            df['occupant_num_T_minus_2min'] = df['occupant_num'].shift(2)
            df['outdoor_temp_T_minus_2min'] = df['outdoor_temp'].shift(2)

            df['FCU_state_T_minus_3min'] = df['FCU_combined_state'].shift(3)
            df['room_temp_T_minus_3min'] = df['mean_room_temp'].shift(3)
            df['occupant_num_T_minus_3min'] = df['occupant_num'].shift(3)
            df['outdoor_temp_T_minus_3min'] = df['outdoor_temp'].shift(3)

            df['FCU_state_T_minus_4min'] = df['FCU_combined_state'].shift(4)
            df['room_temp_T_minus_4min'] = df['mean_room_temp'].shift(4)
            df['occupant_num_T_minus_4min'] = df['occupant_num'].shift(4)
            df['outdoor_temp_T_minus_4min'] = df['outdoor_temp'].shift(4)

            all_shifted_daily_dfs.append(df)
        else:
            # print(f"Warning: File not found for room {room_id}, day {day}: {file_path}") # 可以取消注释进行调试
            pass

    if all_shifted_daily_dfs:
        # 将处理好的每天数据拼接起来，形成该房间的完整数据
        room_dfs[room_id] = pd.concat(all_shifted_daily_dfs, ignore_index=True)
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

print(f"Found {len(common_timestamps)} common timestamps where all 7 rooms have data.")

# Initialize lists for SFT data categories (这些分类仍可用于数据分析或平衡，与新输出格式不冲突)
sft_data_type1 = []  # All rooms: prev FCU == 0 and curr FCU == 0
sft_data_type2 = []  # All rooms: prev FCU == curr FCU but not all are 0; OR mixed stable
sft_data_type3 = []  # At least one room: prev FCU != curr FCU

# --- Generate SFT Data Entries ---
print("Generating SFT data entries...")
# 需要至少5个时间戳 (T, T-1min, T-2min, T-3min, T-4min) 来生成完整数据
# 由于每个shift操作都会在一天开始时引入NaN，我们实际上需要确保在 common_timestamps 列表中有足够的数据点
# 且这些数据点不能是每天开始的那些因为shift而为NaN的点。
# 筛选掉那些包含NaN的行，或者在循环中检查。在 `is_valid_entry` 检查中已经包含了对NaN的判断。
if len(common_timestamps) > 4:  # 从索引4开始，确保有T-4min的数据
    for timestamp in common_timestamps:
        current_status_lines = []
        status_T_minus_1min_lines = []
        status_T_minus_2min_lines = []
        status_T_minus_3min_lines = []
        status_T_minus_4min_lines = []

        # FCU states for current (T) and T-1min are used for expert adjustment check
        current_fcu_states = {}  # FCU state at T (active from T-1min to T)
        fcu_state_T_minus_1min = {}  # FCU state at T-1min (active from T-2min to T-1min)

        is_valid_entry = True
        # Initialize outdoor temperatures for all 5 timestamps
        current_outdoor_temp = None
        outdoor_temp_T_minus_1min = None
        outdoor_temp_T_minus_2min = None
        outdoor_temp_T_minus_3min = None
        outdoor_temp_T_minus_4min = None

        # 获取当前和前4个时间步的室外温度
        # 只需要从任意一个房间的df中获取室外温度，因为它是共享的
        # 注意：现在outdoor_temp_T_minus_Xmin也是通过daily shift获得的
        outdoor_temp_data_row = room_dfs[ROOM_IDS[0]].loc[room_dfs[ROOM_IDS[0]]['datetime'] == timestamp]
        if not outdoor_temp_data_row.empty:
            current_outdoor_temp = outdoor_temp_data_row['outdoor_temp'].iloc[0]
            outdoor_temp_T_minus_1min = outdoor_temp_data_row['outdoor_temp_T_minus_1min'].iloc[0]
            outdoor_temp_T_minus_2min = outdoor_temp_data_row['outdoor_temp_T_minus_2min'].iloc[0]
            outdoor_temp_T_minus_3min = outdoor_temp_data_row['outdoor_temp_T_minus_3min'].iloc[0]
            outdoor_temp_T_minus_4min = outdoor_temp_data_row['outdoor_temp_T_minus_4min'].iloc[0]
        else:
            is_valid_entry = False

        # 检查所有室外温度值是否有效
        if any(pd.isna(temp) for temp in [current_outdoor_temp, outdoor_temp_T_minus_1min, outdoor_temp_T_minus_2min,
                                          outdoor_temp_T_minus_3min, outdoor_temp_T_minus_4min]):
            is_valid_entry = False

        if not is_valid_entry:
            continue  # 跳过此时间戳，如果室外温度数据缺失

        for room_id in ROOM_IDS:
            room_data_at_ts = room_dfs[room_id].loc[room_dfs[room_id]['datetime'] == timestamp]

            if room_data_at_ts.empty:
                is_valid_entry = False
                break

            # 获取当前时间步 (T) 的数据
            curr_temp = room_data_at_ts['mean_room_temp'].iloc[0]
            curr_occupant = room_data_at_ts['occupant_num'].iloc[0]
            curr_fan_speed = room_data_at_ts['FCU_combined_state'].iloc[0]

            # 获取 T-1min 的数据
            temp_T_minus_1min = room_data_at_ts['room_temp_T_minus_1min'].iloc[0]
            occupant_T_minus_1min = room_data_at_ts['occupant_num_T_minus_1min'].iloc[0]
            fan_speed_T_minus_1min = room_data_at_ts['FCU_state_T_minus_1min'].iloc[0]

            # 获取 T-2min 的数据
            temp_T_minus_2min = room_data_at_ts['room_temp_T_minus_2min'].iloc[0]
            occupant_T_minus_2min = room_data_at_ts['occupant_num_T_minus_2min'].iloc[0]
            fan_speed_T_minus_2min = room_data_at_ts['FCU_state_T_minus_2min'].iloc[0]

            # 获取 T-3min 的数据
            temp_T_minus_3min = room_data_at_ts['room_temp_T_minus_3min'].iloc[0]
            occupant_T_minus_3min = room_data_at_ts['occupant_num_T_minus_3min'].iloc[0]
            fan_speed_T_minus_3min = room_data_at_ts['FCU_state_T_minus_3min'].iloc[0]

            # 获取 T-4min 的数据
            temp_T_minus_4min = room_data_at_ts['room_temp_T_minus_4min'].iloc[0]
            occupant_T_minus_4min = room_data_at_ts['occupant_num_T_minus_4min'].iloc[0]
            fan_speed_T_minus_4min = room_data_at_ts['FCU_state_T_minus_4min'].iloc[0]

            # 检查所有过去时间步的数据是否有效 (包含当前时间步)
            if any(pd.isna(val) for val in [
                curr_temp, curr_occupant, curr_fan_speed,
                temp_T_minus_1min, occupant_T_minus_1min, fan_speed_T_minus_1min,
                temp_T_minus_2min, occupant_T_minus_2min, fan_speed_T_minus_2min,
                temp_T_minus_3min, occupant_T_minus_3min, fan_speed_T_minus_3min,
                temp_T_minus_4min, occupant_T_minus_4min, fan_speed_T_minus_4min
            ]):
                is_valid_entry = False
                break

            current_status_lines.append(
                f"Room {room_id} — Temperature: {curr_temp:.2f}°C, Occupant Num: {int(curr_occupant)}, Fan Speed: {int(curr_fan_speed)};"
            )
            status_T_minus_1min_lines.append(
                f"Room {room_id} — Temperature: {temp_T_minus_1min:.2f}°C, Occupant Num: {int(occupant_T_minus_1min)}, Fan Speed: {int(fan_speed_T_minus_1min)};"
            )
            status_T_minus_2min_lines.append(
                f"Room {room_id} — Temperature: {temp_T_minus_2min:.2f}°C, Occupant Num: {int(occupant_T_minus_2min)}, Fan Speed: {int(fan_speed_T_minus_2min)};"
            )
            status_T_minus_3min_lines.append(
                f"Room {room_id} — Temperature: {temp_T_minus_3min:.2f}°C, Occupant Num: {int(occupant_T_minus_3min)}, Fan Speed: {int(fan_speed_T_minus_3min)};"
            )
            status_T_minus_4min_lines.append(
                f"Room {room_id} — Temperature: {temp_T_minus_4min:.2f}°C, Occupant Num: {int(occupant_T_minus_4min)}, Fan Speed: {int(fan_speed_T_minus_4min)};"
            )

            current_fcu_states[room_id] = curr_fan_speed
            fcu_state_T_minus_1min[room_id] = fan_speed_T_minus_1min  # 用于判断专家在T时刻是否进行了调整

        if not is_valid_entry:
            continue

        # 构建完整的 input_str，包含当前和前1、2、3、4分钟的状态
        input_str = (
                "You will be provided with real-time status at the current moment (T), and also the status from 1, 2, 3, and 4 minutes ago.\n\n"
                "Current Status (at Time T, FCU Fan Speed was active from T-1min to T):\n" +
                "\n".join(current_status_lines) +
                f"\nOutdoor - Temperature: {current_outdoor_temp:.2f}°C\n\n"
                "Status 1 minute ago (at Time T-1min, FCU Fan Speed was active from T-2min to T-1min):\n" +
                "\n".join(status_T_minus_1min_lines) +
                f"\nOutdoor - Temperature: {outdoor_temp_T_minus_1min:.2f}°C\n\n"
                "Status 2 minutes ago (at Time T-2min, FCU Fan Speed was active from T-3min to T-2min):\n" +
                "\n".join(status_T_minus_2min_lines) +
                f"\nOutdoor - Temperature: {outdoor_temp_T_minus_2min:.2f}°C\n\n"
                "Status 3 minutes ago (at Time T-3min, FCU Fan Speed was active from T-4min to T-3min):\n" +
                "\n".join(status_T_minus_3min_lines) +
                f"\nOutdoor - Temperature: {outdoor_temp_T_minus_3min:.2f}°C\n\n"
                "Status 4 minutes ago (at Time T-4min, FCU Fan Speed was active from T-5min to T-4min):\n" +
                "\n".join(status_T_minus_4min_lines) +
                f"\nOutdoor - Temperature: {outdoor_temp_T_minus_4min:.2f}°C"
        )

        # --- 修改原始的 output_json 生成逻辑以适应新格式 ---
        # 确定专家在当前时间戳实际调整了哪些房间 (即 FCU 状态发生变化，比较T和T-1min)
        expert_adjusted_room_ids = []
        for room_id in ROOM_IDS:
            if current_fcu_states[room_id] != fcu_state_T_minus_1min[room_id]:
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
        explanation = ""  # 此处可以添加更复杂的解释逻辑，例如基于温度、人数等条件

        # 构建最终的 output JSON 对象
        output_object = {
            "confidence_scores": confidence_scores_array,
            "explanation_and_thought_process": explanation
        }
        output_str = json.dumps(output_object, indent=2, ensure_ascii=False)
        # --- output_str 生成逻辑替换结束 ---

        # 以下的分类逻辑 (is_type3_category, is_type1_category) 仍然有效，
        # 它们是根据输入和专家行为对样本进行分类，而不是输出格式。
        # 因此保持不变，现在基于T和T-1min进行比较。
        is_type3_category = False
        for room_id in ROOM_IDS:
            if current_fcu_states[room_id] != fcu_state_T_minus_1min[room_id]:  # 比较T和T-1min
                is_type3_category = True
                break

        is_type1_category = True
        if not is_type3_category:  # 如果没有房间被调整
            for room_id in ROOM_IDS:
                # 如果所有房间在T-1min和T时刻都保持FCU关闭 (0)
                if not (current_fcu_states[room_id] == 0 and fcu_state_T_minus_1min[room_id] == 0):
                    is_type1_category = False
                    break
        else:  # 如果有房间被调整，则不可能是Type 1
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
        "Not enough common timestamps to generate SFT data (need at least five for T and T-1min, T-2min, T-3min, T-4min comparison).")

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

