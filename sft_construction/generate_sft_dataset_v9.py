import pandas as pd
import json
import os
from datetime import datetime
import random  # 导入 random 模块用于生成随机置信度
import numpy as np


with open("../llm_baseline_prompt/zero_shot_prompt_action_candidates.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

OPTIMAL_TEMP_MIN = 25
OPTIMAL_TEMP_MAX = 27

# !!! 修改数据目录为1分钟间隔的数据集 !!!
DATA_DIR = "../expert_data/step8_expert_data_remap_without_downsampling"
OUTPUT_DIR = "sft_data_v9"  # 更新输出目录名称以区分

ROOM_IDS = range(1, 8)
DAYS = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]  # 8月份的特定天

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and Aggregate Data ---
room_dfs = {}

print("1. Loading and preprocessing data for each room...")

for room_id in ROOM_IDS:
    all_daily_dfs = []
    for day in DAYS:
        file_path = os.path.join(DATA_DIR, f"room_{room_id}_2021_8_{day}_0900_1900.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # 1.1 时间处理
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M')
            df = df.sort_values(by='datetime').reset_index(drop=True)

            # 1.2 计算平均室温
            df['mean_room_temp'] = (df['room_temp1'] + df['room_temp2']) / 2

            # 1.3 补充 interpret_obs 所需的供回水温度 (如果CSV没有，补0以满足shape要求)
            if 'supply_temp' not in df.columns:
                df['supply_temp'] = -1.0
            if 'return_temp' not in df.columns:
                df['return_temp'] = -1.0


            # =========================================================
            # 关键步骤：生成 Action (Label) -> T+1 时刻的状态
            # shift(-1) 会把下一行的数据移到当前行，最后一行变为 NaN
            # =========================================================
            df['action_next_fcu'] = df['FCU_combined_state'].shift(-1)

            # =========================================================
            # 关键步骤：生成 History (Observation) -> T-1 到 T-4
            # =========================================================
            cols_to_shift = {
                'FCU_combined_state': 'FCU_state',
                'mean_room_temp': 'room_temp',
                'occupant_num': 'occupant_num',
                'outdoor_temp': 'outdoor_temp',
                'supply_temp': 'supply_temp',
                'return_temp': 'return_temp'
            }

            # 生成 T-1, T-2, T-3, T-4
            for lag in range(1, 5):
                for col_orig, col_prefix in cols_to_shift.items():
                    df[f'{col_prefix}_T_minus_{lag}min'] = df[col_orig].shift(lag)

            # 1.4 删除无效数据
            # shift(-1) 导致最后一行 NaN (无法预测未来)
            # shift(4) 导致前四行 NaN (没有足够的历史)
            # 使用 dropna 删除包含任何 NaN 的行，确保数据完整
            df.dropna(inplace=True)

            all_daily_dfs.append(df)
        else:
            pass  # 文件不存在则跳过

    if all_daily_dfs:
        room_dfs[room_id] = pd.concat(all_daily_dfs, ignore_index=True)
    else:
        print(f"Warning: No data for Room {room_id}")

# 2. 寻找所有房间共有的时间戳
print("2. Finding common timestamps...")
if not room_dfs:
    print("No data loaded.")
    exit()

first_rid = list(room_dfs.keys())[0]
common_timestamps_set = set(room_dfs[first_rid]['datetime'])

for room_id in ROOM_IDS:
    if room_id in room_dfs:
        common_timestamps_set = common_timestamps_set.intersection(set(room_dfs[room_id]['datetime']))

common_timestamps = sorted(list(common_timestamps_set))
print(f"Found {len(common_timestamps)} common valid timestamps.")

# 3. 构建 Numpy Arrays
print("3. Constructing Observation and Action arrays...")

all_obs_list = []
all_act_list = []

# 为了加速，我们可以先将需要的列提取出来，避免在循环中频繁访问 DataFrame
# 但为了代码逻辑清晰，这里保持基于 timestamp 的索引（数据量不大时性能可接受）

# 定义 interpret_obs 需要的顺序：[T-4, T-3, T-2, T-1, T]
time_lags = [4, 3, 2, 1, 0]

for ts in common_timestamps:
    current_obs_frame = []
    current_actions = []  # 存储当前时间步下，7个房间的 T+1 动作

    # --- 获取该时间戳下各房间的数据行 ---
    room_rows = {}
    valid_ts = True
    for rid in ROOM_IDS:
        # 假设 timestamp 唯一，取第一行
        row = room_dfs[rid].loc[room_dfs[rid]['datetime'] == ts]
        if row.empty:
            valid_ts = False
            break
        room_rows[rid] = row

    if not valid_ts:
        continue

    # --- A. 构建 Action (Shape: 7,) ---
    # 定义：Action 为所有 7 个房间在 T+1 时刻的 FCU 状态
    for rid in ROOM_IDS:
        act = room_rows[rid]['action_next_fcu'].iloc[0]
        current_actions.append(int(act))

    # --- B. 构建 Observation (Shape: 180,) ---
    # 结构：5个时间步 * (1个室外 + 7个房间 * 5个特征) = 5 * 36 = 180

    for lag in time_lags:
        suffix = f"_T_minus_{lag}min" if lag > 0 else ""

        # 1. Outdoor Temp (1 value) - 任意取 Room1 的数据
        if lag == 0:
            outdoor = room_rows[1]['outdoor_temp'].iloc[0]
        else:
            outdoor = room_rows[1][f'outdoor_temp{suffix}'].iloc[0]
        current_obs_frame.append(float(outdoor))

        # 2. Room Data (7 rooms * 5 values)
        for rid in ROOM_IDS:
            row = room_rows[rid]
            # 确定列名
            if lag == 0:
                vals = [
                    row['mean_room_temp'].iloc[0],
                    row['FCU_combined_state'].iloc[0],
                    row['supply_temp'].iloc[0],
                    row['return_temp'].iloc[0],
                    row['occupant_num'].iloc[0]
                ]
            else:
                vals = [
                    row[f'room_temp{suffix}'].iloc[0],
                    row[f'FCU_state{suffix}'].iloc[0],
                    row[f'supply_temp{suffix}'].iloc[0],
                    row[f'return_temp{suffix}'].iloc[0],
                    row[f'occupant_num{suffix}'].iloc[0]
                ]
            current_obs_frame.extend([float(v) for v in vals])

    # 收集样本
    all_obs_list.append(np.array(current_obs_frame, dtype=np.float32))
    all_act_list.append(np.array(current_actions, dtype=np.int64))  # Action通常是离散整数

# 4. 转换为最终的 Numpy Array
observations = np.array(all_obs_list)
actions = np.array(all_act_list)

print("=" * 40)
print("Data Generation Complete")
print(f"Observations Shape: {observations.shape}")  # 期望 (N, 180)
print(f"Actions Shape     : {actions.shape}")  # 期望 (N, 7)
print("=" * 40)

# 简单的验证
if len(observations) > 0:
    # 验证 Obs 形状
    assert observations.shape[1] == 180, "Error: Obs length is not 180"
    # 验证对齐 (Action行数等于Obs行数)
    assert observations.shape[0] == actions.shape[0], "Error: Length mismatch"

    print("Verification Passed: Shapes are correct.")

    # 可选：保存数据
    np.savez(os.path.join(OUTPUT_DIR, "expert_trajectories.npz"), obs=observations, actions=actions)
    print("Saved to expert_trajectories.npz")