import pandas as pd
import os  # 导入os模块用于文件路径操作
import numpy as np  # 用于处理NaN值


def export_room_data_to_csv(room_name: str, target_date_str: str, start_time_str: str, end_time_str: str):
    """
    读取指定房间在特定日期时间范围内的数据，并将其与室外温度合并后，
    将指定列的数据整理成CSV格式并保存到文件。

    CSV文件将包含 'date', 'time', 'outdoor_temp', 'room_temp1', 'room_temp2',
    'room_RH1', 'room_RH2', 'occupant_num', 'FCU_fan_feedback', 'FCU_onoff_feedback',
    'FCU_combined_state' 等列。

    Args:
        room_name (str): 房间名称，例如 'room_1'。会根据此名称构建CSV文件路径。
        target_date_str (str): 目标日期字符串，格式例如 'YYYY/MM/DD'，例如 '2021/8/7'。
        start_time_str (str): 开始时间字符串，格式例如 'HH:MM'，例如 '09:00'。
        end_time_str (str): 结束时间字符串，格式例如 'HH:MM'，例如 '21:00'。
    """

    # 构造 CSV 文件路径
    room_csv_file_path = f'expert_data_new/{room_name}_result.csv'
    pump_csv_file_path = 'expert_data/pump_1_result.csv'  # 室外温度数据来源

    # --- 1. 读取房间 CSV 文件 (room_X_result.csv) ---
    try:
        df = pd.read_csv(room_csv_file_path)
        print(f"\nCSV 文件 '{room_csv_file_path}' 读取成功。")
    except FileNotFoundError:
        print(f"错误：'{room_csv_file_path}' 文件未找到。请检查文件路径或确保文件存在。")
        return  # 文件未找到时退出函数
    except Exception as e:
        print(f"读取房间 CSV 文件时发生错误: {e}")
        return

    # --- 2. 读取泵 CSV 文件 (pump_1_result.csv，用于室外温度) ---
    df_pump = pd.DataFrame()  # 初始化为空 DataFrame，以防读取失败
    try:
        df_pump = pd.read_csv(pump_csv_file_path)
        print(f"CSV 文件 '{pump_csv_file_path}' 读取成功。")
    except FileNotFoundError:
        print(f"警告：'{pump_csv_file_path}' 文件未找到。无法加载室外温度数据。")
    except Exception as e:  # 捕获其他可能的读取错误
        print(f"读取 pump CSV 文件时发生错误: {e}")

    # --- 3. 数据预处理: 将 'date' 和 'time' 列合并为一个 'datetime' 索引 ---
    try:
        # 处理房间数据
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        df = df.set_index('datetime').sort_index()
        print("\n房间数据的 'datetime' 列已创建并设为索引。")

        # 处理泵数据 (如果已成功加载)
        if not df_pump.empty:
            df_pump['datetime'] = pd.to_datetime(df_pump['date'] + ' ' + df_pump['time'], errors='coerce')
            df_pump.dropna(subset=['datetime'], inplace=True)
            df_pump = df_pump.set_index('datetime').sort_index()
            print("泵数据的 'datetime' 列已创建并设为索引。")

    except Exception as e:
        print(f"日期时间转换时发生错误：{e}")
        print("请检查 'date' 和 'time' 列的数据格式是否正确。")
        return  # 如果日期时间转换失败，后续操作将无法进行

    # --- 4. 筛选指定日期和时间范围的数据 ---
    # 创建完整的 datetime 对象用于筛选
    try:
        start_dt = pd.to_datetime(f'{target_date_str} {start_time_str}')
        end_dt = pd.to_datetime(f'{target_date_str} {end_time_str}')
    except ValueError as e:
        print(f"解析目标日期/时间时发生错误：{e}")
        print("请确保 'target_date_str', 'start_time_str', 'end_time_str' 格式正确。")
        return

    # 筛选房间数据
    df_filtered = df.loc[start_dt:end_dt].copy()

    if df_filtered.empty:
        print(f"\n未找到 {room_name} 在 {target_date_str} 从 {start_time_str} 到 {end_time_str} 之间的数据。")
        print("请检查日期、时间范围或 CSV 文件中的数据。")
        return
    else:
        print(f"\n{room_name} 数据已筛选，日期为 {target_date_str}，时间范围从 {start_time_str} 到 {end_time_str}。")
        print(f"筛选后的数据形状: {df_filtered.shape}")

    # --- 5. 添加 FCU 综合状态列 ---
    # 确保在计算FCU_combined_state之前，FCU_onoff_feedback和FCU_fan_feedback列存在
    if 'FCU_onoff_feedback' in df_filtered.columns and 'FCU_fan_feedback' in df_filtered.columns:
        df_filtered['FCU_combined_state'] = df_filtered.apply(
            lambda row: row['FCU_fan_feedback'] if row['FCU_onoff_feedback'] == 1 else 0, axis=1
        )
    else:
        print("警告: 缺少 'FCU_onoff_feedback' 或 'FCU_fan_feedback' 列，无法计算 FCU 综合状态。")
        df_filtered['FCU_combined_state'] = np.nan  # 设置为NaN以保持列存在

    # --- 6. 筛选泵数据并合并室外温度到 df_filtered ---
    if not df_pump.empty:
        # 筛选泵数据到相同的日期时间范围，并只选择 'outdoor_temp' 列
        df_pump_filtered = df_pump.loc[start_dt:end_dt][['outdoor_temp']].copy()
        if not df_pump_filtered.empty:
            # 使用 join 将 outdoor_temp 列合并到 df_filtered
            df_filtered = df_filtered.join(df_pump_filtered, how='left')
            print("室外温度数据已与房间数据合并。")
        else:
            print("警告: 泵文件中在指定日期和时间范围内未找到室外温度数据。")
    else:
        print("警告: 未加载泵文件，或泵文件为空，因此无法获取室外温度数据。")

    # --- 7. 准备用于 CSV 导出的 DataFrame ---
    # 首先，重置索引，将 'datetime' 变成一个普通列
    df_csv_output = df_filtered.reset_index()

    # 从 'datetime' 列派生 'date' 和 'time' 字符串列
    df_csv_output['date'] = df_csv_output['datetime'].dt.strftime('%Y/%m/%d')  # 保持原始日期格式
    df_csv_output['time'] = df_csv_output['datetime'].dt.strftime('%H:%M')

    # 定义希望在CSV中包含的列及其顺序
    desired_cols_order = [
        'date', 'time', 'outdoor_temp', 'room_temp1', 'room_temp2',
        'room_RH1', 'room_RH2', 'occupant_num',
        'FCU_fan_feedback', 'FCU_onoff_feedback', 'FCU_combined_state'
    ]

    # 过滤出 DataFrame 中实际存在的列
    final_cols_to_export = [col for col in desired_cols_order if col in df_csv_output.columns]

    df_csv_output = df_csv_output[final_cols_to_export].copy()

    if df_csv_output.empty:
        print("筛选并选择列后，DataFrame为空。不生成CSV文件。")
        return

    # --- 8. 准备保存 CSV 文件 ---
    output_folder = 'expert_data_clean'  # 新的输出文件夹
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\n创建了输出文件夹: '{output_folder}'")

    # 构建文件名，包含房间名、日期和时间范围，替换特殊字符
    # 例如: room_1_2021_8_7_0900_2100.csv
    csv_file_name = (f'{room_name}_{target_date_str.replace("/", "_")}_'
                     f'{start_time_str.replace(":", "")}_{end_time_str.replace(":", "")}.csv')
    csv_save_path = os.path.join(output_folder, csv_file_name)

    # --- 9. 将 DataFrame 转换为 CSV 并保存 ---
    try:
        # index=False 表示不将 DataFrame 的索引写入 CSV 文件
        # encoding='utf-8' 确保中文字符等能正确保存
        df_csv_output.to_csv(csv_save_path, index=False, encoding='utf-8')
        print(f"\n数据已成功保存到 CSV 文件: {csv_save_path}")
    except Exception as e:
        print(f"保存 CSV 文件时发生错误: {e}")


# --- 示例调用 ---
# if __name__ == "__main__":
#     # 示例1: 正常调用
#     room_id = 1
#     date = 7  # 这里的date是日期的数字部分
#
#     room_name_ex1 = f'room_{room_id}'
#     target_date_str_ex1 = f'2021/8/{date}'  # 目标日期
#     start_time_str_ex1 = '09:00'  # 开始时间
#     end_time_str_ex1 = '21:00'  # 结束时间
#
#     print("--- 示例1: 生成 room_1 在 2021/8/7 的CSV文件 ---")
#     export_room_data_to_csv(room_name_ex1, target_date_str_ex1, start_time_str_ex1, end_time_str_ex1)
#
#     print("\n--- 示例2: 尝试生成另一个房间/日期的CSV文件 ---")
#     # 假设 room_1_result.csv 包含 2021/8/8 的数据
#     export_room_data_to_csv('room_1', '2021/8/8', '10:00', '18:00')
#
#     print("\n--- 示例3: 尝试一个可能没有数据的日期范围 (例如，文件存在但日期无数据) ---")
#     export_room_data_to_csv('room_1', '2022/1/1', '00:00', '23:59')  # 假设这个日期没有数据
# --- 主执行部分 ---
if __name__ == "__main__":
    # 定义房间和可用日期的映射表
    room_availability_data = {
        "room_1": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
        "room_2": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
        "room_3": [], # 无可用日期 (没有人员数据)
        "room_4": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
        "room_5": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
        "room_6": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
        "room_7": [16, 17, 18, 20, 21]
    }

    fixed_year = 2021
    fixed_month = 8 # 八月

    # 固定的时间范围
    start_time_fixed = '09:00'
    end_time_fixed = '21:00'

    print("--- 开始批量导出房间数据到 CSV 文件 ---")

    for room_name, available_dates in room_availability_data.items():
        if not available_dates:
            print(f"\n跳过 {room_name}: 表格中指示无可用日期数据。")
            continue

        print(f"\n--- 正在处理 {room_name} 的数据 ---")
        for date_day in available_dates:
            target_date_str = f'{fixed_year}/{fixed_month}/{date_day}'
            print(f"正在尝试导出 {room_name} - {target_date_str} ({start_time_fixed}-{end_time_fixed}) 的数据...")
            # 调用之前定义的函数
            export_room_data_to_csv(room_name, target_date_str, start_time_fixed, end_time_fixed)
            print(f"已完成 {room_name} - {target_date_str} 的导出尝试。")

    print("\n--- 所有房间数据导出任务完成 ---")