import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os # 导入os模块用于文件路径操作

# --- 添加以下代码来支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 尝试多种中文字体，按顺序优先使用
plt.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题
# --- 中文显示配置结束 ---

def plot_room_data(room_name: str, target_date_str: str, start_time_str: str, end_time_str: str):
    """
    绘制指定房间在特定日期时间范围内的FCU、人员、温度和湿度数据曲线。
    同时，从 pump_1_result.csv 读取室外温度并与室内温度一同绘制。

    Args:
        room_name (str): 房间名称，例如 'room_1'。会根据此名称构建CSV文件路径。
        target_date_str (str): 目标日期字符串，格式例如 'YYYY/MM/DD'，例如 '2021/8/7'。
        start_time_str (str): 开始时间字符串，格式例如 'HH:MM'，例如 '09:00'。
        end_time_str (str): 结束时间字符串，格式例如 'HH:MM'，例如 '21:00'。
    """

    # 构造 CSV 文件路径
    room_csv_file_path = f'expert_data_new/{room_name}_result.csv'

    # --- 1. 读取房间 CSV 文件 (room_X_result.csv) ---
    try:
        df = pd.read_csv(room_csv_file_path)
        print(f"\nCSV 文件 '{room_csv_file_path}' 读取成功。")
    except FileNotFoundError:
        print(f"错误：'{room_csv_file_path}' 文件未找到。请检查文件路径或确保文件存在。")
        return # 文件未找到时退出函数
    except Exception as e:
        print(f"读取房间 CSV 文件时发生错误: {e}")
        return

    # --- 2. 读取泵 CSV 文件 (pump_1_result.csv，用于室外温度) ---
    pump_csv_file_path = 'expert_data/pump_1_result.csv'
    df_pump = pd.DataFrame() # 初始化为空 DataFrame，以防读取失败
    try:
        df_pump = pd.read_csv(pump_csv_file_path)
        print(f"CSV 文件 '{pump_csv_file_path}' 读取成功。")
    except FileNotFoundError:
        print(f"警告：'{pump_csv_file_path}' 文件未找到。无法加载室外温度数据。")
    except Exception as e:
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
        return # 如果日期时间转换失败，后续绘图将无法进行

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

    # 筛选泵数据并合并室外温度到 df_filtered
    if not df_pump.empty:
        # 筛选泵数据到相同的日期时间范围，并只选择 'outdoor_temp' 列
        df_pump_filtered = df_pump.loc[start_dt:end_dt][['outdoor_temp']].copy()
        if not df_pump_filtered.empty:
            # 使用 join 将 outdoor_temp 列合并到 df_filtered
            # 默认是左连接 (how='left')，会保留df_filtered的所有索引，并根据索引匹配加入outdoor_temp
            df_filtered = df_filtered.join(df_pump_filtered, how='left')
            print("室外温度数据已与房间数据合并。")
        else:
            print("警告: 泵文件中在指定日期和时间范围内未找到室外温度数据。")
    else:
        print("警告: 未加载泵文件，或泵文件为空，因此无法获取室外温度数据。")


    # --- 5. 绘图 ---
    # 创建一个图表和三个子图，共享 X 轴 (时间轴)
    fig, axes = plt.subplots(3, 1, figsize=(18, 24), sharex=True)

    # 绘图 1: FCU_onoff_feedback 和 FCU_fan_feedback
    ax1 = axes[0]
    # 计算新的综合状态：如果FCU关闭(FCU_onoff_feedback=0)，则状态为0；否则为风扇档位
    # apply函数在没有数据时可能导致FCU_combined_state为空或全NaN
    df_filtered['FCU_combined_state'] = df_filtered.apply(
        lambda row: row['FCU_fan_feedback'] if row['FCU_onoff_feedback'] == 1 else 0, axis=1
    )
    df_filtered['FCU_combined_state'].plot(ax=ax1, label='FCU 启停与风扇档位',
                                           marker='.', linestyle='-', drawstyle='steps-post', alpha=0.8)
    ax1.set_title(f'FCU 启停与风扇档位 ({room_name} - {target_date_str}, {start_time_str}-{end_time_str})')
    ax1.set_ylabel('档位 (0=关)')
    ax1.legend()
    ax1.grid(True)
    # 设定Y轴刻度，FCU_combined_state 预期为 0, 1, 2, 3
    valid_fcu_states = df_filtered['FCU_combined_state'].dropna().unique()
    if valid_fcu_states.size == 0:
        fcu_min_tick = 0
        fcu_max_tick = 3 # 默认最大档位
    else:
        fcu_min_tick = int(valid_fcu_states.min())
        fcu_max_tick = int(valid_fcu_states.max())

    # 确保Y轴刻度从0开始，并至少覆盖到最高档位3
    ax1.set_yticks(np.arange(0, max(fcu_max_tick + 1, 4)))
    ax1.set_ylim(bottom=0) # 确保Y轴底部从0开始


    # 绘图 2: occupant_num
    ax2 = axes[1]
    df_filtered['occupant_num'].plot(ax=ax2, label='Occupant Number (人数)', marker='.', linestyle='-', alpha=0.8)
    ax2.set_title(f'人员数据 ({room_name} - {target_date_str}, {start_time_str}-{end_time_str})')
    ax2.set_ylabel('数量')
    ax2.legend()
    ax2.grid(True)
    # 调整人员数量的Y轴刻度，处理NaN情况
    max_occ_val = df_filtered['occupant_num'].max()
    if pd.isna(max_occ_val): # 检查是否为NaN
        max_occ = 0 # 如果所有值都是NaN，将最大值视为0
    else:
        max_occ = int(max_occ_val) # 转换为整数
    # 确保Y轴刻度至少从0到1或2，即使最大人数是0，也显示至少两个刻度
    y_ticks_upper_bound = max(1, max_occ + 1) # 确保Y轴上限至少为1，并为最大值留出一点空间
    ax2.set_yticks(np.arange(0, y_ticks_upper_bound + 1)) # +1 确保包含上限值

    # 绘图 3: room_temp1, room_RH1, room_temp2, room_RH2, outdoor_temp
    ax3 = axes[2]
    # 在主Y轴上绘制温度曲线
    df_filtered['room_temp1'].plot(ax=ax3, label='房间温度 1 (°C)', color='red', linestyle='-', alpha=0.8)
    df_filtered['room_temp2'].plot(ax=ax3, label='房间温度 2 (°C)', color='darkred', linestyle='--', alpha=0.8)

    # 绘制室外温度 (如果数据存在)
    if 'outdoor_temp' in df_filtered.columns and not df_filtered['outdoor_temp'].dropna().empty:
        df_filtered['outdoor_temp'].plot(ax=ax3, label='室外温度 (°C)', color='green', linestyle=':', alpha=0.9)
        ax3.set_ylabel('温度 (°C)', color='red') # Y轴标签保持不变，因为室外温度与室内温度同类
        ax3.tick_params(axis='y', labelcolor='red')
    else:
        print("注意: 未能绘制室外温度数据，可能数据不存在或全为 NaN。")
        ax3.set_ylabel('温度 (°C)', color='red') # 如果没有室外温度，Y轴标签也保持
        ax3.tick_params(axis='y', labelcolor='red')


    # 创建一个次级Y轴用于绘制湿度曲线
    ax3_twin = ax3.twinx()
    df_filtered['room_RH1'].plot(ax=ax3_twin, label='房间湿度 1 (%)', color='blue', linestyle='-', alpha=0.8)
    df_filtered['room_RH2'].plot(ax=ax3_twin, label='房间湿度 2 (%)', color='darkblue', linestyle='--', alpha=0.8)
    ax3_twin.set_ylabel('相对湿度 (%)', color='blue')
    ax3_twin.tick_params(axis='y', labelcolor='blue')

    # 合并所有Y轴的图例
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='center right')

    ax3.set_title(f'房间温度、湿度与室外温度 ({room_name} - {target_date_str}, {start_time_str}-{end_time_str})')
    ax3.set_xlabel('时间')
    ax3.grid(True)

    # 自动格式化X轴的日期标签，防止重叠
    fig.autofmt_xdate()

    # 添加整体标题
    plt.suptitle(f'{room_name} 环境数据分析 - {target_date_str}', y=0.99, fontsize=16)

    # --- 保存图片到指定文件夹 ---
    output_folder = 'expert_data_curve'
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\n创建了输出文件夹: '{output_folder}'")

    # 构建文件名，包含房间名、日期和时间范围，替换特殊字符
    file_name = (f'{room_name}_{target_date_str.replace("/", "_")}_'
                 f'{start_time_str.replace(":", "")}_{end_time_str.replace(":", "")}_curve.png')
    save_path = os.path.join(output_folder, file_name)

    # 调整布局以避免标题和图例重叠
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # rect参数用于调整子图区域，避免与suptitle重叠

    # 保存图表
    plt.savefig(save_path)
    print(f"\n图表已保存到: {save_path}")

    # # 显示图表
    # plt.show()

# --- 示例调用 ---
if __name__ == "__main__":
    for date in range(7, 23):
        for room_id in range(1, 8):
            # 定义输入参数

            room_name = f'room_{room_id}'
            target_date_str = f'2021/8/{date}'  # 目标日期
            start_time_str = '09:00'  # 开始时间
            end_time_str = '21:00'  # 结束时间

            # 调用函数
            plot_room_data(room_name, target_date_str, start_time_str, end_time_str)
