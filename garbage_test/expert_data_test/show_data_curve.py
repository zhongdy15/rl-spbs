import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np  # 用于生成演示数据，实际使用时可删除
# --- 添加以下代码来支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 尝试多种中文字体，按顺序优先使用
plt.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题
# --- 中文显示配置结束 ---

# --- 1. 读取 CSV 文件 ---
# 请确保 'room_1_result.csv' 文件与您的 Python 脚本在同一目录下，
# 或者提供完整的文件路径。
try:
    df = pd.read_csv('expert_data_new/room_1_result.csv')
    print("\nCSV 文件 'room_1_result.csv' 读取成功。")
    print("数据前5行:")
    print(df.head())
    print("\nDataFrame 信息:")
    print(df.info())
except FileNotFoundError:
    print("错误：'room_1_result.csv' 文件未找到。请检查文件路径或确保文件存在。")
    exit()  # 文件未找到时退出程序

# --- 2. 数据预处理: 将 'date' 和 'time' 列合并为一个 'datetime' 索引 ---
try:
    # 尝试将 'date' 和 'time' 合并为 datetime 对象
    # errors='coerce' 会将无法解析的值设为 NaT (Not a Time)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    # 删除 datetime 转换失败的行
    df.dropna(subset=['datetime'], inplace=True)
    # 将新的 datetime 列设置为 DataFrame 的索引
    df = df.set_index('datetime')
    # 确保索引按时间顺序排序，这对于时间序列分析很重要
    df = df.sort_index()
    print("\n'datetime' 列已创建并设为索引。")
    print(df.head())
except Exception as e:
    print(f"日期时间转换时发生错误：{e}")
    print("请检查 'date' 和 'time' 列的数据格式是否正确。")
    exit()  # 如果日期时间转换失败，后续绘图将无法进行

# --- 3. 筛选指定日期和时间范围的数据 ---
target_date_str = '2021/8/7'  # 目标日期
start_time_str = '09:00'  # 开始时间
end_time_str = '21:00'  # 结束时间

# 创建完整的 datetime 对象用于筛选
try:
    start_dt = pd.to_datetime(f'{target_date_str} {start_time_str}')
    end_dt = pd.to_datetime(f'{target_date_str} {end_time_str}')
except ValueError as e:
    print(f"解析目标日期/时间时发生错误：{e}")
    print("请确保 'target_date_str', 'start_time_str', 'end_time_str' 格式正确。")
    exit()

# 使用 .loc 进行基于时间的索引筛选
df_filtered = df.loc[start_dt:end_dt].copy()  # .copy() 避免 SettingWithCopyWarning

if df_filtered.empty:
    print(f"\n未找到 {target_date_str} 在 {start_time_str} 到 {end_time_str} 之间的数据。")
    print("请检查日期、时间范围或 CSV 文件中的数据。")
else:
    print(f"\n数据已筛选，日期为 {target_date_str}，时间范围从 {start_time_str} 到 {end_time_str}。")
    print(f"筛选后的数据形状: {df_filtered.shape}")
    print("筛选后数据的前5行:")
    print(df_filtered.head())

    # --- 4. 绘图 ---
    # 创建一个图表和三个子图，共享 X 轴 (时间轴)
    fig, axes = plt.subplots(3, 1, figsize=(18, 24), sharex=True)
    # fig.subplots_adjust(hspace=2.0)  # 单独调用 fig.subplots_adjust 来设置间距

    # 绘图 1: FCU_onoff_feedback 和 FCU_fan_feedback
    ax1 = axes[0]
    # 计算新的综合状态：如果FCU关闭(FCU_onoff_feedback=0)，则状态为0；否则为风扇档位
    # df_filtered['FCU_combined_state'] = df_filtered['FCU_fan_feedback'] * df_filtered['FCU_onoff_feedback']
    # 更安全的写法，因为FCU_fan_feedback在FCU关闭时可能也有非0值，但我们希望它显示为0
    df_filtered['FCU_combined_state'] = df_filtered.apply(
        lambda row: row['FCU_fan_feedback'] if row['FCU_onoff_feedback'] == 1 else 0, axis=1
    )

    df_filtered['FCU_combined_state'].plot(ax=ax1, label='FCU 启停与风扇档位',
                                           marker='.', linestyle='-', drawstyle='steps-post', alpha=0.8)

    ax1.set_title(f'FCU 启停与风扇档位 ({target_date_str}, {start_time_str}-{end_time_str})')
    ax1.set_ylabel('档位 (0=关)')
    ax1.legend()
    ax1.grid(True)
    # 设定Y轴刻度，假设FCU_onoff是0/1，风扇速度是1/2/3
    # 自动确定Y轴刻度范围
    min_val_fcu = df_filtered[['FCU_onoff_feedback', 'FCU_fan_feedback']].min().min() if not df_filtered.empty else 0
    max_val_fcu = df_filtered[['FCU_onoff_feedback', 'FCU_fan_feedback']].max().max() if not df_filtered.empty else 3
    ax1.set_yticks(np.arange(int(min_val_fcu), int(max_val_fcu) + 1))

    # 绘图 2: occupant_num 和 occupant_transform
    ax2 = axes[1]
    df_filtered['occupant_num'].plot(ax=ax2, label='Occupant Number (人数)', marker='.', linestyle='-', alpha=0.8)
    # df_filtered['occupant_transform'].plot(ax=ax2, label='Occupant Transform (存在状态)', marker='.', linestyle='--',
    #                                        alpha=0.8)
    ax2.set_title(f'人员数据 ({target_date_str}, {start_time_str}-{end_time_str})')
    ax2.set_ylabel('数量')
    ax2.legend()
    ax2.grid(True)
    # 调整人员数量的Y轴刻度
    max_occ = df_filtered['occupant_num'].max() if not df_filtered['occupant_num'].empty else 1
    ax2.set_yticks(np.arange(0, int(max_occ) + 2))

    # 绘图 3: room_temp1, room_RH1, room_temp2, room_RH2
    ax3 = axes[2]
    # 在主Y轴上绘制温度曲线
    df_filtered['room_temp1'].plot(ax=ax3, label='房间温度 1 (°C)', color='red', linestyle='-', alpha=0.8)
    df_filtered['room_temp2'].plot(ax=ax3, label='房间温度 2 (°C)', color='darkred', linestyle='--', alpha=0.8)
    ax3.set_ylabel('温度 (°C)', color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    # 创建一个次级Y轴用于绘制湿度曲线
    ax3_twin = ax3.twinx()
    df_filtered['room_RH1'].plot(ax=ax3_twin, label='房间湿度 1 (%)', color='blue', linestyle='-', alpha=0.8)
    df_filtered['room_RH2'].plot(ax=ax3_twin, label='房间湿度 2 (%)', color='darkblue', linestyle='--', alpha=0.8)
    ax3_twin.set_ylabel('相对湿度 (%)', color='blue')
    ax3_twin.tick_params(axis='y', labelcolor='blue')

    # 合并两个Y轴的图例
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='center right')

    ax3.set_title(f'房间温度与湿度 ({target_date_str}, {start_time_str}-{end_time_str})')
    ax3.set_xlabel('时间')
    ax3.grid(True)

    # 自动格式化X轴的日期标签，防止重叠
    fig.autofmt_xdate()

    # 添加整体标题
    plt.suptitle(f'房间环境数据分析 - {target_date_str}', y=0.99, fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

