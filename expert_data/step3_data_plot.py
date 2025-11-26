import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # 导入 matplotlib.dates 模块
import numpy as np
import os

# --- 添加以下代码来支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 尝试多种中文字体，按顺序优先使用
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


# --- 中文显示配置结束 ---

def plot_processed_csv_data(csv_file_path: str):
    """
    读取指定路径的CSV文件，并绘制FCU控制档位、人员数量、温度（室内外）三张子图。
    横坐标刻度精确到30分钟。

    Args:
        csv_file_path (str): 已经处理并导出的CSV文件的完整路径，例如
                             'expert_data_csv/room_1_2021_8_7_0900_2100.csv'。
                             此CSV文件应包含以下列：
                             date, time, outdoor_temp, room_temp1, room_temp2,
                             room_RH1, room_RH2, occupant_num, FCU_fan_feedback,
                             FCU_onoff_feedback, FCU_combined_state。
    """

    # --- 1. 读取 CSV 文件 ---
    try:
        df = pd.read_csv(csv_file_path)
        print(f"\nCSV 文件 '{csv_file_path}' 读取成功。")
    except FileNotFoundError:
        print(f"错误：'{csv_file_path}' 文件未找到。请检查文件路径或确保文件存在。")
        return  # 文件未找到时退出函数
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {e}")
        return

    # 检查必要的列是否存在
    required_cols = [
        'date', 'time', 'outdoor_temp', 'room_temp1', 'room_temp2',
        'occupant_num', 'FCU_combined_state'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误：CSV 文件中缺少以下必需列：{', '.join(missing_cols)}。无法绘图。")
        return

    # --- 2. 数据预处理: 将 'date' 和 'time' 列合并为一个 'datetime' 索引 ---
    try:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        df = df.set_index('datetime').sort_index()
        print("\n'datetime' 列已创建并设为索引。")
    except Exception as e:
        print(f"日期时间转换时发生错误：{e}")
        print("请检查 'date' 和 'time' 列的数据格式是否正确。无法绘图。")
        return

    if df.empty:
        print(f"经过处理后，DataFrame为空。无法绘图。")
        return

    # --- 3. 从文件名提取信息用于标题 ---
    base_filename = os.path.basename(csv_file_path)
    # 假设文件名格式为 room_X_YYYY_M_D_HHMM_HHMM.csv
    parts = base_filename.replace('.csv', '').split('_')
    try:
        room_name = f"{parts[0]}_{parts[1]}"  # 例如 'room_1'
        target_date_str_formatted = f"{parts[2]}/{parts[3]}/{parts[4]}"  # 例如 '2021/8/7'
        start_time_str_formatted = f"{parts[5][:2]}:{parts[5][2:]}"  # 例如 '09:00'
        end_time_str_formatted = f"{parts[6][:2]}:{parts[6][2:]}"  # 例如 '21:00'
    except IndexError:
        print("警告: 无法从文件名解析房间名、日期和时间。将使用默认标题。")
        room_name = "未知房间"
        target_date_str_formatted = "未知日期"
        start_time_str_formatted = "未知时间"
        end_time_str_formatted = "未知时间"

    # --- 4. 绘图 ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 24), sharex=True)

    # 绘图 1: FCU_combined_state (固定纵坐标轴0-4)
    ax1 = axes[0]
    df['FCU_combined_state'].plot(ax=ax1, label='FCU 控制档位',
                                  marker='.', linestyle='-', drawstyle='steps-post', alpha=0.8)
    ax1.set_title(
        f'FCU 控制档位 ({room_name} - {target_date_str_formatted}, {start_time_str_formatted}-{end_time_str_formatted})')
    ax1.set_ylabel('档位 (0-4)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 4)  # 固定Y轴范围
    ax1.set_yticks(np.arange(0, 5))  # 固定Y轴刻度 0, 1, 2, 3, 4

    # 绘图 2: occupant_num
    ax2 = axes[1]
    df['occupant_num'].plot(ax=ax2, label='Occupant Number (人数)', marker='.', linestyle='-', alpha=0.8)
    ax2.set_title(
        f'人员数量 ({room_name} - {target_date_str_formatted}, {start_time_str_formatted}-{end_time_str_formatted})')
    ax2.set_ylabel('数量')
    ax2.legend()
    ax2.grid(True)
    # 调整人员数量的Y轴刻度，处理NaN情况
    max_occ_val = df['occupant_num'].max()
    if pd.isna(max_occ_val):
        max_occ = 0
    else:
        max_occ = int(max_occ_val)
    y_ticks_upper_bound = max(1, max_occ + 1)
    ax2.set_yticks(np.arange(0, y_ticks_upper_bound + 1))

    # 绘图 3: room_temp1, room_temp2, outdoor_temp
    ax3 = axes[2]
    df['room_temp1'].plot(ax=ax3, label='房间温度 1 (°C)', color='red', linestyle='-', alpha=0.8)
    df['room_temp2'].plot(ax=ax3, label='房间温度 2 (°C)', color='darkred', linestyle='--', alpha=0.8)

    # 检查 outdoor_temp 列是否存在且有数据，然后绘制
    if 'outdoor_temp' in df.columns and not df['outdoor_temp'].dropna().empty:
        df['outdoor_temp'].plot(ax=ax3, label='室外温度 (°C)', color='green', linestyle=':', alpha=0.9)
    else:
        print("注意: CSV 中没有室外温度数据或数据全为 NaN，未绘制室外温度曲线。")

    ax3.set_title(
        f'房间温度与室外温度 ({room_name} - {target_date_str_formatted}, {start_time_str_formatted}-{end_time_str_formatted})')
    ax3.set_ylabel('温度 (°C)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlabel('时间')

    # --- 修改点: 设置横坐标刻度为30分钟间隔 ---
    # 定义主要刻度定位器为每30分钟
    major_locator = mdates.MinuteLocator(interval=30)
    # 定义刻度标签格式为小时:分钟
    formatter = mdates.DateFormatter('%H:%M')

    # 将主要刻度定位器和格式器应用到X轴
    # 因为 sharex=True，所以只需要对其中一个轴（例如 ax3）设置即可
    ax3.xaxis.set_major_locator(major_locator)
    ax3.xaxis.set_major_formatter(formatter)

    # 自动格式化X轴的日期标签，防止重叠（即使设置了定位器，这个也有助于旋转标签）
    fig.autofmt_xdate()

    # 添加整体标题
    plt.suptitle(f'{room_name} 数据曲线 - {target_date_str_formatted}', y=0.99, fontsize=16)

    # --- 5. 保存图片到指定文件夹 ---
    output_folder = 'step3_expert_data_plots_from_clean_csv'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\n创建了输出文件夹: '{output_folder}'")

    plot_file_name = base_filename.replace('.csv', '_plot.png')
    save_path = os.path.join(output_folder, plot_file_name)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path)
    print(f"\n图表已保存到: {save_path}")

    # plt.show()


# --- 示例调用 (仅修改 main 函数部分) ---
if __name__ == "__main__":
    # 定义包含CSV文件的目录
    csv_directory = 'step2_expert_data_clean'

    # 检查目录是否存在
    if not os.path.exists(csv_directory):
        print(f"错误: 目录 '{csv_directory}' 不存在。请确保CSV文件已生成并存放在此目录。")
    else:
        print(f"--- 开始批量绘制 '{csv_directory}' 目录下的所有CSV文件 ---")

        # 获取目录下所有文件
        all_files = os.listdir(csv_directory)

        # 筛选出CSV文件
        csv_files = [f for f in all_files if f.endswith('.csv')]

        if not csv_files:
            print(f"在目录 '{csv_directory}' 中没有找到任何CSV文件。")
        else:
            print(f"找到了 {len(csv_files)} 个CSV文件，开始逐一处理...")
            for csv_file_name in csv_files:
                full_csv_file_path = os.path.join(csv_directory, csv_file_name)
                print(f"\n--- 尝试绘制 CSV 文件: {full_csv_file_path} ---")
                plot_processed_csv_data(full_csv_file_path)
                print(f"已完成对文件 '{csv_file_name}' 的绘图尝试。")

            print("\n--- 所有CSV文件绘图任务完成 ---")


