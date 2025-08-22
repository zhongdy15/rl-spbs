import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_and_save_all_rooms_data_for_day(
        day: int,
        data_folder: str = "expert_data_augment",
        save_folder: str = "expert_data_replot_from_augment",
        year: int = 2021,
        month: int = 8
):
    """
    在一张图上绘制指定日期下所有房间的数据，并将图表保存到指定文件夹。
    每个房间包含两个子图：
    1. 时间 vs. outdoor_temp, room_temp1, room_temp2
    2. 时间 vs. occupant_num, FCU_combined_state

    Args:
        day (int): 要绘制数据的日期（例如：7 代表 8月7日）。
        data_folder (str): 存放CSV文件的文件夹路径。默认为 "expert_data_augment"。
        save_folder (str): 保存生成图片的文件夹路径。
        year (int): 数据年份。默认为 2021。
        month (int): 数据月份。默认为 8。
    """

    # 设置图表布局：7行（每个房间一行），2列（每个房间两个子图）
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(18, 4 * 7), sharex=True)
    fig.suptitle(f"Room Data for August {day}, {year}", fontsize=18, y=1.005)  # 调整主标题位置

    processed_any_room = False  # 标记是否成功处理了至少一个房间的数据

    # 遍历所有房间 (room_1 到 room_7)
    for i, room_num in enumerate(range(1, 8)):
        room_str = f"room_{room_num}"
        # 构造文件名
        filename = f"{room_str}_{year}_{month}_{day}_0900_2100.csv"
        filepath = os.path.join(data_folder, filename)

        if not os.path.exists(filepath):
            print(f"⚠️ 警告: 文件未找到 - {filepath}。跳过 {room_str} 在 {day} 日的数据绘制。")
            # 如果文件不存在，隐藏对应的子图，防止出现空白框
            axes[i, 0].set_visible(False)
            axes[i, 1].set_visible(False)
            continue

        try:
            # 读取CSV文件
            df = pd.read_csv(filepath)

            # 确保 DataFrame 不为空
            if df.empty:
                print(f"⚠️ 警告: 文件 {filepath} 为空。跳过 {room_str} 在 {day} 日的数据绘制。")
                axes[i, 0].set_visible(False)
                axes[i, 1].set_visible(False)
                continue

            # 将 'date' 和 'time' 列合并为 datetime 对象，方便绘图
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

            # --- 第一个子图：温度曲线 ---
            ax1 = axes[i, 0]
            ax1.plot(df['datetime'], df['outdoor_temp'], label='Outdoor Temp', color='red', alpha=0.8)
            ax1.plot(df['datetime'], df['room_temp1'], label='Room Temp 1', color='blue', alpha=0.8)
            ax1.plot(df['datetime'], df['room_temp2'], label='Room Temp 2', color='green', alpha=0.8)
            ax1.set_title(f'{room_str} - Temperatures')
            ax1.set_ylabel('Temperature (°C)')
            ax1.legend(loc='upper right', fontsize='small')
            ax1.grid(True, linestyle='--', alpha=0.6)

            # --- 第二个子图：人员数量和FCU档位 ---
            ax2 = axes[i, 1]
            ax2.plot(df['datetime'], df['occupant_num'], label='Occupant Num', color='purple', linewidth=1.5)
            ax2.plot(df['datetime'], df['FCU_combined_state'], label='FCU State', color='orange', linestyle='--',
                     linewidth=1.5)
            ax2.set_title(f'{room_str} - Occupancy & FCU State')
            ax2.set_ylabel('Count / State')
            ax2.legend(loc='upper right', fontsize='small')
            ax2.grid(True, linestyle='--', alpha=0.6)

            # 设置FCU状态的Y轴范围，确保0和1清晰可见，并稍微留白
            max_occupant = df['occupant_num'].max() if not df['occupant_num'].empty else 0
            ax2.set_ylim([-0.1, max(max_occupant * 1.1, 1.1)])  # 确保至少能显示到1.1，兼顾occupant_num

            # 仅在最底部的子图设置X轴标签
            if i == 6:  # room_7 是最后一个房间
                ax1.set_xlabel('Time')
                ax2.set_xlabel('Time')

            processed_any_room = True  # 至少一个房间数据被成功处理

        except Exception as e:
            print(f"❌ 错误: 读取或绘制 {filepath} 时发生异常 - {e}。跳过此文件。")
            axes[i, 0].set_visible(False)
            axes[i, 1].set_visible(False)
            continue

    if not processed_any_room:
        print(f"❌ 警告: 在日期 {day} 没有成功处理任何房间的数据，不生成图片。")
        plt.close(fig)  # 关闭空图
        return

    # 自动格式化X轴的日期标签，防止重叠
    fig.autofmt_xdate()
    # 调整整体布局，使子图和标题紧凑
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # rect=[left, bottom, right, top] 调整主标题下方空间

    # 构造保存路径和文件名
    save_filename = f"room_data_{year}_{month}_{day}.png"
    full_save_path = os.path.join(save_folder, save_filename)

    # 保存图表
    plt.savefig(full_save_path, dpi=300)  # dpi=300 可以获得更高分辨率的图片
    print(f"✅ 图表已保存: {full_save_path}")
    plt.close(fig)  # 关闭图表，释放内存


# --- 主执行部分 ---
if __name__ == "__main__":
    # 定义要绘制的所有日期
    target_dates = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]

    # 定义数据源文件夹和图片保存文件夹
    data_source_folder = "expert_data_remap" #"expert_data_augment"
    output_images_folder = "expert_data_replot_from_remap"

    # 创建图片保存文件夹（如果不存在）
    os.makedirs(output_images_folder, exist_ok=True)
    print(f"\n✅ 确保图片保存目录 '{output_images_folder}' 存在。")

    print("\n--- 开始生成并保存所有日期的图表 ---")
    for day_to_plot in target_dates:
        print(f"\n>>>> 正在处理日期: 8月{day_to_plot}日 <<<<")
        plot_and_save_all_rooms_data_for_day(day_to_plot, data_folder=data_source_folder,
                                             save_folder=output_images_folder)

    print("\n--- 所有图表生成完成 ---")
    print(f"所有生成的图片已保存至 '{output_images_folder}' 文件夹。")
