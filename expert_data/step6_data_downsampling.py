import pandas as pd
import os


def process_and_downsample_csv(input_filepath: str):
    """
    读取指定路径的CSV文件，执行以下操作：
    1. 筛选时间段在09:00到19:00之间的数据。
    2. 每隔5分钟取一次数据点。
    3. 将处理后的数据保存到 'expert_data_downsampling' 文件夹中，保留原始文件名。

    Args:
        input_filepath (str): 输入CSV文件的完整路径，例如：
                              "expert_data_augment/room_1_2021_8_7_0900_2100.csv"
    """

    # 定义源文件夹和目标文件夹名称
    # 注意：这些路径是相对于脚本运行目录的
    source_dir_name = "expert_data_augment"
    output_dir_name = "expert_data_downsampling"

    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(output_dir_name, exist_ok=True)

    # 从输入路径中提取文件名，并构造输出文件的完整路径
    filename = os.path.basename(input_filepath)
    output_filepath = os.path.join(output_dir_name, filename)

    print(f"正在处理文件: {input_filepath}")
    print(f"结果将保存至: {output_filepath}")

    try:
        # 1. 读取CSV文件
        df = pd.read_csv(input_filepath)

        # 检查'date'和'time'列是否存在
        if 'date' not in df.columns or 'time' not in df.columns:
            raise ValueError("CSV文件中必须包含'date'和'time'列。")

        # 2. 合并'date'和'time'列，创建datetime对象，并设置为DataFrame的索引
        # 指定格式以确保正确解析日期和时间
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M')
        df = df.set_index('datetime')

        # 3. 筛选时间段在09:00到19:00之间的数据（包含起始和结束时间）
        df_filtered = df.between_time('09:00', '19:00')

        # 4. 每隔5分钟取一次数据
        # '5T'表示5分钟。'.first()'表示在每个5分钟间隔内，取第一个可用的数据点。
        df_downsampled = df_filtered.resample('5T').first()

        # 移除因没有数据而导致所有值都为NaN的行（例如，某个5分钟间隔内没有原始数据点）
        df_downsampled = df_downsampled.dropna(how='all')

        # 5. 重置索引，并重新创建'date'和'time'列，以匹配原始数据格式
        df_downsampled = df_downsampled.reset_index()
        df_downsampled['date'] = df_downsampled['datetime'].dt.strftime('%Y/%m/%d')
        df_downsampled['time'] = df_downsampled['datetime'].dt.strftime('%H:%M')

        # 6. 调整列的顺序，使'date'和'time'列回到最前面，并保留其他列的原始顺序
        # 首先获取原始CSV文件的列名（不包括临时的datetime列）
        # 使用nrows=0快速读取列名，避免读取整个文件
        original_cols_template = [col for col in pd.read_csv(input_filepath, nrows=0).columns if
                                  col not in ['date', 'time']]

        # 构建最终输出的列顺序
        output_cols_order = ['date', 'time'] + original_cols_template

        # 确保所有期望的列都在下采样后的DataFrame中，如果缺少则跳过这些列
        final_cols_to_select = [col for col in output_cols_order if col in df_downsampled.columns]

        df_final = df_downsampled[final_cols_to_select]

        # 7. 保存处理后的DataFrame到新的CSV文件
        df_final.to_csv(output_filepath, index=False)
        print(f"文件 '{filename}' 处理成功并已保存。")

    except FileNotFoundError:
        print(f"错误：文件 '{input_filepath}' 未找到。请确保路径正确。")
    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{input_filepath}' 是空的或格式不正确。")
    except Exception as e:
        print(f"处理文件 '{filename}' 时发生错误: {e}")


# 示例用法：
if __name__ == "__main__":
    source_dir = "expert_data_augment"
    output_dir = "expert_data_downsampling"
    os.makedirs(output_dir, exist_ok=True)

    # 构造一个示例CSV文件的路径
    # dummy_file_name = "room_1_2021_8_7_0900_2100.csv"
    # dummy_file_path = os.path.join(source_dir, dummy_file_name)
    #
    # process_and_downsample_csv(dummy_file_path)

    # --- 批量处理多个文件示例 (可选) ---
    # 如果您有多个文件需要处理，可以使用以下方式遍历 'expert_data_augment' 文件夹：
    print("\n--- 批量处理示例 ---")
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".csv"):
            full_input_path = os.path.join(source_dir, file_name)
            process_and_downsample_csv(full_input_path)
