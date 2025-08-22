import os
import pandas as pd


def analyze_room_control_data(directory_path):
    """
    读取指定文件夹下所有CSV文件中的历史控制数据，
    并分析在不同入住人数情况下的室内平均温度和FCU组合状态。

    参数:
        directory_path (str): 包含CSV文件的目录路径。例如: 'expert_data/expert_data_remap'

    返回:
        dict: 包含分析结果的字典：
              - 'no_occupants': 当 occupant_num (入住人数) 为 0 时的平均室内温度和FCU组合状态。
              - 'with_occupants': 当 occupant_num (入住人数) 大于 0 时，按人数分组的平均室内温度和FCU组合状态。
              如果未找到有效数据或发生错误，则返回 None。
    """
    all_data = []

    # 检查目录是否存在且是目录
    if not os.path.exists(directory_path):
        print(f"错误：未找到指定的目录路径 '{directory_path}'。请检查路径是否正确。")
        return None
    if not os.path.isdir(directory_path):
        print(f"错误：'{directory_path}' 不是一个有效的目录路径。")
        return None

    # 获取目录中所有以 .csv 结尾的文件
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"在目录 '{directory_path}' 中未找到任何CSV文件。")
        return None

    print(f"在 '{directory_path}' 中找到 {len(csv_files)} 个CSV文件。正在读取数据...")

    # 逐个读取CSV文件并合并数据
    for filename in csv_files:
        filepath = os.path.join(directory_path, filename)
        try:
            df = pd.read_csv(filepath)
            all_data.append(df)
            # print(f"成功读取文件: {filename}") # 可以取消注释用于调试
        except Exception as e:
            print(f"读取文件 '{filename}' 时出错，已跳过: {e}")
            continue

    if not all_data:
        print("未能从任何CSV文件中加载有效数据。请检查CSV文件内容和格式。")
        return None

    # 将所有DataFrame合并为一个
    combined_df = pd.concat(all_data, ignore_index=True)

    # 检查合并后的数据中是否包含所有必需的列
    required_cols = ['room_temp1', 'room_temp2', 'occupant_num', 'FCU_combined_state']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]

    if missing_cols:
        print(
            f"错误：合并后的数据中缺少必需的列：{', '.join(missing_cols)}。\n请确保所有CSV文件包含 'room_temp1', 'room_temp2', 'occupant_num', 'FCU_combined_state' 这些列。")
        return None

    # 计算室内平均温度 (room_temp1 和 room_temp2 的平均值)
    combined_df['average_room_temp'] = (combined_df['room_temp1'] + combined_df['room_temp2']) / 2

    results = {}

    # --- 1. 分析当房间里没有人时 (occupant_num = 0) 的数据 ---
    no_occupants_df = combined_df[combined_df['occupant_num'] == 0]
    if not no_occupants_df.empty:
        avg_temp_no_occ = no_occupants_df['average_room_temp'].mean()
        avg_fcu_no_occ = no_occupants_df['FCU_combined_state'].mean()
        results['no_occupants'] = {
            'average_room_temp': avg_temp_no_occ,
            'average_fcu_state': avg_fcu_no_occ
        }
        print("\n--- 分析结果：当房间里没有人时 (occupant_num = 0) ---")
        print(f"  室内平均温度: {avg_temp_no_occ:.2f} °C")
        print(f"  FCU组合状态平均值: {avg_fcu_no_occ:.2f}")
    else:
        print("\n--- 分析结果：未找到当房间里没有人时 (occupant_num = 0) 的数据。---")
        results['no_occupants'] = None

    # --- 2. 分析当房间里有人时 (occupant_num > 0) 的数据 ---
    with_occupants_df = combined_df[combined_df['occupant_num'] > 0]
    if not with_occupants_df.empty:
        # 按 occupant_num 分组，计算室内平均温度和FCU组合状态的平均值
        # 将 occupant_num 转换为整数类型，因为它代表人数
        grouped_data = with_occupants_df.copy()
        grouped_data['occupant_num'] = grouped_data['occupant_num'].astype(int)

        grouped_summary = grouped_data.groupby('occupant_num').agg(
            average_room_temp=('average_room_temp', 'mean'),
            average_fcu_state=('FCU_combined_state', 'mean')
        ).reset_index()

        results['with_occupants'] = grouped_summary
        print("\n--- 分析结果：当房间里有人时 (occupant_num > 0) ---")
        print(grouped_summary.to_string(index=False, float_format="%.2f"))  # 使用 to_string 打印完整的 DataFrame，并格式化浮点数
    else:
        print("\n--- 分析结果：未找到当房间里有人时 (occupant_num > 0) 的数据。---")
        results['with_occupants'] = None

    return results


# --- 使用示例 ---
# 假设您的CSV文件位于 'expert_data/expert_data_remap' 目录下
# 如果该目录不存在，请先创建它并放入示例CSV文件，或者修改为您的实际路径

# 为了演示，我们可以创建一些虚拟数据和目录 (您在实际使用时无需运行此段代码)
if __name__ == "__main__":
    # 创建虚拟目录和文件
    dummy_dir = '../../expert_data/expert_data_remap'
    # 调用函数进行分析
    analysis_results = analyze_room_control_data(dummy_dir)

    if analysis_results:
        print("\n--- 最终结果汇总 ---")
        if analysis_results['no_occupants']:
            print(f"当房间里没有人时 (occupant_num = 0):")
            print(f"  室内平均温度: {analysis_results['no_occupants']['average_room_temp']:.2f} °C")
            print(f"  FCU组合状态平均值: {analysis_results['no_occupants']['average_fcu_state']:.2f}")
        else:
            print("没有找到当房间里没有人时的数据。")

        if analysis_results['with_occupants'] is not None and not analysis_results['with_occupants'].empty:
            print("\n当房间里有人时 (occupant_num > 0):")
            print(analysis_results['with_occupants'].to_string(index=False))
        else:
            print("没有找到当房间里有人时的数据。")

    # 清理虚拟数据和目录 (如果您不需要它们，可以取消注释以下行)
    # for filename in os.listdir(dummy_dir):
    #     os.remove(os.path.join(dummy_dir, filename))
    # os.rmdir(dummy_dir)
