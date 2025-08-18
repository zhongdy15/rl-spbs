import os
import csv


def clean_csv_file(input_filepath, output_directory, output_filename):
    """
    清理CSV文件：
    1. 根据行数据列数删除特定列。
    2. 统一日期格式为 'YYYY/M/D'。
    3. 将清理后的数据保存到指定的新文件中。

    Args:
        input_filepath (str): 输入CSV文件的完整路径。
        output_directory (str): 存放输出CSV文件的新文件夹路径。
        output_filename (str): 输出CSV文件的文件名。
    """

    # 定义期望的表头和列数
    expected_header = [
        'date', 'time', 'FCU_temp_setpoint', 'FCU_onoff_setpoint', 'FCU_fan_setpoint',
        'FCU_temp_feedback', 'FCU_mode_setpoint', 'FCU_temp_setpoint_feedback',
        'FCU_onoff_feedback', 'FCU_lock_feedback', 'FCU_fan_feedback',
        'FCU_mode_feedback', 'supply_pressure', 'return_pressure', 'waterflow',
        'supply_temp', 'return_temp', 'occupant_num', 'occupant_transform',
        'room_temp1', 'room_RH1', 'room_temp2', 'room_RH2', 'differential_pressure'
    ]
    expected_num_columns = len(expected_header)

    # 构建输出文件的完整路径
    os.makedirs(output_directory, exist_ok=True)  # 如果目录不存在则创建
    output_filepath = os.path.join(output_directory, output_filename)

    processed_rows_count = 0
    skipped_rows_count = 0

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
                open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # --- 关键修改：处理输入文件的表头 ---
            try:
                # 读取原始文件的表头行，并将其从后续数据中跳过
                input_header = next(reader)
                print(f"已从原始文件读取表头: {input_header}")
                if len(input_header) != expected_num_columns:
                    print(f"警告：原始文件表头列数 ({len(input_header)}) 与期望列数 ({expected_num_columns}) 不符。")
                elif input_header != expected_header:
                    print(f"提示：原始文件表头与预定义表头内容不完全一致。将使用预定义表头。")
            except StopIteration:
                print("错误：输入文件为空，没有表头或数据。无法进行清理。")
                return  # 提前退出函数
            except Exception as e:
                print(f"读取原始文件表头时发生错误：{e}")
                return  # 提前退出函数

            # 写入期望的表头到新文件
            writer.writerow(expected_header)
            print(f"已将预定义表头写入新文件: {output_filepath}")

            # 逐行读取并处理数据
            # enumerate 从 i=0 开始计数，表示当前是读取的第几行数据（不包括已读取的表头）
            for i, row in enumerate(reader):
                line_num = i + 2  # 实际行号，因为跳过了表头，所以数据行从文件第2行开始计数

                # 确保行不为空
                if not row:
                    skipped_rows_count += 1
                    print(f"警告: 第 {line_num} 行为空，已跳过。")
                    continue

                current_cols = len(row)
                cleaned_row = list(row)  # 创建一个可修改的副本

                # 1. 根据列数删除特定列
                if current_cols == expected_num_columns:
                    # 正常列数，无需删除
                    pass
                elif current_cols == 28:
                    # 删除第13, 14, 15, 16列 (Python索引为12, 13, 14, 15)
                    indices_to_delete = sorted([12, 13, 14, 15], reverse=True)  # 从大到小删除，避免索引变化问题
                    for idx in indices_to_delete:
                        if idx < len(cleaned_row):  # 确保索引有效
                            del cleaned_row[idx]
                    if len(cleaned_row) != expected_num_columns:
                        print(
                            f"警告: 第 {line_num} 行 (原始28列) 处理后列数不符 ({len(cleaned_row)} != {expected_num_columns})，跳过此行。")
                        skipped_rows_count += 1
                        continue
                elif current_cols == 29:
                    # 删除第8, 14, 15, 16, 17列 (Python索引为7, 13, 14, 15, 16)
                    indices_to_delete = sorted([7, 13, 14, 15, 16], reverse=True)
                    for idx in indices_to_delete:
                        if idx < len(cleaned_row):
                            del cleaned_row[idx]
                    if len(cleaned_row) != expected_num_columns:
                        print(
                            f"警告: 第 {line_num} 行 (原始29列) 处理后列数不符 ({len(cleaned_row)} != {expected_num_columns})，跳过此行。")
                        skipped_rows_count += 1
                        continue
                else:
                    # 其他非预期列数，跳过此行并记录
                    print(f"警告: 第 {line_num} 行有非预期的列数 {current_cols}，已跳过此行。")
                    skipped_rows_count += 1
                    continue

                # 2. 统一日期格式为 'YYYY/M/D'
                # 假设 'date' 列始终是第一个（索引为0）
                if len(cleaned_row) > 0:  # 确保行不为空
                    date_str = str(cleaned_row[0])  # 确保是字符串
                    cleaned_row[0] = date_str.replace('-', '/')  # 将 '-' 替换为 '/'

                # 将处理后的行写入新文件
                writer.writerow(cleaned_row)
                processed_rows_count += 1

        print(f"\n清理完成！")
        print(f"原始文件: '{input_filepath}'")
        print(f"清理后的文件保存到: '{output_filepath}'")
        print(f"成功处理行数: {processed_rows_count}")
        print(f"跳过行数: {skipped_rows_count}")

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_filepath}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")


# --- 调用清理函数 ---
if __name__ == "__main__":
    for room_id in range(1, 8):
        file_name = f'room_{room_id}_result.csv'
        input_csv_path = f'expert_data/{file_name}'
        output_folder = 'expert_data_new'
        output_csv_filename = file_name

        clean_csv_file(input_csv_path, output_folder, output_csv_filename)
