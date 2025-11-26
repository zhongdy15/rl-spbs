import os
import shutil
import re  # 用于更精确地解析文件名中的房间号


def remap_room_data_files(source_dir: str = "expert_data_augment",
                          target_dir: str = "expert_data_remap"):
    """
    遍历指定源目录下的CSV文件，根据预设的房间映射规则重命名文件，
    并将重命名后的文件复制到目标目录。

    Args:
        source_dir (str): 存放原始CSV文件的文件夹路径 (e.g., "expert_data_augment")
        target_dir (str): 存放重新映射后CSV文件的文件夹路径 (e.g., "expert_data_remap")
    """

    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(target_dir, exist_ok=True)
    print(f"源文件夹: {source_dir}")
    print(f"目标文件夹: {target_dir}\n")

    # 定义房间映射规则
    room_mapping = {
        1: 1,  # Room 1 -> Room 1
        2: 2,  # Room 2 -> Room 2
        3: 3,  # Room 3 -> Room 3
        4: 5,  # Room 4 -> Room 5
        5: 6,  # Room 5 -> Room 6
        6: 7,  # Room 6 -> Room 7
        7: 4  # Room 7 -> Room 4
    }

    processed_count = 0
    skipped_count = 0

    if not os.path.exists(source_dir):
        print(f"错误：源文件夹 '{source_dir}' 不存在。请检查路径。")
        return

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_dir):
        # 只处理CSV文件，并且文件名符合 'room_X_...' 格式
        if filename.endswith(".csv") and filename.startswith("room_"):
            source_filepath = os.path.join(source_dir, filename)

            # 使用正则表达式从文件名中提取房间号
            match = re.match(r"room_(\d+)_.*\.csv", filename)
            if match:
                original_room_str = match.group(1)
                try:
                    original_room_num = int(original_room_str)

                    if original_room_num in room_mapping:
                        new_room_num = room_mapping[original_room_num]
                        # 构造新的文件名
                        new_filename = filename.replace(
                            f"room_{original_room_num}", f"room_{new_room_num}"
                        )
                        target_filepath = os.path.join(target_dir, new_filename)

                        try:
                            # 复制文件并重命名
                            shutil.copy2(source_filepath, target_filepath)
                            print(f"已处理: '{filename}' -> '{new_filename}'")
                            processed_count += 1
                        except Exception as e:
                            print(f"错误：复制文件 '{filename}' 到 '{target_filepath}' 失败: {e}")
                            skipped_count += 1
                    else:
                        print(f"警告：文件 '{filename}' 包含未知的房间号 '{original_room_num}'。已跳过。")
                        skipped_count += 1
                except ValueError:
                    print(f"警告：文件 '{filename}' 无法解析房间号。已跳过。")
                    skipped_count += 1
            else:
                print(f"警告：文件 '{filename}' 不符合预期的命名格式 'room_X_...csv'。已跳过。")
                skipped_count += 1
        # else:
        #     print(f"跳过非CSV文件或不符合命名规则的文件: {filename}") # 可以取消注释查看被跳过的文件

    print(f"\n--- 处理完成 ---")
    print(f"成功处理文件数量: {processed_count}")
    print(f"跳过文件数量: {skipped_count}")
    print(f"所有重新映射后的文件已保存到 '{target_dir}' 文件夹。")


# 示例用法：
if __name__ == "__main__":
    # --- 设置示例文件和目录 ---
    source_folder = "step4_expert_data_augment"
    output_folder = "step8_expert_data_remap_without_downsampling"

    # 清理旧的输出目录（可选，方便测试）
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"已清空旧的 '{output_folder}' 文件夹。")


    print("\n--- 开始房间名称重新映射 ---\n")
    # 调用函数处理文件
    remap_room_data_files(source_dir=source_folder, target_dir=output_folder)

    # 验证结果 (可选)
    print("\n--- 验证结果 ---")
    print(f"'{output_folder}' 文件夹中的文件:")
    for f in os.listdir(output_folder):
        print(f)
