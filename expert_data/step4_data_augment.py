import pandas as pd
import os
import shutil
import random  # 引入 random 模块用于随机选择

# --- 1. 定义常量和配置 ---
input_dir = "step2_expert_data_clean"
output_dir = "step4_expert_data_augment"
year = 2021
month = 8  # 八月

# 所有房间和需要生成的目标日期
target_dates = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]

# 房间和可用日期的映射表 (从问题描述中获取)
room_availability_data = {
    "room_1": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
    "room_2": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
    "room_3": [],  # 无可用日期 (没有人员数据)
    "room_4": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
    "room_5": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
    "room_6": [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21],
    "room_7": [16, 17, 18, 20, 21]
}
all_rooms = sorted(list(room_availability_data.keys()))  # 确保房间顺序一致

# room_7 明确缺失的日期 (根据问题描述)
room7_explicitly_missing_dates = [7, 9, 10, 11, 12, 13, 14]

# --- 2. 创建输出目录 ---
os.makedirs(output_dir, exist_ok=True)
print(f"✅ 确保输出目录 '{output_dir}' 存在。")

# --- 3. 收集所有可用的原始数据文件路径，并按日期分类作为模拟模板源 ---
# 字典，用于存储每个日期对应的可用原始文件路径列表
# 例如：{7: ['expert_data_clean/room_1_..._7.csv', 'expert_data_clean/room_2_..._7.csv', ...], 9: [...]}
available_sources_by_day = {day: [] for day in target_dates}

for room, dates in room_availability_data.items():
    # room_3 没有原始数据，不能作为模板源
    if room == "room_3":
        continue
    for day in dates:
        template_filename = f"{room}_{year}_{month}_{day}_0900_1900.csv"
        template_path = os.path.join(input_dir, template_filename)
        # 仅当文件实际存在时才加入模板列表
        if os.path.exists(template_path):
            available_sources_by_day[day].append(template_path)
        else:
            print(f"[❗ 警告] 模板源文件不存在：'{template_path}'。将跳过此文件作为模板。")

# 检查是否收集到任何模板
total_available_templates = sum(len(paths) for paths in available_sources_by_day.values())
if total_available_templates == 0:
    print("❌ 错误：在 expert_data_clean 目录中没有找到任何可用作模板的原始数据文件。无法进行数据模拟。请确保 'expert_data_clean' 文件夹中包含原始数据文件。")
    exit()
else:
    print(f"✅ 已收集 {total_available_templates} 个可用作模拟模板的原始数据文件，并已按日期分类。")

# --- 4. 遍历所有房间和日期，进行数据处理 ---
print("\n--- 开始处理数据：生成/复制文件至 expert_data_augment ---")

for room in all_rooms:
    for day in target_dates:
        # 构建当前文件在 expert_data_clean 和 expert_data_augment 中的路径
        output_filename = f"{room}_{year}_{month}_{day}_0900_1900.csv"  # day:02d 格式化为两位数，如 07
        source_path = os.path.join(input_dir, output_filename)
        destination_path = os.path.join(output_dir, output_filename)

        # 判断是否需要生成模拟数据（room_3 的所有日期，或 room_7 明确缺失的日期）
        if room == "room_3" or (room == "room_7" and day in room7_explicitly_missing_dates):
            # 获取当前日期可用的模板文件列表 (即其他房间在当天的文件)
            potential_templates_for_this_day = available_sources_by_day.get(day, [])

            if not potential_templates_for_this_day:
                print(f"[❌ 错误] 无法为 {output_filename} 生成数据。在 expert_data_clean 中没有找到该日期 ({day}) 的任何其他房间数据作为模板。请检查源数据。")
                continue  # 跳过当前文件，继续下一个，因为没有模板可以用来生成

            # 从当前日期可用的模板中随机选择一个文件路径
            random_template_path = random.choice(potential_templates_for_this_day)

            try:
                # 读取随机选择的模板 DataFrame
                df_to_save = pd.read_csv(random_template_path)
                # 修改 'date' 列为目标日期
                df_to_save['date'] = f"{year}/{month:02d}/{day}"
                # 保存到目标路径
                df_to_save.to_csv(destination_path, index=False)
                # 打印生成信息，显示使用了哪个随机模板
                print(f"[✅ 生成] {output_filename} (使用随机模板: {os.path.basename(random_template_path)})")
            except Exception as e:
                print(f"[❌ 错误] 无法为 {output_filename} 生成数据。模板文件 '{random_template_path}' 读取失败: {e}")
        else:
            # 对于其他情况 (room_1,2,4,5,6 的所有目标日期，以及 room_7 已有的日期)，
            # 直接从 expert_data_clean 复制文件
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"[➡️ 复制] {output_filename} (从 {input_dir})")
            else:
                # 理论上，如果 room_availability_data 准确，并且 expert_data_clean 包含所有应有的文件，此警告不应出现
                print(f"[❗ 警告] 预期的文件未找到：'{source_path}'。无法复制。请检查源数据或 room_availability_data 配置。")

print("\n--- 数据处理完成 ---")
print(f"所有扩增/复制的文件已成功保存至 '{output_dir}' 文件夹。")
