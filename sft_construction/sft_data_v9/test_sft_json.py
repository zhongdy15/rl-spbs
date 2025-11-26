import json
import os


def test_sft_dataset(json_path):
    # 1. 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    print(f"正在读取 {json_path} ...")

    # 2. 读取 JSON 文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except json.JSONDecodeError:
        print("错误: 文件不是有效的 JSON 格式。")
        return

    # 3. 基本信息统计
    print(f"成功加载！数据集包含 {len(dataset)} 条样本。")
    print("-" * 50)

    if len(dataset) == 0:
        print("警告: 数据集为空。")
        return

    # 4. 提取第一条数据
    first_sample = dataset[120]

    # 5. 打印 Instruction (用户指令)
    print("【Instruction (Input)】:")
    print(first_sample.get("instruction", "Key 'instruction' missing"))

    print("\n" + "-" * 50 + "\n")

    # 6. 打印 Output (模型回答)
    print("【Output (Label)】:")
    output_str = first_sample.get("output", "Key 'output' missing")

    # 尝试解析 Output 中的 JSON 字符串以便更漂亮地打印
    try:
        output_json = json.loads(output_str)
        print(json.dumps(output_json, indent=2, ensure_ascii=False))
    except:
        # 如果 output 不是 json 字符串，直接打印原样
        print(output_str)

    print("\n" + "-" * 50)
    print("测试完成。")


if __name__ == "__main__":
    # 确保这里的路径和你生成的文件名一致
    JSON_FILE = "sft_training_data.json"
    test_sft_dataset(JSON_FILE)
