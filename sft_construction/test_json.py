import json


try:
    # 实际场景中，你会这样加载文件：
    with open('all_types_sft_data/type3_adjusted.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查数据是否为空
    if not data:
        print("JSON文件为空或没有数据。")
    else:
        # 获取第一条数据，根据您的JSON结构，是 data[0]
        first_entry = data[0]

        # 构造 prompt
        prompt = f"{first_entry['system']}\n{first_entry['instruction']}\n{first_entry['input']}"

        # 获取 output
        output = first_entry['output']

        print("===== 第一条数据的 Prompt =====")
        # print(r"{}".format(prompt))
        print(prompt.replace("\n", "\\n"))
        print(prompt)

        print("\n===== 第一条数据的 Output =====")
        print(output)

except FileNotFoundError:
    print("错误：文件 'sft_priority_room_dataset.json' 未找到。请确保文件存在于正确路径。")
except json.JSONDecodeError:
    print("错误：JSON文件格式不正确，无法解析。")
except IndexError:
    print("错误：JSON数据结构与预期不符，无法获取第一条数据。")
except KeyError as e:
    print(f"错误：JSON数据缺少预期的键 '{e}'。")

