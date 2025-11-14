import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def calculate_and_print_stats(data_dict):
    """
    计算并打印每个算法的统计数据（平均值 ± 标准差）。

    参数:
    data_dict (dict): 一个字典，键是算法名称(str)，值是包含实验数据的二维Numpy数组。
                      数组的每一行代表一次实验，每一列代表一个指标。
    """

    means_list = []
    stds_list = []
    # 遍历字典中的每个算法及其数据
    for algorithm_name, data_array in data_dict.items():

        # 检查数据是否为空
        if data_array.size == 0:
            print(f"| **{algorithm_name}** | {' | '.join(['N/A'] * len(headers))} |")
            continue

        # 沿列（axis=0）计算平均值
        means = np.mean(data_array, axis=0)

        # 沿列（axis=0）计算样本标准差 (ddof=1)
        # ddof=1 表示自由度减1，计算的是无偏估计的样本标准差，这是统计学中的标准做法。
        stds = np.std(data_array, axis=0, ddof=1)

        means_list.append(means)
        stds_list.append(stds)

    return means_list,stds_list


# --- 如何使用 ---

# 1. 定义您的指标名称（表头）
column_headers = ["PPD（越小越好）", "PMV（越小越好）", "能量消耗FCU power", "测试奖励值"]

# 2. 将您的数据定义为Numpy数组
#    每个算法对应一个二维数组，其中每行是一次实验，每列是一个指标。
all_data = {
    'Meta-Llama-3-8B': np.array([
        [27.26, 0.96, 134.71, -17703.1],
        [25.0, 0.9, 137.69, -16379.5],
        [27.26, 0.94, 135.33, -17707.6],
        [30.7, 1.05, 132.21, -19740.2],
        [28.36, 0.99, 132.34, -18339.4]
    ]),

    'Qwen2.5-7B': np.array([
        [20.76, 0.71, 165.37, -14110.4],
        [19.84, 0.70, 156.56, -13467.9],
        [20.0, 0.72, 157.79, -13575.9],
        [20.87, 0.74, 170.69, -14226.9],
        [19.66, 0.69, 162.56, -13424.4]
    ]),

    'Qwen2.5-14B': np.array([
        [15.67, 0.6, 155.95, -10961.3],
        [15.79, 0.59, 157.26, -11046.2],
        [18.07, 0.66, 158.64, -12426.6],
        [16.45, 0.62, 155.62, -11426.1],
        [16.64, 0.63, 155.55, -11539.7],
        [15.71, 0.6, 155.43, -10982.3],
        [16.53, 0.62, 155.15, -11471.7]
    ]),

    'Qwen2.5-32B': np.array([
        [12.55, 0.47, 143.56, -8964.8],
        [12.24, 0.45, 143.83, -8779.4],
        [11.27, 0.44, 142.29, -8185.2],
        [11.25, 0.43, 142.43, -8172.4],
        [12.25, 0.48, 145.61, -8805.3]
    ]),

    'Qwen2.5-72B': np.array([
        [15.03, 0.56, 156.65, -10582.2],
        [16.94, 0.60, 160.35, -11768.6],
        [15.12, 0.58, 156.60, -10636.7],
        [14.75, 0.56, 155.08, -10398.8],
        [15.96, 0.58, 158.43, -11158.8]
    ]),

    'A2C': np.array([
        [13.69, 0.47, 153.86, -9750.3],
        [13.88, 0.46, 154.32, -9871.6],
        [15.77, 0.53, 171.45, -11177.9],
        [13.96, 0.48, 155.00, -9924.5],
        [13.94, 0.45, 172.27, -10089.1]
    ]),

    'PPO': np.array([
        [13.54, 0.44, 151.76, -9642.0],
        [13.75, 0.45, 152.49, -9777.5],
        [14.21, 0.47, 151.01, -10035.0],
        [12.99, 0.44, 149.68, -9290.8],
        [13.46, 0.43, 157.71, -9651.1]
    ]),

    'DQN': np.array([
        [7.76, 0.28, 163.19, -6289.9],
        [8.15, 0.30, 155.98, -6448.2],
        [8.39, 0.32, 156.68, -6599.4],
        [7.33, 0.25, 153.31, -5929.4],
        [8.26, 0.3, 156.47, -6519.6]
    ])
}


means_list,stds_list = calculate_and_print_stats(all_data)
algorithm_list = list(all_data.keys())


# 格式化结果字符串
# 打印表头
print(f"| 算法 | {' | '.join(column_headers)} |")
print(f"|{':---|' * (len(column_headers) + 1)}")
for algorithm_name,means,stds in zip(algorithm_list,means_list,stds_list):
    formatted_results = [f"{mean:.2f} ± {std:.2f}" for mean, std in zip(means, stds)]

    # 打印当前算法的结果行
    print(f"| **{algorithm_name}** | {' | '.join(formatted_results)} |")


# keys_to_plot = ["Meta-Llama-3-8B", "Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B", "A2C", "PPO", "DQN"]
keys_to_plot = ["Meta-Llama-3-8B",  "Qwen2.5-14B", "Qwen2.5-72B", "A2C", "PPO", "DQN"]
name_to_plot_dict = {
    "Meta-Llama-3-8B": "Llama 8B",
    "Qwen2.5-7B": "Qwen 7B",
    "Qwen2.5-14B": "Qwen 14B",
    "Qwen2.5-32B": "Qwen 32B",
    "Qwen2.5-72B": "Qwen 72B",
    "A2C": "A2C",
    "PPO": "PPO",
    "DQN": "DQN"
}

# 2. 准备绘图所需的数据
plot_labels = [name_to_plot_dict[key] for key in keys_to_plot]
ppd_means = []
ppd_stds = []
fcu_means = []
fcu_stds = []

# PPD在第0列，FCU power在第2列
ppd_index = 0
fcu_index = 2

# 从完整的统计结果中提取需要绘图的数据
all_stats = {name: {'mean': mean, 'std': std} for name, mean, std in zip(algorithm_list, means_list, stds_list)}

for key in keys_to_plot:
    stats = all_stats[key]
    ppd_means.append(stats['mean'][ppd_index])
    ppd_stds.append(stats['std'][ppd_index])
    fcu_means.append(stats['mean'][fcu_index])
    fcu_stds.append(stats['std'][fcu_index])

# 3. 开始绘图
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6)) # 创建一个1行2列的图表

# --- 绘制PPD柱状图 ---
ax1 = axes[0]
bars1 = ax1.bar(plot_labels, ppd_means, yerr=ppd_stds, capsize=5, color='skyblue', alpha=0.8, label='PPD均值')
ax1.set_ylabel('PPD (%)')
ax1.set_title('PPD Performance of Different Algorithms')
ax1.set_xticklabels(plot_labels, rotation=45, ha='right') # 旋转x轴标签以防重叠
ax1.grid(axis='y', linestyle='--', alpha=0.7)


# 在每个柱子上方显示数值
index = 0
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + ppd_stds[index] + 0.5, f'{yval:.2f} ± {ppd_stds[index]:.2f}', ha='center', va='bottom')
    index += 1



# --- 绘制FCU power柱状图 ---
ax2 = axes[1]
bars2 = ax2.bar(plot_labels, fcu_means, yerr=fcu_stds, capsize=5, color='lightcoral', alpha=0.8, label='FCU Power均值')
ax2.set_ylabel('FCU Power (W)')
ax2.set_title('FCU Power Consumption of Different Algorithms')
ax2.set_xticklabels(plot_labels, rotation=45, ha='right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 在每个柱子上方显示数值
index = 0
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + fcu_stds[index] + 0.5, f'{yval:.2f} ± {fcu_stds[index]:.2f}', ha='center', va='bottom')
    index += 1

# 自动调整布局并显示图表
plt.tight_layout()
plt.show()