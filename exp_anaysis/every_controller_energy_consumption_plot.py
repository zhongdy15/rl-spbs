import matplotlib.pyplot as plt
import numpy as np
constant_1000 = np.array([5.376, 6.912, 5.856, 5.184, 3.072, 4.992, 2.304])
constant_100 = np.array([28.512, 41.856, 36.096, 19.104, 17.952, 25.152, 16.032])
constant_10 = np.array([40.896, 39.648, 41.28, 24.576,22.368,24.192 ,19.872])
constant_0 = np.array([40.896, 36.48, 40.224,24.96,21.888,23.808, 17.472])

energy_per_min = 0.096
constant_1000_opentime = constant_1000/energy_per_min
constant_100_opentime = constant_100/energy_per_min
constant_10_opentime = constant_10/energy_per_min
constant_0_opentime = constant_0/energy_per_min

# print(constant_1000_opentime)
# print(constant_100_opentime)
# print(constant_10_opentime)
# print(constant_0_opentime)

# 画出7个控制器的直方图，每一个控制器画四根柱形
plt.bar(np.arange(7)-0.2, constant_1000_opentime, width=0.2, label='constant 1000')
plt.bar(np.arange(7), constant_100_opentime, width=0.2, label='constant 100')
plt.bar(np.arange(7)+0.2, constant_10_opentime, width=0.2, label='cosntant 10')
plt.bar(np.arange(7)+0.4, constant_0_opentime, width=0.2, label='constant 0')

# 设置x轴刻度标签
plt.xticks(np.arange(7), ['Controller 1', 'Controller 2', 'Controller 3', 'Controller 4', 'Controller 5', 'Controller 6', 'Controller 7'])

# 设置y轴刻度标签
# plt.yticks(np.arange(0, 50, 10))

# 设置图例
plt.legend()

# 设置标题
plt.title('Energy Consumption of Every Controller')

# 显示图形
plt.show()