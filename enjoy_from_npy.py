
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 6),sharex=True)
axes = axes.flatten()

file_name_list = ["exp_anaysis/0122_test_dor_plot/0122_test_dor_plot/0122_Baseline_OCC_PPD_with_energy_const10_DQN",
                  "exp_anaysis/0122_test_dor_plot/0122_test_dor_plot/0122_Baseline_OCC_PPD_with_energy_const10_HGQN",
                  "exp_anaysis/0122_test_dor_plot/0122_test_dor_plot/0122_Baseline_with_energy_const10_DQN",
                  "exp_anaysis/0122_test_dor_plot/0122_test_dor_plot/0122_Baseline_with_energy_const10_HGQN"]


file_name = file_name_list[1]
file_name2 = file_name_list[3]
file_name3 = file_name_list[0]
label1 = "THGQN + PPD-based Reward"
label2 = "THGQN + Fixed Comfort Temperature Range"
label3 = "DQN + PPD-based Reward"
color1 = 'blue'
color2 = 'orange'
color3 = 'green'

frame_skip = 5
temp_time = np.arange(0, 600, frame_skip)
# 子图： FCU Power
ax3 = axes[0]

FCU_power = np.load(file_name + "_FCU.npy") * 60
FCU_power_2 = np.load(file_name2 + "_FCU.npy") * 60
FCU_power_3 = np.load(file_name3 + "_FCU.npy") * 60
FCU_power_total = np.sum(FCU_power)
ax3.plot(temp_time, FCU_power[temp_time], marker='o', linestyle='-', color=color1, label=label1)
ax3.plot(temp_time, FCU_power_2[temp_time], marker='o', linestyle='-', color=color2, label=label2)
# ax3.plot(temp_time, FCU_power_3[temp_time], marker='o', linestyle='-', color=color3, label=label3)
# ax3.set_xlabel('Time Steps')
ax3.set_ylabel('FCU Power (kW)')
# ax3.set_title('Total FCU Power: ' + str(FCU_power_total))
ax3.set_xlim(-20, 601)  # Set x-axis limits
ax3.xaxis.set_ticks(range(0, 601, 60))  # Set x-ticks at intervals of 60
time_labels = ['9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00','18:00',"19:00"]
ax3.set_xticklabels(time_labels)  # Set x-tick labels
ax3.legend(loc='upper left',ncol=2,bbox_to_anchor=(-0.01,1.3))
ax3.grid(True, linestyle='--', linewidth=0.5, color='gray')


# 子图：PMV 的ABS均值
ax4 = axes[1]
# pmv_mean = data_recorder["training"]["mean_pmv"]
pmv_mean = np.load(file_name + "_pmv.npy")
pmv_mean_2 = np.load(file_name2 + "_pmv.npy")
pmv_mean_3 = np.load(file_name3 + "_pmv.npy")
pmv_mean_avarege = np.mean(pmv_mean)
ax4.plot(temp_time, pmv_mean[temp_time], marker='o', linestyle='-', color=color1, label=label1)
ax4.plot(temp_time, pmv_mean_2[temp_time], marker='o', linestyle='-', color=color2, label=label2)
# ax4.plot(temp_time, pmv_mean_3[temp_time], marker='o', linestyle='-', color=color3, label=label3)
ax4.plot([-25, 625], [0.5,0.5], marker='o', linestyle='--', color='red', label='Comfort Range')
# ax4.set_xlabel('Time Steps')
ax4.set_ylabel('PMV Abs Mean')
# ax4.set_title('PMV Mean: '+ str(round(pmv_mean_avarege, 2)))
# ax4.legend(loc='upper left',ncol=2)
ax4.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 子图：PPD的均值
ax5 = axes[2]
# ppd_mean = data_recorder["training"]["mean_ppd"]
ppd_mean = np.load(file_name + "_ppd.npy")
ppd_mean_2 = np.load(file_name2 + "_ppd.npy")
ppd_mean_3 = np.load(file_name3 + "_ppd.npy")
ppd_mean_avarege = np.mean(ppd_mean)
ax5.plot(temp_time, ppd_mean[temp_time], marker='o', linestyle='-', color=color1, label=label1)
ax5.plot(temp_time, ppd_mean_2[temp_time], marker='o', linestyle='-', color=color2, label=label2)
# ax5.plot(temp_time, ppd_mean_3[temp_time], marker='o', linestyle='-', color=color3, label=label3)

ax5.plot([-25, 625], [10,10], marker='o', linestyle='--', color='red', label='Comfort Range')
ax5.set_xlabel('Time')
ax5.set_ylabel('PPD Mean (%)')
# ax5.set_title('PPD Mean: '+str(round(ppd_mean_avarege, 2)))
# ax5.legend(loc='upper left',ncol=2)
ax5.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 调整子图间距
plt.tight_layout()

# plt.show()
plt.savefig("OCC_vs_temp.pdf", dpi=300)




