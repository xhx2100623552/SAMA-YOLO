# import matplotlib.pyplot as plt
# import numpy as np

# # ========== 严格对应论文表格真实数据 ==========
# # M3FD 数据集
# alpha_m3fd = [0.1, 0.3, 0.5, 0.7, 0.9]
# mAP_m3fd = [0.875, 0.876, 0.884, 0.887, 0.883]    # mAP@0.5
# recall_m3fd = [0.837, 0.826, 0.822, 0.845, 0.828]# Recall

# # DroneVehicle 数据集
# alpha_dv = [0.1, 0.5, 0.7]
# mAP_dv = [0.756, 0.752, 0.745]
# recall_dv = [0.725, 0.735, 0.724]

# # ========== SPIE 期刊无网格纯白全局配置 ==========
# plt.rcParams["axes.grid"] = False
# plt.rcParams["axes.facecolor"] = "white"
# plt.rcParams["figure.facecolor"] = "white"
# plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.spines.right"] = False

# # ========== 1行2张子图 ==========
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

# # ---------------- 左图 M3FD ----------------
# ax1_twin = ax1.twinx()
# # 红色实线圆点：mAP
# l1 = ax1.plot(alpha_m3fd, mAP_m3fd, color="#C44E52", marker="o", linewidth=2.2, label="mAP@0.5")
# # 蓝色虚线方块：Recall
# l2 = ax1_twin.plot(alpha_m3fd, recall_m3fd, color="#4C72B0", marker="s", linestyle="--", linewidth=2.2, label="Recall")

# ax1.set_title("Impact of $\\alpha$ on M3FD Dataset", fontsize=13)
# ax1.set_xlabel("NWD Weight $\\alpha$", fontsize=11)
# ax1.set_ylabel("mAP@0.5", color="#C44E52", fontsize=11)
# ax1_twin.set_ylabel("Recall", color="#4C72B0", fontsize=11)
# ax1.tick_params(axis='y', labelcolor="#C44E52")
# ax1_twin.tick_params(axis='y', labelcolor="#4C72B0")
# # 只显示真实实验α，去除多余中间刻度
# ax1.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
# ax1.set_xlim(0.05, 0.95)

# # ---------------- 右图 DroneVehicle ----------------
# ax2_twin = ax2.twinx()
# l3 = ax2.plot(alpha_dv, mAP_dv, color="#C44E52", marker="o", linewidth=2.2, label="mAP@0.5")
# l4 = ax2_twin.plot(alpha_dv, recall_dv, color="#4C72B0", marker="s", linestyle="--", linewidth=2.2, label="Recall")

# ax2.set_title("Impact of $\\alpha$ on DroneVehicle Dataset", fontsize=13)
# ax2.set_xlabel("NWD Weight $\\alpha$", fontsize=11)
# ax2.set_ylabel("mAP@0.5", color="#C44E52", fontsize=11)
# ax2_twin.set_ylabel("Recall", color="#4C72B0", fontsize=11)
# ax2.tick_params(axis='y', labelcolor="#C44E52")
# ax2_twin.tick_params(axis='y', labelcolor="#4C72B0")
# ax2.set_xticks([0.1, 0.5, 0.7])
# ax2.set_xlim(0.05, 0.75)

# # 合并统一图例
# lines = l1 + l2
# labels = [line.get_label() for line in lines]
# fig.legend(lines, labels, loc="upper right", frameon=False, fontsize=11)

# plt.tight_layout()
# # ========== 输出高清PNG，dpi=1200 满足SPIE期刊要求 ==========
# plt.savefig("alpha_trend.png", dpi=600, bbox_inches="tight")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 1. 像素级还原主公原图中的核心实验指标
categories = ['Car', 'Freight-Car', 'Truck', 'Bus', 'Van', 'mAP@0.5']
ctu_yolo = [96.9, 55.9, 65.7, 95.7, 55.6, 73.9]
sama_yolo = [95.1, 43.9, 79.5, 94.9, 64.5, 75.6]

x = np.arange(len(categories))
width = 0.35  # 柱子宽度

# 建立画布，锁死高级学术尺寸
fig, ax = plt.subplots(figsize=(10, 5.5))

# 2. 绘制大阵柱体：注入学术黑框(edgecolor)与黑白可分的双向斜线条纹(hatch)
rects1 = ax.bar(x - width/2, ctu_yolo, width, label='CTU-YOLO (Competitor)',
                color='#4f7cba', edgecolor='black', linewidth=1.2, hatch='//')

rects2 = ax.bar(x + width/2, sama_yolo, width, label='SAMA-YOLO (Ours)',
                color='#c94c4c', edgecolor='black', linewidth=1.2, hatch='\\\\')

# 3. 严格遵循官方指南：关闭一切背景网格辅助线(Grid)，保持高冷纯白背景
ax.grid(False)
ax.set_facecolor('white')

# 4. 完美克隆图表文本与坐标轴格式
ax.set_ylabel('AP@0.5 (%)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Per-Class Performance Comparison on DroneVehicle Dataset', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 110)

# 5. 在柱体上方自动咬合标注精确的数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 向上偏移4个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# 6. 精准复刻红线增长箭头：整体上移避开数字
# 标注 Truck 增长 (+13.8%)
# 修改点：xy（箭头尖）从 79.5 抬升到 84；xytext（文字）从 92 抬升到 96
ax.annotate('+13.8%', xy=(2 + width/2, 84), xytext=(2 + width/2, 96),
            arrowprops=dict(facecolor='#a62424', edgecolor='#a62424', shrink=0.05, width=1.2, headwidth=6, headlength=6),
            ha='center', va='bottom', color='#a62424', fontsize=11, fontweight='bold')

# 标注 Van 增长 (+8.9%)
# 修改点：xy（箭头尖）从 64.5 抬升到 69；xytext（文字）从 77 抬升到 81
ax.annotate('+8.9%', xy=(4 + width/2, 69), xytext=(4 + width/2, 81),
            arrowprops=dict(facecolor='#a62424', edgecolor='#a62424', shrink=0.05, width=1.2, headwidth=6, headlength=6),
            ha='center', va='bottom', color='#a62424', fontsize=11, fontweight='bold')

# 7. 剥离冗余边界，只留学术底线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# 8. 放置下置平行图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11)

# 保存为符合 Overleaf 调用的高清 PDF 矢量图，绝不失真
plt.tight_layout()
plt.savefig('bar_per_class_spie.pdf', dpi=300, bbox_inches='tight')
print("大功告成！高阶学术纯白条纹柱状图已生成为 bar_per_class_spie.pdf！")
