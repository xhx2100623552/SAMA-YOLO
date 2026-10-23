# import matplotlib.pyplot as plt
# import numpy as np

# # ====================== 1. 数据集数据 ======================
# # M3FD Dataset (左图)
# alpha_m3fd = [0.1, 0.3, 0.5, 0.7, 0.9]
# mAP_m3fd = [87.4, 87.6, 88.4, 88.7, 88.3]    # 红色实线 mAP@0.5
# recall_m3fd = [83.6, 82.6, 81.5, 84.0, 82.8] # 蓝色虚线 Recall

# # DroneVehicle Dataset (右图)
# alpha_dv = [0.1, 0.4, 0.5, 0.7]
# mAP_dv = [75.6, 75.2, 74.2, 74.5]
# recall_dv = [72.4, 73.0, 73.4, 72.4]

# # ====================== 2. SPIE期刊全局无网格配置 ======================
# plt.rcParams["axes.grid"] = False
# plt.rcParams["axes.facecolor"] = "white"
# plt.rcParams["figure.facecolor"] = "white"
# # 隐藏右上边框，画面干净
# plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.spines.right"] = False

# # ====================== 3. 创建1行2列并排子图 ======================
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# # ---------------------- 左子图：M3FD Dataset ----------------------
# # 双Y轴创建
# ax1_twin = ax1.twinx()
# # 绘制mAP红色实线，圆点标记
# line1_m3fd = ax1.plot(alpha_m3fd, mAP_m3fd, color="#c42e2e", marker="o", linewidth=2, label="mAP@0.5")
# # 绘制Recall蓝色虚线，方块标记
# line2_m3fd = ax1_twin.plot(alpha_m3fd, recall_m3fd, color="#4C72B0", linestyle="--", marker="s", linewidth=2, label="Recall")

# ax1.set_title("Impact of $\\alpha$ on M3FD Dataset", fontsize=14)
# ax1.set_xlabel("NWD Weight $\\alpha$", fontsize=12)
# ax1.set_ylabel("mAP@0.5 (%)", color="#c42e2e", fontsize=12)
# ax1_twin.set_ylabel("Recall (%)", color="#4C72B0", fontsize=12)
# ax1.tick_params(axis="y", labelcolor="#c42e2e")
# ax1_twin.tick_params(axis="y", labelcolor="#4C72B0")
# ax1.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

# # ---------------------- 右子图：DroneVehicle Dataset ----------------------
# ax2_twin = ax2.twinx()
# line1_dv = ax2.plot(alpha_dv, mAP_dv, color="#c42e2e", marker="o", linewidth=2, label="mAP@0.5")
# line2_dv = ax2_twin.plot(alpha_dv, recall_dv, color="#4C72B0", linestyle="--", marker="s", linewidth=2, label="Recall")

# ax2.set_title("Impact of $\\alpha$ on DroneVehicle Dataset", fontsize=14)
# ax2.set_xlabel("NWD Weight $\\alpha$", fontsize=12)
# ax2.set_ylabel("mAP@0.5 (%)", color="#c42e2e", fontsize=12)
# ax2_twin.set_ylabel("Recall (%)", color="#4C72B0", fontsize=12)
# ax2.tick_params(axis="y", labelcolor="#c42e2e")
# ax2_twin.tick_params(axis="y", labelcolor="#4C72B0")
# ax2.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])

# # 统一图例（放右上角，不遮挡曲线）
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax1_twin.get_legend_handles_labels()
# fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right", bbox_to_anchor=(0.98, 0.96), frameon=False)

# # 紧凑排版，裁掉多余白边
# plt.tight_layout()
# # 输出矢量PDF，适配SPIE投稿
# plt.savefig("alpha_trend.pdf", bbox_inches="tight")
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 1. 准备全量数据
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# M3FD 数据 (左图)
m3fd_map = [0.875, 0.876, 0.884, 0.887, 0.883]
m3fd_recall = [0.837, 0.826, 0.822, 0.845, 0.828]

# DroneVehicle 数据 (右图，含最新加入的 0.3 和 0.9)
dv_map = [0.756, 0.749, 0.752, 0.745, 0.751]
dv_recall = [0.725, 0.722, 0.735, 0.724, 0.723]

# 2. 设置学术风格画布 (宽 14，高 5.5)，两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ================= 左图：M3FD Dataset =================
# 左边画 mAP@0.5 (红色实线)
ax1.plot(alpha_values, m3fd_map, color='#c94c4c', marker='o', linestyle='-', linewidth=2, label='mAP@0.5')
ax1.set_xlabel('NWD Weight α', fontsize=12, fontweight='bold')
ax1.set_ylabel('mAP@0.5', color='#c94c4c', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#c94c4c')
ax1.set_xticks(alpha_values)
ax1.grid(False) # 学术图不画网格
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#c94c4c')

# 右边画 Recall (蓝色虚线)
ax1_twin = ax1.twinx()
ax1_twin.plot(alpha_values, m3fd_recall, color='#4f7cba', marker='s', linestyle='--', linewidth=2, label='Recall')
ax1_twin.set_ylabel('Recall', color='#4f7cba', fontsize=12, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor='#4f7cba')
ax1_twin.spines['top'].set_visible(False)
ax1_twin.spines['right'].set_color('#4f7cba')

ax1.set_title('Impact of α on M3FD Dataset', fontsize=13, fontweight='bold')

# ================= 右图：DroneVehicle Dataset =================
# 左边画 mAP@0.5 (红色实线)
ax2.plot(alpha_values, dv_map, color='#c94c4c', marker='o', linestyle='-', linewidth=2, label='mAP@0.5')
ax2.set_xlabel('NWD Weight α', fontsize=12, fontweight='bold')
ax2.set_ylabel('mAP@0.5', color='#c94c4c', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#c94c4c')
ax2.set_xticks(alpha_values)
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#c94c4c')

# 右边画 Recall (蓝色虚线)
ax2_twin = ax2.twinx()
ax2_twin.plot(alpha_values, dv_recall, color='#4f7cba', marker='s', linestyle='--', linewidth=2, label='Recall')
ax2_twin.set_ylabel('Recall', color='#4f7cba', fontsize=12, fontweight='bold')
ax2_twin.tick_params(axis='y', labelcolor='#4f7cba')
ax2_twin.spines['top'].set_visible(False)
ax2_twin.spines['right'].set_color('#4f7cba')

ax2.set_title('Impact of α on DroneVehicle Dataset', fontsize=13, fontweight='bold')

# 3. 合并图例并放在底部（完美复刻原图风格）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
# 把图例放在左图的下方，或者整个图的底部
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)

# 4. 保存为高清 PDF
plt.tight_layout()
plt.savefig('alpha_trend_new.pdf', dpi=300, bbox_inches='tight')
print("大功告成！带最新 α=0.3 和 0.9 数据的 alpha_trend_new.pdf 已生成！")