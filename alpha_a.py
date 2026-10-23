
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