import matplotlib.pyplot as plt
import numpy as np

# 数据准备
classes = ['Car', 'Freight-Car', 'Truck', 'Bus', 'Van', 'mAP@0.5']
ctu_yolo = [96.9, 55.9, 65.7, 95.7, 55.6, 73.9]
sash_yolo = [95.1, 43.9, 79.5, 94.9, 64.5, 75.6]

x = np.arange(len(classes))
width = 0.35

# 启用顶级学术排版风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

# 使用高级的深海蓝与赤红色对比，加入黑色边框增加立体感
color1 = '#2b5b84' # 深海蓝 (CTU-YOLO)
color2 = '#c1272d' # 赤红 (SASH-YOLO)

rects1 = ax.bar(x - width/2, ctu_yolo, width, label='CTU-YOLO (Competitor)', color=color1, edgecolor='black', linewidth=1, alpha=0.85)
rects2 = ax.bar(x + width/2, sash_yolo, width, label='SASH-YOLO (Ours)', color=color2, edgecolor='black', linewidth=1, alpha=0.95)

# 加入隐约的横向虚线网格
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)

# 轴标签与标题定制
ax.set_ylabel('AP@0.5 (%)', fontsize=14, fontweight='bold')
ax.set_title('Per-Class Performance Comparison on DroneVehicle Dataset', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=13, fontweight='bold')
ax.tick_params(axis='y', labelsize=12)

# 去除上方和右侧的边框 (极简学术风)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 在每个柱子上方自动标注数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 垂直偏移4个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

autolabel(rects1)
autolabel(rects2)

# --- 终极杀招：高亮巨大的性能提升 ---
# Truck (索引2)
truck_diff = sash_yolo[2] - ctu_yolo[2]
ax.annotate(f'+{truck_diff:.1f}%', xy=(x[2] + width/2, sash_yolo[2] + 4), 
            xytext=(x[2] + width/2, sash_yolo[2] + 16),
            color='#c1272d', fontweight='bold', fontsize=14, ha='center',
            arrowprops=dict(facecolor='#c1272d', shrink=0.05, width=2, headwidth=8))

# Van (索引4)
van_diff = sash_yolo[4] - ctu_yolo[4]
ax.annotate(f'+{van_diff:.1f}%', xy=(x[4] + width/2, sash_yolo[4] + 4), 
            xytext=(x[4] + width/2, sash_yolo[4] + 16),
            color='#c1272d', fontweight='bold', fontsize=14, ha='center',
            arrowprops=dict(facecolor='#c1272d', shrink=0.05, width=2, headwidth=8))

# 图例放在底部，不遮挡数据
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=13, frameon=False)

plt.tight_layout()
# 存为高精度 PDF 矢量图
plt.savefig('bar_chart_dv_advanced.pdf', bbox_inches='tight')
plt.show()