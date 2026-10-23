import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 填入您的真实实验数据
# ==========================================
models = ['YOLOv5s', 'YOLOv8s', 'MMFN', 'Fusion-Mamba', 'CTU-YOLO', 'SAMA-YOLO (Ours)']

# 参数量 Parameters (M) -> 越小越好
params = [7.2, 11.1, 45.3, 35.8, 25.4, 4.48]  # SASH-YOLO 填了您的 4.48M

# 推理速度 FPS -> 越大越好
fps = [85, 72, 24, 30, 45, 63.79]             # SASH-YOLO 填了您的 63.79 FPS

# 精度 mAP@0.5 (%) -> 越大越好
map50 = [85.0, 86.5, 86.2, 88.0, 86.8, 88.4]  # SASH-YOLO 填了您的 88.4%

# 定义每个模型的颜色和形状 (保持和顶会一致的学术配色)
colors = ['#8C8C8C', '#8C8C8C', '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']
markers = ['o', 's', '^', 'D', 'v', '*']

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ==========================================
# 2. 绘制左图：Accuracy vs. Parameters
# ==========================================
for i in range(len(models)):
    # 画散点，SASH-YOLO 的星星画大一点 (s=300)
    size = 300 if 'Ours' in models[i] else 150
    ax1.scatter(params[i], map50[i], color=colors[i], marker=markers[i], s=size, zorder=3)
    # 添加文字标签
    ax1.annotate(models[i], (params[i], map50[i]), textcoords="offset points", xytext=(0,12), ha='center', fontsize=10)

# 计算并绘制帕累托前沿线 (参数越小越好)
sorted_indices = np.argsort(params)
pareto_params, pareto_map = [], []
current_max_map = -1
for idx in sorted_indices:
    if map50[idx] > current_max_map:
        pareto_params.append(params[idx])
        pareto_map.append(map50[idx])
        current_max_map = map50[idx]
ax1.plot(pareto_params, pareto_map, color='#D62728', linestyle='--', alpha=0.5, linewidth=2, zorder=2)

ax1.set_xlabel('Parameters (M)', fontweight='bold')
ax1.set_ylabel('mAP@0.5 (%)', fontweight='bold')
ax1.set_title('Accuracy vs. Parameters', fontweight='bold')
# ax1.grid(True, linestyle=':', alpha=0.7, zorder=1)

# ==========================================
# 3. 绘制右图：Accuracy vs. Inference Speed
# ==========================================
for i in range(len(models)):
    size = 300 if 'Ours' in models[i] else 150
    ax2.scatter(fps[i], map50[i], color=colors[i], marker=markers[i], s=size, label=models[i], zorder=3)
    ax2.annotate(models[i], (fps[i], map50[i]), textcoords="offset points", xytext=(0,12), ha='center', fontsize=10)

# 计算并绘制帕累托前沿线 (FPS越大越好)
sorted_indices_fps = np.argsort(fps)[::-1]
pareto_fps, pareto_map_fps = [], []
current_max_map_fps = -1
for idx in sorted_indices_fps:
    if map50[idx] > current_max_map_fps:
        pareto_fps.append(fps[idx])
        pareto_map_fps.append(map50[idx])
        current_max_map_fps = map50[idx]
ax2.plot(pareto_fps, pareto_map_fps, color='#D62728', linestyle='--', alpha=0.5, linewidth=2, label='Pareto Frontier', zorder=2)

ax2.set_xlabel('Inference Speed (FPS)', fontweight='bold')
ax2.set_ylabel('mAP@0.5 (%)', fontweight='bold')
ax2.set_title('Accuracy vs. Inference Speed', fontweight='bold')
# ax2.grid(True, linestyle=':', alpha=0.7, zorder=1)

# 添加图例 (放在右下角)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)

# ==========================================
# 4. 保存高清图片
# ==========================================
plt.tight_layout()
# 保存为 png 和 pdf (pdf 格式插入 LaTeX 放大不失真！)
plt.savefig('pareto_plot.png', dpi=600, bbox_inches='tight')
plt.savefig('pareto_plot.pdf', bbox_inches='tight')
print("绘制成功！已保存为 pareto_plot.png 和 pareto_plot.pdf")