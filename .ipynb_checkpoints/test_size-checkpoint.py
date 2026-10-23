# import os
# import numpy as np
# from pathlib import Path

# def compute_area_distribution(label_dir, img_size=640):
#     areas = []
#     label_files = list(Path(label_dir).glob('*.txt'))
#     for lf in label_files:
#         with open(lf, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) == 5:
#                     _, xc, yc, w, h = map(float, parts)
#                     w_abs = w * img_size
#                     h_abs = h * img_size
#                     areas.append(w_abs * h_abs)
#     areas = np.array(areas)
#     print(f"Total boxes: {len(areas)}")
#     print(f"Min area: {areas.min():.1f}, Max area: {areas.max():.1f}")
#     print(f"Mean area: {areas.mean():.1f}, Median area: {np.median(areas):.1f}")
#     print(f"25th percentile: {np.percentile(areas, 25):.1f}")
#     print(f"75th percentile: {np.percentile(areas, 75):.1f}")
#     # 建议阈值：取 median 或 25th percentile
#     recommended_threshold = np.percentile(areas, 25)  # 或者 np.median(areas)
#     print(f"\nRecommended area threshold for NWD weighting: {recommended_threshold:.0f}")
#     return areas

# if __name__ == '__main__':
#     # 修改为你的训练集标签路径
#     compute_area_distribution('/root/autodl-tmp/RealData/DroneVehicle/labels/train', img_size=640)
import cv2
import os
img_path = "/root/autodl-tmp/VEDAI_YOLO_Format/images/val/00000011.png"
label_path = img_path.replace("images", "labels").replace(".png", ".txt")
if os.path.exists(label_path):
    with open(label_path) as f:
        lines = f.readlines()
    print(f"Label file has {len(lines)} objects")
    for line in lines:
        cls, x, y, w, h = map(float, line.split())
        print(f"class={int(cls)}, cx={x}, cy={y}, w={w}, h={h}")
else:
    print("Label file missing")