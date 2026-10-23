from ultralytics import YOLO

# 1. 设置你刚才准备的“VIP图片”路径 (如果你的多模态只用输入主路径，请自行调整)
source_images = 'paper_images/images_ir' # 通常给IR路径或联合路径，取决于你的代码设定

# 2. 跑 Baseline (原版 CIoU)
print("🚀 正在生成 Baseline 预测图...")
model_base = YOLO('runs/detect/你的Baseline文件夹/weights/best.pt')  # 换成你的 CIoU 权重路径
model_base.predict(
    source=source_images, 
    save=True,               # 必须为 True，保存画好框的图
    project='Paper_Compare', # 输出的主文件夹
    name='Baseline_CIoU',    # 输出的子文件夹名
    conf=0.3,                # 置信度阈值 (建议0.3-0.4，太低框太乱，太高框不出来)
    line_width=2             # 框的粗细，论文图建议设为 2 或 3
)

# 3. 跑 你的创新版 (NWD + IoU)
print("🚀 正在生成 NWD+IoU 预测图...")
model_ours = YOLO('runs/detect/你的NWD文件夹/weights/best.pt')  # 换成你的 NWD 权重路径
model_ours.predict(
    source=source_images, 
    save=True, 
    project='Paper_Compare', 
    name='Ours_NWD', 
    conf=0.3,                # ⚠️ 必须和上面 Baseline 保持完全一致！
    line_width=2
)

print("✅ 全部生成完毕！快去 Paper_Compare 文件夹里看图吧！")