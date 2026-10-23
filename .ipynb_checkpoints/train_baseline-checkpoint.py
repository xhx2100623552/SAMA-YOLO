import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 解决多线程报错的常规设置
    os.environ["NUMEXPR_MAX_THREADS"] = "8"
    os.environ["OMP_NUM_THREADS"] = "8"

    print("🚀 正在启动 VEDAI 纯可见光(RGB)基线测试...")

    # 1. 加载官方原版模型与 COCO 预训练权重
    # 这里不要用你写的任何 YAML，直接传字符串 'yolo11n.pt'
    # Ultralytics 会自动去下载/加载带有先验知识的权重文件
    model = YOLO('yolo11n.pt') 

    # 2. 启动纯净版训练
    model.train(
        # 数据集配置
        data='/root/autodl-tmp/VEDAI_YOLO_Format/vedai.yaml',       # ⚠️ 确保这里的路径只指向可见光(RGB)的图片！
        
        # 基础训练参数
        epochs=100,              # 基线测 100 轮足够看出趋势了
        imgsz=1024,              # 保持你之前用的 1024 高分辨率
        batch=16,                # 原版模型非常轻，16 显存绝对够
        
        # 优化器与学习率 (使用官方默认的 SGD 跑基线最稳)
        optimizer='auto',        
        lr0=0.01,                # 有预训练权重的情况下，0.01 是最佳初始值
        pretrained=True,         # 强制启用预训练权重 (关键！)
        
        # 硬件与保存记录
        device=0,
        workers=8,
        project='VEDAI_Baseline_Project',
        name='YOLO11n_RGB_Only',
        save=True,
        plots=True
    )

    print("✅ 基线训练任务已启动！")