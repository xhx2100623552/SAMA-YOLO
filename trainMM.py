import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 强制设为 1，解决 0 导致的崩溃
# 获取当前脚本所在目录并加入搜索路径的第一位
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from ultralytics import YOLOMM

if __name__ == '__main__':
    # 🚨 1. 核心修正：必须使用多模态的 YAML！
    model = YOLOMM('/root/autodl-tmp/mutilmodel/ultralytics/cfg/models/mm/Ablation/YOLO11-Baseline-BiFPN-LSCD.yaml') 
    # model.load('/root/autodl-tmp/mutilmodel/ResTest/DroneVehicle_TEST/DV_MEAF-MCF_LSCD_MHCA2/weights/epoch80.pt')
    model.train(
        # === 数据配置 ===
        data='/root/autodl-tmp/RealData/DroneVehicle/dronevehicle.yaml',
        # === 核心训练参数 ===
        epochs=300,         # 真正训练至少需要 100-200 轮，epochs=2 只是用来测试代码通不通的
        batch=16,           # 3090 有 24G 显存，跑 n 模型开 32 毫无压力，训练速度翻倍DV16,M3FD24
        imgsz=640,# 🎯 极度关键：DroneVehicle 预处理后的矩形尺寸 [高, 宽][512,640]
        # scale='n',          # 保持 n（Nano版）作为初期验证最合适
        optimizer='SGD',
        lr0=0.01,           # 🎯 保持 0.01 的高起步，给模型足够的“推力”0.0005
        lrf=0.1,            # 🎯 关键修改：最终倍率从 0.01 改为 0.1
        # optimizer='AdamW',
        # lr0=0.001,
        warmup_epochs=3.0,
        momentum=0.937,
        amp=True,
        # === 硬件与加速 ===
        device=0,           # 明确指定使用第一块 GPU
        workers=12,          # AutoDL 上通常给 8 或 12 个 CPU 线程加载数据最快DV8
        cache=True,         # 将图片缓存到内存，极大提升训练速度
        box=7.5,
        # === 训练策略 ===
        patience=50,        # 早停机制：如果连续 50 轮没进步，自动停止，防止过拟合
        loss26=True,        # 保持你原有的特定 Loss 开关
        close_mosaic=10,
        # === 日志与保存 ===
        project='ResTest/DroneVehicle_TEST',
        name='train_DV_Baseline_BiFPN_LSCD_loss(0.9nwd)',
        # save_period=10,
        exist_ok=True,
    )