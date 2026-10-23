# from ultralytics import YOLO

# def check_model_complexity(yaml_path):
#     print(f"\n========== 正在分析: {yaml_path} ==========")
    
#     # 仅加载模型结构，任务指定为 detect
#     model = YOLO(yaml_path, task='detect')
    
#     # 🚨 修复报错：去掉了 imgsz 参数，YOLO 默认按 640x640 计算
#     info = model.info(detailed=False)
    
#     # 提取并打印参数量和计算量
#     # info 通常返回一个元组: (layers, parameters, gradients, GFLOPs)
#     try:
#         params = info[1] / 1e6
#         flops = info[3]
#         print(f"✅ 总参数量 (Params): {params:.2f} M")
#         print(f"✅ 计算量 (GFLOPs): {flops:.1f} G")
#     except Exception as e:
#         print(f"⚠️ 解析 info 时出错: {e}，请直接查看上面 YOLO 打印出的日志信息。")

# if __name__ == "__main__":
#     # 建议先测你的 Baseline，再测纯 LSCD
#     # 请确保这里的 yaml 路径是你真实的路径
#     # yaml_1 = '/root/autodl-tmp/mutilmodel/ultralytics/cfg/models/mm/Ablation/YOLO11-Baseline.yaml'
#     yaml_2 = '/root/autodl-tmp/mutilmodel/ultralytics/cfg/models/mm/Ablation/YOLO11-LSCD-MRCA.yaml'
    
#     check_model_complexity(yaml_2)

import torch
from thop import profile
from ultralytics import YOLO

def test_macs_and_params(yaml_path):
    print(f"\n========== 开始底层物理测试: {yaml_path} ==========")
    
    # 1. 加载模型
    model = YOLO(yaml_path, task='detect')
    
    # 2. 剥离 Ultralytics 的外壳，获取纯 PyTorch 模型结构
    pt_model = model.model.eval()
    
    # 3. 构造虚拟输入 (Dummy Input)
    # 🚨 注意：如果是 RGBT 多模态双路输入早期融合，通常是 6 通道
    # (Batch_size=1, Channels=6, Height=640, Width=640)
    dummy_input = torch.randn(1, 6, 640, 640)
    
    # 4. 使用 thop 库进行严格的物理计算
    flops, params = profile(pt_model, inputs=(dummy_input, ), verbose=False)
    
    # 5. 换算单位并打印
    # 注意：thop 算出的是 MACs (乘加操作数)，通常 1 MAC = 2 FLOPs
    # YOLO 官方打印的 GFLOPs 实际上是 G-MACs * 2
    gflops = (flops * 2) / 1e9 
    params_m = params / 1e6
    
    print(f"✅ 总参数量 (Params): {params_m:.2f} M")
    print(f"✅ 计算量 (GFLOPs): {gflops:.2f} G")

if __name__ == "__main__":
    # 测试基线模型
    test_macs_and_params('/root/autodl-tmp/mutilmodel/ultralytics/cfg/models/mm/Ablation/YOLO11-LSCD-MRCA.yaml')