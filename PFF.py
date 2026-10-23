import torch
import time
from thop import profile

try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:
    pass

def test_yolo_yaml_efficiency(yaml_path, device='cuda:0'):
    print(f"1. 正在从 {yaml_path} 构建模型...")
    
    # 构建模型 (如果原本设定的 ch=3 会报错，请改成 ch=6)
    model = DetectionModel(cfg=yaml_path, ch=6, nc=6).to(device)
    model.eval()

    # 构造双模态假数据 (各自是 3 通道)
    dummy_rgb = torch.randn(1, 3, 640, 640).to(device)
    dummy_ir  = torch.randn(1, 3, 640, 640).to(device)

    # ==========================================
    # 🚨 核心修复：通道拼接！
    # 将 RGB 和 IR 在通道维度(dim=1)拼接，变成一个 6 通道的 Tensor
    # 形状从两个 (1, 3, 640, 640) 变成一个 (1, 6, 640, 640)
    # ==========================================
    packed_input = torch.cat([dummy_rgb, dummy_ir], dim=1)

    print("2. 正在计算 FLOPs 和 参数量 (Params)...")
    # 现在 packed_input 是一个正常的 Tensor，直接传进去
    macs, params = profile(model, inputs=(packed_input, ), verbose=False)

    gflops = (macs * 2) / 1e9
    params_m = params / 1e6

    print("-" * 40)
    print(f"📊 模型参数量 (Parameters): {params_m:.2f} M")
    print(f"📊 理论计算量 (GFLOPs)  : {gflops:.2f} GFLOPs")
    print("-" * 40)

    print("3. 正在测试推理速度 (GPU 预热中...)")
    for _ in range(50):
        _ = model(packed_input)

    iterations = 200
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(packed_input)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    fps = 1.0 / avg_time
    
    print(f"🚀 推理速度 (FPS): {fps:.2f} FPS (测试设备: {torch.cuda.get_device_name(device)})")
    print("-" * 40)

if __name__ == "__main__":
    # 请填入您 yaml 的真实路径
    yaml_file_path = '/root/autodl-tmp/mutilmodel/ultralytics/cfg/models/mm/Ablation/YOLO11-Baseline-BiFPN-LSCD.yaml' 
    test_yolo_yaml_efficiency(yaml_file_path)