import torch
from ultralytics import YOLOMM  # 确保你的环境能导入这个类

# 1. 设置路径
baseline_path = '/root/autodl-tmp/mutilmodel/VEDAI_Baseline_Project/YOLO11n_RGB_Only/weights/best.pt'
new_yaml_path = '/root/autodl-tmp/mutilmodel/ultralytics/cfg/models/mm/YOLO11-MEAF-MCF-Hybrid.yaml' # 你的双流YAML
output_path = '/root/autodl-tmp/mutilmodel/VEDAI_Baseline_Project/YOLO11n_RGB_Only/seed_v3.pt'

# 2. 初始化全新的双流模型 (此时参数是随机初始化的)
print("🏗️ 正在构建双流模型架构...")
new_model = YOLOMM(new_yaml_path)
new_state_dict = new_model.model.state_dict()

# 3. 加载单模态基线权重
print(f"📦 正在加载基线权重: {baseline_path}")
baseline_ckpt = torch.load(baseline_path, map_location='cpu')
# 注意：有些版本的ckpt里模型在'model'键下，有些直接是字典
baseline_weights = baseline_ckpt['model'].state_dict() if 'model' in baseline_ckpt else baseline_ckpt

# 4. 精准搬砖逻辑
print("🚀 开始精准搬运骨干网(Backbone)权重...")
# 只搬运 0-10 层（这是通用的骨干网层）
for i in range(11):
    rgb_prefix = f'model.{i}.'
    ir_prefix = f'model.{i + 11}.' # 对应双流中的红外路径
    
    # 在基线中寻找这一层的权重
    keys = [k for k in baseline_weights.keys() if k.startswith(rgb_prefix)]
    
    for k in keys:
        # 搬给新模型的 RGB 分支 (0-10)
        if k in new_state_dict:
            new_state_dict[k] = baseline_weights[k]
        
        # 镜像搬给新模型的 IR 分支 (11-21)
        ir_key = k.replace(rgb_prefix, ir_prefix)
        if ir_key in new_state_dict:
            new_state_dict[ir_key] = baseline_weights[k]
            # print(f"  [Mirror] {k} -> {ir_key}")

# 5. 保存这个纯净的种子权重
# 我们保存整个 ckpt 结构，模仿官方格式
torch.save({'model': new_model.model, 'ema': None, 'updates': None, 'opt': None}, output_path)

print(f"✅ 种子权重已成功制作: {output_path}")
print("🎯 现在去 trainMM.py 里加载这个 seed_v2.pt 启动训练吧！")