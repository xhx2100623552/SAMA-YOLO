# #这是提供 参考示例的 YOLOMM验证器使用方法
# #按必须按照你的需求来更改而不是盲目去使用
# from ultralytics import YOLOMM

# model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM12/weights/best.pt')
# model.val(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
#           split='test',device='1',
#         #   modality='x',模态消融参数 非必要不得开启
#           project='ResTest',
#           name='Val')
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 强制设为 1，解决 0 导致的崩溃
# 获取当前脚本所在目录并加入搜索路径的第一位
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from ultralytics import YOLOMM

# 1. 加载你刚跑完的最强权重
# 路径请核对好，如果是刚才的实验，应该是这个位置：
model = YOLOMM('/root/autodl-tmp/mutilmodel/ResTest/DroneVehicle_baseline/weights/best.pt')

# 2. 执行验证/测试
metrics = model.val(
    data='/root/autodl-tmp/M3FD/vip1/dronevehicle_vip1.yaml', # 你的数据集配置文件
    split='test',            # 关键！'val'是验证集，'test'是测试集，论文通常报 test 的分数
    batch=16,                # 验证时可以稍微大一点
    imgsz=640,               # 必须和训练时的分辨率保持一致
    device=0,                # 指定你的 GPU 编号
    project='ResTest/DroneVehicle_TEST',       # 保存到的项目名
    name='TEST_Baseline_vip1(baseline3)' # 保存到的文件夹名
)

# 3. 打印核心指标
print(f"测试集 mAP50: {metrics.box.map50:.4f}")
print(f"测试集 mAP50-95: {metrics.box.map:.4f}")