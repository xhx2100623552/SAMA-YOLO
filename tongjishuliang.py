import os

def count_png_recursive(root_path):
    total_png = 0
    # os.walk 会递归遍历指定路径下的所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                total_png += 1
    return total_png

# 👇 请在这里修改成你的数据集根目录路径
dataset_path = "/root/autodl-tmp/M3FD/M3FD_split/labels/test"  

if os.path.exists(dataset_path):
    total_count = count_png_recursive(dataset_path)
    print(f"📊 在路径 {dataset_path} 及其所有子目录下，总共找到 {total_count} 张 PNG 图片。")
else:
    print(f"❌ 找不到路径: {dataset_path}，请检查路径是否正确。")