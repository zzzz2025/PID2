import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==========================================
# 1. 路径配置 (请根据你的实际路径修改)
# ==========================================
# 输入文件夹：你的 KAIST 训练集 RGB 图像路径
INPUT_DIR = "/root/data2/PID/dataset/KAIST/visible"
# 输出文件夹：我们新增的 Mask 存放路径
OUTPUT_DIR = "/root/data2/PID/dataset/KAIST/train_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 物理材质类别映射字典 (核心逻辑)
# ==========================================
# Cityscapes 预训练模型会输出 0~18 共 19 个类别。
# 我们需要把它们合并成我们需要的 5 大类物理材质，以便后续映射发射率(Emissivity)
# 目标类别定义:
# 0: Sky (天空) -> 发射率 ~0.85
# 1: Road/Ground (道路/地面) -> 发射率 ~0.90
# 2: Vehicles/Metal (车辆/金属) -> 发射率 ~0.85
# 3: Vegetation (植被/树木) -> 发射率 ~0.98
# 4: Others (建筑/人/其他物体) -> 发射率 ~0.92

cityscapes_to_physics = {
    0: 1,  # road -> 道路
    1: 1,  # sidewalk -> 道路
    2: 4,  # building -> 其他
    3: 4,  # wall -> 其他
    4: 4,  # fence -> 其他
    5: 4,  # pole -> 其他
    6: 4,  # traffic light -> 其他
    7: 4,  # traffic sign -> 其他
    8: 3,  # vegetation -> 植被
    9: 1,  # terrain -> 道路/地面
    10: 0,  # sky -> 天空
    11: 4,  # person -> 其他
    12: 4,  # rider -> 其他
    13: 2,  # car -> 车辆
    14: 2,  # truck -> 车辆
    15: 2,  # bus -> 车辆
    16: 2,  # train -> 车辆
    17: 2,  # motorcycle -> 车辆
    18: 2  # bicycle -> 车辆
}

# 利用 numpy 向量化操作加速映射过程
mapping_array = np.zeros(19, dtype=np.uint8)
for city_id, physics_id in cityscapes_to_physics.items():
    mapping_array[city_id] = physics_id

# ==========================================
# 3. 加载预训练模型
# ==========================================
print("正在加载预训练的 SegFormer 模型 (轻量级 b0 版本)...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ==========================================
# 4. 开始批量推理并保存
# ==========================================
image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"找到 {len(image_files)} 张图片，开始提取材质 Mask...")

with torch.no_grad():
    for filename in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, filename)
        # out_path = os.path.join(OUTPUT_DIR, filename)  # 保持同名
        out_filename = os.path.splitext(filename)[0] + ".png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        # 已经处理过则跳过（支持断点续传）
        if os.path.exists(out_path):
            continue

        # 读取图像
        image = Image.open(img_path).convert("RGB")
        original_size = image.size[::-1]  # (height, width)

        # 预处理输入
        inputs = processor(images=image, return_tensors="pt").to(device)

        # 模型推理
        outputs = model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

        # 插值恢复到原图尺寸
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=original_size,  # 恢复成原始的 H, W
            mode="bilinear",
            align_corners=False
        )

        # 获取每个像素概率最大的类别索引 (0~18)
        pred_cityscapes = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # 将 Cityscapes 类别映射为我们自定义的 5 大物理材质类别 (0~4)
        pred_physics = mapping_array[pred_cityscapes]

        # 保存为单通道的灰度图
        mask_image = Image.fromarray(pred_physics.astype(np.uint8), mode="L")
        mask_image.save(out_path)

print(f"✅ Mask 提取完成！文件已保存在 {OUTPUT_DIR}")