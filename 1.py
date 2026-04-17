import numpy as np
from PIL import Image

# 读取你觉得是“全黑”的那张图 (路径换成你实际的一张mask图片)
img_path = ("/root/data2/PID2/dataset/KAIST512/masks/set07_V001_I00039.png")
mask_img = Image.open(img_path)
mask_array = np.array(mask_img)

# 1. 打印里面包含的数字
print("这张图里包含的类别 ID 有:", np.unique(mask_array))
# 如果输出类似 [0, 1, 2, 3, 4]，说明分类完全成功！

# 2. 施展魔法：把它强行变亮给人类看
# 把 0~4 的数值乘以 50，拉伸到 0~200 的亮度范围
visible_array = (mask_array * 50).astype(np.uint8)
visible_img = Image.fromarray(visible_array)

# 保存一张可视化图看看
visible_img.save("visible_mask_test.png")
print("可视化版本已保存为 visible_mask_test.png，快打开看看吧！")