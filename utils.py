import numpy as np

TILE_SIZE = 128  # 定义每个图像块的尺寸
PADDING_SIZE = 21  # 定义边缘填充大小，约为块大小的四分之一

LEFT_EDGE = -2  # 左边缘块标志
TOP_EDGE = -1  # 顶部边缘块标志
MIDDLE = 0  # 中间块标志
RIGHT_EDGE = 1  # 右边缘块标志
BOTTOM_EDGE = 2  # 底部边缘块标志

# 定义一个函数用于从大图像中获取小图像块
def get_patches(img, patch_h=TILE_SIZE, patch_w=TILE_SIZE):
    # 计算步长，即不包含边缘填充的部分
    y_stride, x_stride = patch_h - (2 * PADDING_SIZE), patch_w - (2 * PADDING_SIZE)
    # 检查提取的图像块尺寸是否超过原图尺寸
    if (patch_h > img.shape[0]) or (patch_w > img.shape[1]):
        print("Invalid cropping: Cropping dimensions larger than image shapes (%r x %r with %r)" % (
        patch_h, patch_w, img.shape))
        exit(1)

    locations, patches = [], []
    y = 0
    y_done = False
    while y <= img.shape[0] and not y_done:
        x = 0
        # 处理图像边缘情况，确保图像块完整
        if y + patch_h > img.shape[0]:
            y = img.shape[0] - patch_h
            y_done = True
        x_done = False
        while x <= img.shape[1] and not x_done:
            if x + patch_w > img.shape[1]:
                x = img.shape[1] - patch_w
                x_done = True
            # 保存图像块的位置信息和边缘类型
            locations.append(((y, x, y + patch_h, x + patch_w),
                              (y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
                              TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (img.shape[0] - patch_h) else MIDDLE),
                              LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (img.shape[1] - patch_w) else MIDDLE)))
            # 提取图像块
            patches.append(img[y:y + patch_h, x:x + patch_w, :])
            x += x_stride
        y += y_stride

    return locations, patches

# 定义一个函数用于将提取的小图像块拼接回原始图像尺寸
def stitch_together(locations, patches, size, patch_h=TILE_SIZE, patch_w=TILE_SIZE):
    output = np.zeros(size, dtype=np.float32)  # 初始化输出图像大小

    for location, patch in zip(locations, patches):
        outer_bounding_box, inner_bounding_box, y_type, x_type = location
        y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1

        # 根据块的边缘类型设置不同的粘贴参数
        if y_type == TOP_EDGE:
            y_cut = 0
            y_paste = 0
            height_paste = patch_h - PADDING_SIZE
        elif y_type == MIDDLE:
            y_cut = PADDING_SIZE
            y_paste = inner_bounding_box[0]
            height_paste = patch_h - 2 * PADDING_SIZE
        elif y_type == BOTTOM_EDGE:
            y_cut = PADDING_SIZE
            y_paste = inner_bounding_box[0]
            height_paste = patch_h - PADDING_SIZE

        if x_type == LEFT_EDGE:
            x_cut = 0
            x_paste = 0
            width_paste = patch_w - PADDING_SIZE
        elif x_type == MIDDLE:
            x_cut = PADDING_SIZE
            x_paste = inner_bounding_box[1]
            width_paste = patch_w - 2 * PADDING_SIZE
        elif x_type == RIGHT_EDGE:
            x_cut = PADDING_SIZE
            x_paste = inner_bounding_box[1]
            width_paste = patch_w - PADDING_SIZE

        # 在输出图像上粘贴图像块
        output[y_paste:y_paste + height_paste, x_paste:x_paste + width_paste] = patch[y_cut:y_cut + height_paste,
                                                                                x_cut:x_cut + width_paste]

    return output
