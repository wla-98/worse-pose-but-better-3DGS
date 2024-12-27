import os
import cv2
import numpy as np

# 指定文件夹路径
folder_path = r"C:\Users\admin\Desktop\graduationThesis\family"  # 替换为实际文件夹路径
# 配置参数
output_resolution = (960, 540)  # 图像统一分辨率 (宽, 高)
roi_start = (500, 200)  # 红框左上角点
roi_end = (800, 400)  # 红框右下角点
line_color = (0, 0, 255)  # 红色 (B, G, R)
resize_factor = 0.5  # 最终图像缩小比例

# 获取文件夹中所有图片路径（支持常见格式如 jpg, png 等）
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# 按文件名排序（可选，根据需要调整排序规则）
image_paths.sort()

# 确保读取到的图片数量满足要求
if len(image_paths) == 0:
    raise ValueError("指定文件夹中没有找到图片，请检查路径或文件格式。")

# 读取并调整图像分辨率
images = [cv2.resize(cv2.imread(path), output_resolution) for path in image_paths]

# 检查是否成功加载所有图片
if any(img is None for img in images):
    raise ValueError("加载图片失败，请检查图片文件是否有效。")

# 确保加载图片数量正确
num_images = len(images)
if num_images < 6:
    raise ValueError("需要至少6张图片，但当前文件夹中只有 {} 张图片。".format(num_images))

# 使用前6张图片
images = images[:6]

# 连接成一行
first_row = np.hstack(images)

# 创建放大区域和连接线
second_row = []
roi_width = roi_end[0] - roi_start[0]
roi_height = roi_end[1] - roi_start[1]

for i, img in enumerate(images):
    # 提取ROI
    roi = img[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]  # 指定红框区域
    roi_resized = cv2.resize(roi, output_resolution)  # 放大到统一大小
    second_row.append(roi_resized)

    # 在第一行中绘制红框
    start_point = (i * output_resolution[0] + roi_start[0], roi_start[1])
    end_point = (i * output_resolution[0] + roi_end[0], roi_end[1])
    cv2.rectangle(first_row, start_point, end_point, line_color, 2)

    # 计算连接线的顶点
    bottom_left = (start_point[0], end_point[1])  # 红框左下角
    bottom_right = (end_point[0], end_point[1])  # 红框右下角
    top_left = (i * output_resolution[0], output_resolution[1])  # 对应放大图片的左上角
    top_right = (i * output_resolution[0] + output_resolution[0], output_resolution[1])  # 对应放大图片的右上角

    # 在第一行直接绘制连接线，确保颜色一致
    cv2.line(first_row, bottom_left, top_left, line_color, thickness=2)
    cv2.line(first_row, bottom_right, top_right, line_color, thickness=2)

# 拼接第二行
second_row = np.hstack(second_row)

# 合并两行
final_image = np.vstack([first_row, second_row])

# 缩小最终图像
final_image_resized = cv2.resize(final_image, (int(final_image.shape[1] * resize_factor), int(final_image.shape[0] * resize_factor)))

# 保存结果
output_path = "output.jpg"
cv2.imwrite(output_path, final_image_resized)
cv2.imshow("Result", final_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
