import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# 读取图片
my_photo = cv2.imread('myphoto.jpg', 1)
# 获取高和宽 得到图片的大小
height, width = my_photo.shape[:2]

# 显示图片
cv2.imshow('show original my_photo', my_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----------------------------------------------------平移----------------------------------------------------

# tx向右 ty向上
tx, ty = 100, 150

# create the translation matrix using tx and ty, it is a NumPy array
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)

# 将旋转矩阵应用于图像
translated_image = cv2.warpAffine(src=my_photo, M=translation_matrix, dsize=(width, height))

# 显示图片
cv2.imshow('show my_photo traslation',translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------缩放--------------------------------------------------
# 计算比例
scale_up_x = 1024/width
scale_up_y = 768/height
# 原始比例0.6缩小
scale_down = 0.6

scaled_f_down = cv2.resize(my_photo, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)
scaled_f_up = cv2.resize(my_photo, None, fx=scale_up_x, fy=scale_up_y, interpolation=cv2.INTER_LINEAR)

# 显示图像
cv2.imshow('Scale down to 1024*768.', scaled_f_up)
cv2.waitKey()
cv2.imshow('Scale down', scaled_f_down)
cv2.waitKey()
cv2.destroyAllWindows()

# --------------------------------------------------翻转---------------------------------------------------
img_0 = cv2.flip(my_photo, 1)
img_1 = cv2.flip(my_photo, 0)
img_2 = cv2.flip(my_photo, -1)

cv2.imshow('flip horizontal', img_0)
cv2.waitKey(0)
cv2.imshow('flip vertical', img_1)
cv2.waitKey(0)
cv2.imshow('flip vertical + horizontal', img_2)
cv2.waitKey(0)

# --------------------------------------------------旋转---------------------------------------------------
# 图片中心
center = (width/2, height/2)
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)

# rotate the image using cv2.warpAffine
rotated_image = cv2.warpAffine(src=my_photo, M=rotate_matrix, dsize=(width, height))

cv2.imshow('Rotated image', rotated_image)
# wait indefinitely, press any key on keyboard to exit
cv2.waitKey(0)

# --------------------------------------------------缩略---------------------------------------------------
# 使用PIL
# 目标尺寸
target_width = 1024/2
target_height = 768/2
# 读取图片
image = Image.open('myphoto.jpg')
# 生成缩略图
image.thumbnail((target_width, target_height))

# 显示图像
image.show()
cv2.waitKey(0)
