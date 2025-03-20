import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# 读取图片
my_photo = cv2.imread('myphoto.jpg', 1)

# 显示图片
cv2.imshow('show my_photo', my_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 添加文本部分
photo_copy = my_photo.copy()
img_text = Image.fromarray(cv2.cvtColor(photo_copy, cv2.COLOR_BGR2RGB))  # 转换为PIL库可以处理的图片形式

# 设置字体和大小 这里用到的是mac自带的字体
font_path = "/System/Library/Fonts/STHeiti Medium.ttc"  # 替换为你的字体文件路径
font_size = 40
font = ImageFont.truetype(font_path, font_size)

# 要添加的文本
text = '22122128 孔馨怡'

# 文本位置
org = (50, 100)

# 设置文本颜色（红色）
color = (255, 0, 0)

# 创建Draw对象
draw = ImageDraw.Draw(img_text)

# 写入文本到图片上
draw.text(org, text, fill=color, font=font)

# 转换回OpenCV格式
photo_with_text = cv2.cvtColor(np.array(img_text), cv2.COLOR_RGB2BGR)

#保存图片
cv2.imwrite('photo_text.jpg', photo_with_text)

# 显示带有文本的图片
cv2.imshow("Image Text", photo_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()
