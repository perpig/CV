import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取彩色图像home_color
image = cv2.imread("home_color.png")
gray = cv2.imread("home_color.png", 0)

cv2.imshow("image", image)
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. 画出灰度化图像home_gray的灰度直方图，并拼接原灰度图与结果图
# 3. 画出彩色home_color图像的直方图，并拼接原彩色图与结果图，且与上一问结果放在同一个窗口中显示
fig = plt.figure("gray and color",figsize=(10, 8))
ax = fig.add_subplot(221, projection='3d') # 添加子图
ax.set_title('home_gray') # 设置标题

plt.hist(gray.ravel(), 256, [0, 256])
plt.show()
# 绘制彩色图像的直方图
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.hist(gray.ravel(), 256, [0, 256])############
    plt.xlim([0, 256])

plt.show()

# 4. 画出ROI（感兴趣区域）的直方图
x, y, w, h = 50, 100, 50, 100  # ROI的坐标和大小
roi = image[y:y+h, x:x+w]
mask = np.zeros_like(image)
cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)  # 创建ROI的掩码
masked_image = cv2.bitwise_and(image, mask)

# 显示原图和ROI区域
cv2.imshow('Original Image', image); cv2.waitKey(0)
cv2.imshow('ROI', roi); cv2.waitKey(0)
cv2.imshow('Mask', mask); cv2.waitKey(0)
cv2.imshow('ROI Extracted', masked_image); cv2.waitKey(0)
cv2.destroyAllWindows()
# 绘制ROI的直方图
plt.figure()
plt.hist(masked_image.ravel(), 256, [0, 256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
