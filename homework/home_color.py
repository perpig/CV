import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1.读取彩色图像home_color
image = cv2.imread("home_color.png")
gray = cv2.imread("home_color.png",0)

cv2.imshow("image",image); cv2.waitKey(0)
cv2.imshow("gray",gray); cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.画出灰度化图像home_gray的灰度直方图，并拼接原灰度图与结果图
# 3.画出彩色home_color图像的直方图，并拼接原彩色图与结果图，且与上一问结果放在同一个窗口中显示
# 第一行左边放灰度直方图 右边放灰度原图 第二行画RGB直方图和原图 以子图形式呈现 全部放在一个窗口里
# 创建画布
fig = plt.figure("Gray and Color",figsize=(10, 8))

ax1 = fig.add_subplot(221) # 添加子图
plt.hist(gray.ravel(),256,[0,256],color='gray')
ax1.set_title('Gray Histogram') # 设置标题

ax2 = fig.add_subplot(222) # 添加子图
ax2.imshow(gray, cmap='gray')
ax2.set_title('Gray Image') # 设置标题

ax3 = fig.add_subplot(223) # 添加子图
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    ax3.plot(hist, color = col)
ax3.set_title('Color Histogram') # 设置标题

ax4 = fig.add_subplot(224) # 添加子图
ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # color原图
ax4.set_title('Color Image') # 设置标题

plt.show()

# 4.画出ROI（感兴趣区域 ）的直方图，ROI区域为 x：50-100，y：100-200
# 将原图home_color，ROI的mask图，ROI提取后的图及其直方图放在一个窗口内显示。
# 生成一个mask
mask = np.zeros(gray.shape[:2], np.uint8) # gray.shape取出长宽
mask[50:100, 100:200] = 255 # 225为白色

masked_img = cv2.bitwise_and(gray,gray,mask = mask)
# 计算掩码区域和非掩码区域的直方图
# 检查作为掩码的mask参数
hist_full = cv2.calcHist([gray],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([gray],[0],mask,[256],[0,256])

# 创建画布
fig2 = plt.figure("masked_image and hist",figsize=(10, 8))
# 分别生成子图
ax1 = fig2.add_subplot(221) # 添加子图
ax1.imshow(gray, 'gray') # 原图
ax1.set_title('gray image') # 设置标题

ax2 = fig2.add_subplot(222) # 添加子图
ax2.imshow(mask,'gray')  # 遮掩图
ax2.set_title('mark') # 设置标题

ax3 = fig2.add_subplot(223) # 添加子图
ax3.imshow(masked_img, 'gray') # marked_image
ax3.set_title('marked image') # 设置标题

ax4 = fig2.add_subplot(224) # 添加子图
ax4.plot(hist_full), plt.plot(hist_mask) # hist
ax4.set_title('hists') # 设置标题

plt.xlim([0,256])
plt.show()

