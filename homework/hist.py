# 编程实现直方图均衡化，给出测试效果
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取图像
image = cv2.imread("IMG_8297.jpeg", 0)  # 以灰度模式读取图像

# 对图像进行直方图均衡化
equ = cv2.equalizeHist(image)

# 将两个图拼在一起
res = np.hstack((image,equ))

# 显示
cv2.imshow("equalizeHist",res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 计算直方图
hist_image = cv2.calcHist([image],[0],None,[256],[0,256])
hist_equ = cv2.calcHist([equ],[0],None,[256],[0,256])

plt.figure("Hists")
plt.plot(hist_image, label='Original Image')
plt.plot(hist_equ, label='Equalized Image')
plt.legend()
plt.show()


