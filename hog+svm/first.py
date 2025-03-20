'''
1、在数据集INRIADATA上，使用hog+svm实现行人检测
2、模型参数调优，提升检测效果
3、画出roc曲线
'''

# 用normalized_images目录下的图片做训练，或者用original_images目录下的图片+annotations获取行人区域做训练；测试则都在original_images/test/pos上测试。

# encoding: UTF-8
# 文件: hog.py
# 描述: 提取图片的HOG特征

from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt

def extract_hog_feature(filename):
    # 提取filename文件的HOG特征
    image = imread(filename, as_gray=True)
    # 读取图片，as_gray=True表示读取成灰度图
    feature,visimg = hog( # 提取HOG特征
        image, # 图片
        orientations=9, # 方向的个数，即bin的个数B
        pixels_per_cell=(8, 8), # 格子的大小，C×C
        cells_per_block=(2, 2), # 一块有2×2个格子
        block_norm='L2-Hys', # 归一化方法
        visualize=True # 是否返回可视化图像
    )
    plt.imshow(visimg)
    plt.show()
    return feature

if __name__ == '__main__':
    feature = extract_hog_feature('hog_test.png')
    print(feature) # 显示HOG特征
    print(feature.shape) # 显示HOG特征的维数

