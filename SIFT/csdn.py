# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from numpy import *
import os
from pcv.tools.imtools import get_imlist  # 导入原书的PCV模块
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
from pcv.localdescriptors import sift

filelist = get_imlist('data/')

# 输入的图片
im1f = '488.jpg'
im1 = array(Image.open(im1f))
sift.process_image(im1f, 'out_sift_1.txt')
l1, d1 = sift.read_features_from_file('out_sift_1.txt')

i = 0
num = [0] * len(filelist)  # 存放匹配值
for infile in filelist:  # 对文件夹下的每张图片进行如下操作
    im2 = array(Image.open(infile))
    sift.process_image(infile, 'out_sift_2.txt')
    l2, d2 = sift.read_features_from_file('out_sift_2.txt')
    matches = sift.match_twosided(d1, d2)
    num[i] = len(matches.nonzero()[0])
    i = i + 1
    print('{} matches'.format(num[i - 1]))  # 输出匹配值

i = 1
figure()
while i < 4:  # 循环三次，输出匹配最多的三张图片
    index = num.index(max(num))
    print(index, filelist[index])
    lena = mpimg.imread(filelist[index])  # 读取当前匹配最大值的图片
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    # lena.shape  # (512, 512, 3)
    subplot(1, 3, i)
    plt.imshow(lena)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    num[index] = 0  # 将当前最大值清零
    i = i + 1
show()

