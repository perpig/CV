#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""

pandas == 1.1.5
matplotlib == 3.3.4
seaborn == 0.11.1
scikit-learn == 0.24.1
Pillow  == 8.2.0

"""

# 读取人脸照片数据
import os  # 导入os库
import warnings
from PIL import Image  # 导入图像操作模块
import numpy as np  # 导入科学计算模块
import pandas as pd  # 导入数据处理库

# 忽略警告
warnings.filterwarnings("ignore")

names = os.listdir('data')  # 返回指定文件夹下的文件或文件夹的名称列表
print(names[0:5])  # 输出前5个文件的名称



# 读取和显示第一张图
img0 = Image.open('data//' + names[0])  # 读取第1张图片
img0.show()  # 显示该图片

# 人脸数据处理 - 特征变量提取
# 对上面读取的图片img0进行灰度转换，参数'L'指转 换成灰度格式的图像。在进行灰度处理后，图像的每个像素点的颜色就可以用0～255的数值表示，称为灰度值，
# 其中0代表黑色，255代表白色，（0，255）区间的数值则代表不同程度的灰色。这样便完成了将图像转换成数字的第一步，也是非常重要的一步。
img0 = img0.convert('L')
# 调整图像尺寸为32×32像素，从左上角第1个像素点到右下角最后一个像素点就共有1024个像素点，而每个像素点都对应着一个灰度值（0～255），
# 这样每张图片就都有了1024个特征，每个特征变量的值就是灰度值。这个就是之后构造特征变量搭建模型的基础。
img0 = img0.resize((32, 32))
# 将这1024个像素点的灰度值转换为一个二维数组，并赋给变量arr。其中每个数值都是图像中每个像素点的灰度值，例如，第1行第1个数133就是图像左上角的第1个像素点的灰度值。
arr = np.array(img0)
print('\n图片灰度值 二维数组')
print(arr)

# 此时的二维表格共有32行、32列，每个单元格中的数值就是该像素点的灰度值。
pd.DataFrame(arr)  # 构建DataFrame数据框
arr = arr.reshape(1, -1)  # 上面获得的32×32的二维数组还需要转换成1×1024格式才能用于数据建模
print('\nDataFrame数据框图片灰度值 转成1*1024')
print(arr)

# 因为总共有400张图片的灰度值需要处理，若将400个二维数组堆叠起来会形成三维数组，所以我们需要用flatten()函数将1×1024的二维数组降维成一维数组，
# 并用tolist()函数将其转换为列表，以便之后和其他图片的灰度值一起处理。
print('\n列表图片灰度值')
print(arr.flatten().tolist())

# 批量处理图片
X = []  # 构造一个空列表X用于存放每一张人脸图片的灰度值。
for i in names:  # 通过for循环遍历文件名列表，其中的names就是最开始获取的各张人脸图片的文件名列表。
    # 将每张图片的图像数据转换为灰度值。
    img = Image.open('data//' + i)  # 读取图片
    img = img.convert('L')  # 进行图片灰度转换
    img = img.resize((32, 32))  # 重设图片大小
    arr = np.array(img)  # 转为数组
    X.append(arr.reshape(1, -1).flatten().tolist())  # 第用append()函数将每张图片的灰度值添加到X列表中。

X = pd.DataFrame(X)  # 构建DataFrame数据框
print('************************输出DataFrame数据框所有图片灰度值**************************')
# 每一行数据为每张图片的每个像素点对应的灰度值。
print(X)
print('************************输出数据集形状**************************')
print(X.shape)

# 人脸数据处理 - 目标变量提取
print(int(names[0].split('_')[0]))  # 第一张图片的目标变量读取

y = []  # 定义目标变量空列表
for i in names:  # 循环进行读取
    img = Image.open('data//' + i)  # 读取图片
    y.append(int(i.split('_')[0]))  # 目标变量放入列表

print('************************目标变量列表**************************')
print(y)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split  # 导入数据拆分函数

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 进行数据集拆分

# PCA数据降维
from sklearn.decomposition import PCA  # 导入PCA模型

pca = PCA(n_components=100)  # 将PCA模型赋给变量pca，并设置模型的参数n_components为100，即将这1024个特征进行线性组合，生成互不相关的100个新特征
pca.fit(X_train)  # 使用训练集的特征数据拟合PCA模型。

X_train_pca = pca.transform(X_train)  # 使用拟合好的PCA模型对训练集的特征数据进行降维
X_test_pca = pca.transform(X_test)  # 使用拟合好的PCA模型对测试集的特征数据进行降维

print('************************降维后的训练集形状**************************')
print(X_train_pca.shape)
print('************************降维后的测试集形状**************************')
print(X_test_pca.shape)

print('************************降维后的训练集前5行**************************')
print(pd.DataFrame(X_train_pca).head())
print('************************降维后的测试集前5行**************************')
print(pd.DataFrame(X_test_pca).head())

# 构建PCA人脸识别模型
# 模型搭建
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN分类模型

knn = KNeighborsClassifier()  # 建立KNN模型
knn.fit(X_train_pca, y_train)  # 用降维后的训练集进行训练模型

# 模型预测
y_pred = knn.predict(X_test_pca)  # 用降维后的测试集进行测试

# 模型评估
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # 导入模型评估工具

print('PCA人脸识别模型-准确率分值: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print("PCA人脸识别模型-查准率 :", round(precision_score(y_test, y_pred, average='weighted'), 4), "\n")
print("PCA人脸识别模型-召回率 :", round(recall_score(y_test, y_pred, average='weighted'), 4), "\n")
print("PCA人脸识别模型-F1分值:", round(f1_score(y_test, y_pred, average='weighted'), 4), "\n")

# 查看是否过拟合
print('训练集score: {:.4f}'.format(knn.score(X_train_pca, y_train)))
print('测试集score: {:.4f}'.format(knn.score(X_test_pca, y_test)))

from sklearn.metrics import classification_report  # 导入分类报告工具

# 分类报告
print(classification_report(y_test, y_pred))
