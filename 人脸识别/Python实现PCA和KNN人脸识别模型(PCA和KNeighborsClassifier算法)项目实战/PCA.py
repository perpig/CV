import os
import warnings
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

names = os.listdir('data')  # 读取data文件夹下面的名称

# 空列表：存放每一张人脸图片的灰度值
X = []
for i in names:
    img = Image.open('data//' + i) # 读图
    img = img.convert('L') # 灰度转换
    img = img.resize((32, 32)) # 调整大小
    arr = np.array(img) # 转为数组
    X.append(arr.reshape(1, -1).flatten().tolist()) # 添加
    # arr.reshape(1, -1)：将原始的二维数组 arr 重排成一个新的形状为 (1, -1) 的二维数组。
    # 这里的 -1 表示这一维的大小由数组的大小和另一维的大小推断出来，即保持数组的原有形状。
    # .flatten()：将数组展平为一维数组。
    # .tolist()：转换为列表对象。

X = pd.DataFrame(X)  # 构建DataFrame数据框

print('DataFrame数据框所有图片灰度值:')
# 每一行数据为每张图片的每个像素点对应的灰度值。
print(X)
print('\n\n数据集形状:')
print(X.shape)

# 目标变量
y = []
for i in names:
    img = Image.open('data//' + i)
    y.append(int(i.split('_')[0]))  # 第一张图片的目标变量读取

print('\n\n目标变量列表:')
print(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# random_state 随机种子

# 运行PCA模型并绘制解释方差比率图
pca = PCA()
pca.fit(X_train)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.plot(cumulative_variance_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio')
plt.grid()
plt.show()

# 选择解释方差比率累积达到95%时的主成分数量
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print("\n最合适的PCA主成分数量:",n_components)
# 使用选定的主成分数量重新运行PCA模型
pca = PCA(n_components=n_components)
pca.fit(X_train)

# # 赋值PCA模型
# pca = PCA(n_components=100)
# # 参数意义：即将这1024个特征进行线性组合，生成互不相关的100个新特征
# pca.fit(X_train)  # 使用训练集的特征数据拟合PCA模型

# 使用拟合好的PCA模型对数据集的特征数据进行降维
X_train_pca = pca.transform(X_train)  # 训练集
X_test_pca = pca.transform(X_test)  # 测试集

print('\n降维后的训练集形状:')
print(X_train_pca.shape)
print('降维后的测试集形状:')
print(X_test_pca.shape)

# # 降维后的训练集前5行
# print(pd.DataFrame(X_train_pca).head())
# print(pd.DataFrame(X_test_pca).head())

# 特征向量
V = pca.components_
print('\nV.shape:', V.shape)
# 创建画布和子图对象
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
# 填充图像
for i, ax in enumerate(axes.flat):
    if i < len(V):  # 确保索引不超出数组长度
        ax.imshow(V[i, :].reshape(32, 32), cmap="gray")
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
    else:
        ax.axis('off')  # 关闭多余的子图
plt.show()

# 构建PCA人脸识别模型 构建KNN
# 定义KNN模型的参数网格
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

# 执行网格搜索
grid_search.fit(X_train_pca, y_train)

# 获取最佳模型和参数
best_knn_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 使用最佳模型进行预测
y_predict = best_knn_model.predict(X_test_pca)

# # 模型搭建 建立KNN分类模型
# knn = KNeighborsClassifier()
# knn.fit(X_train_pca, y_train) # 用降维后的训练集进行训练模型
# # 模型预测
# y_predict = knn.predict(X_test_pca)  # 用降维后的测试集进行测试

# 输出最佳参数
print("\n最佳KNN参数:", best_params)

# 模型评估
print('准确率分值: {0:0.4f}'.format(accuracy_score(y_test, y_predict)))
print("查准率 :", round(precision_score(y_test, y_predict, average='weighted'), 4))
print("召回率 :", round(recall_score(y_test, y_predict, average='weighted'), 4))
print("F1分值:", round(f1_score(y_test, y_predict, average='weighted'), 4))

# 率分值：这是模型在测试集上的准确率，表示模型正确预测的样本比例。准确率越高，说明模型的预测结果与实际标签的匹配程度越好。
# 查准率（Precision）：查准率是指模型在预测为正类的样本中，真正为正类的比例。在多类别分类问题中，通过对每个类别计算查准率，然后取平均值（加权平均）得到加权查准率。
# 召回率（Recall）：召回率是指模型能够正确预测为正类的样本中，实际为正类的比例。在多类别分类问题中，通过对每个类别计算召回率，然后取平均值（加权平均）得到加权召回率。
# F1分值：F1 分值是准确率和召回率的调和平均数，它综合考虑了模型的查准率和召回率。在多类别分类问题中，同样使用加权平均来计算加权 F1 分值。

# 获取分类个数
num_classes_train = len(set(y_train))
num_classes_test = len(set(y_test))

print("训练集中的分类个数:", num_classes_train)
print("测试集中的分类个数:", num_classes_test)


# 查看是否过拟合
print('训练集score: {:.4f}'.format(best_knn_model.score(X_train_pca, y_train)))
print('测试集score: {:.4f}'.format(best_knn_model.score(X_test_pca, y_test)))

# 分类报告
print('分类报告：')
print(classification_report(y_test, y_predict))

# 训练集大小：通过 train_test_split 函数中的 test_size 参数控制，这里设置为训练集占总数据集的 80%，即 320 张图片。
# 测试集大小：同样通过 train_test_split 函数中的 test_size 参数控制，设置为测试集占总数据集的 20%，即 80 张图片。
# 分类个数
# 降维维度: 由 PCA 模型中的 n_components 参数控制，这里设置为 100
# KNN 参数：在本实验中，KNN 模型的参数没有进行显式设置，使用了默认参数。
# KNN 模型中的主要参数包括邻居数量 n_neighbors、距离度量方法 metric 等，这些参数的选择可能会对模型的性能产生影响。
