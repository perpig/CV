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

X = [] #空列表：存放每一张人脸图片的灰度值
for i in names:
    img = Image.open('data//' + i) # 读图
    img = img.convert('L') # 灰度转换
    img = img.resize((32, 32)) #调整大小
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

y = [] # 目标变量
for i in names:
    img = Image.open('data//' + i)
    y.append(int(i.split('_')[0]))  # 第一张图片的目标变量读取

print('\n\n目标变量列表:')
print(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# random_state 随机种子

pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
explained_variance_ratio = pca.explained_variance_ratio_
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio')
plt.grid()
plt.show()

# # 赋值PCA模型
# pca = PCA(n_components=100)
# # 参数意义：即将这1024个特征进行线性组合，生成互不相关的100个新特征
# pca.fit(X_train)  # 使用训练集的特征数据拟合PCA模型

# 使用拟合好的PCA模型对数据集的特征数据进行降维
X_train_pca = pca.transform(X_train)  # 训练集
X_test_pca = pca.transform(X_test)  # 测试集

print('\n\n降维后的训练集形状:')
print(X_train_pca.shape)
print('\n\n降维后的测试集形状:')
print(X_test_pca.shape)

# # 降维后的训练集前5行
# print(pd.DataFrame(X_train_pca).head())
# print(pd.DataFrame(X_test_pca).head())

V = pca.components_
print('V.shape:',V.shape)
# 创建画布和子图对象
fig, axes = plt.subplots(10, 10, figsize=(15, 15))  # 不要显示坐标轴

# 填充图像
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i, :].reshape(32, 32), cmap="gray")
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
plt.show()

# 构建PCA人脸识别模型 构建KNN
# 定义KNN模型的参数网格
param_grid = {
    'n_neighbors': [3, 5, 7],
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
y_pred = best_knn_model.predict(X_test_pca)

# # 模型搭建 建立KNN分类模型
# knn = KNeighborsClassifier()
# knn.fit(X_train_pca, y_train) # 用降维后的训练集进行训练模型
# # 模型预测
# y_pred = knn.predict(X_test_pca)  # 用降维后的测试集进行测试

# 输出最佳参数
print("最佳参数:", best_params)

# 模型评估
print('准确率分值: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print("查准率 :", round(precision_score(y_test, y_pred, average='weighted'), 4))
print("召回率 :", round(recall_score(y_test, y_pred, average='weighted'), 4))
print("F1分值:", round(f1_score(y_test, y_pred, average='weighted'), 4))

# 率分值：这是模型在测试集上的准确率，表示模型正确预测的样本比例。准确率越高，说明模型的预测结果与实际标签的匹配程度越好。
# 查准率（Precision）：查准率是指模型在预测为正类的样本中，真正为正类的比例。在多类别分类问题中，通过对每个类别计算查准率，然后取平均值（加权平均）得到加权查准率。
# 召回率（Recall）：召回率是指模型能够正确预测为正类的样本中，实际为正类的比例。在多类别分类问题中，通过对每个类别计算
