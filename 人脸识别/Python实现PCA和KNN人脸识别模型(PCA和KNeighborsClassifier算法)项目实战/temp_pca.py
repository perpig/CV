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
print("最合适的PCA主成分数量:",n_components)
# 使用选定的主成分数量重新运行PCA模型
pca = PCA(n_components=n_components)
pca.fit(X_train)

# 使用拟合好的PCA模型对数据集的特征数据进行降维
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print('\n\n降维后的训练集形状:')
print(X_train_pca.shape)
print('\n\n降维后的测试集形状:')
print(X_test_pca.shape)

# 特征向量
V = pca.components_
print('V.shape:',V.shape)
# 创建画布和子图对象
fig, axes = plt.subplots(10,10,figsize = (15,15))#,subplot_kw={"ticks":[],"yticks":[]})#不要显示坐标轴
#填充图像
for i,ax in enumerate(axes.flat):
    if i < len(V):  # 确保索引不超出数组长度
        ax.imshow(V[i,:].reshape(32,32),cmap = "gray")
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
    else:
        ax.axis('off')  # 关闭多余的子图

plt.show() #显示图像

# 构建KNN模型并进行参数调优
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

best_knn_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 使用最佳模型进行预测
y_pred = best_knn_model.predict(X_test_pca)

# 输出最佳参数
print("最佳KNN参数:", best_params)

# 模型评估
print('准确率分值: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print("查准率 :", round(precision_score(y_test, y_pred, average='weighted'), 4))
print("召回率 :", round(recall_score(y_test, y_pred, average='weighted'), 4))
print("F1分值:", round(f1_score(y_test, y_pred, average='weighted'), 4))

# 分类报告
print('分类报告：')
print(classification_report(y_test, y_pred))
