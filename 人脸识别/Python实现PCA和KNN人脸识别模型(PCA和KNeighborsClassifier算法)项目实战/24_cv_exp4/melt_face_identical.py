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
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
names = os.listdir('data')

X = []    # 存放人脸数据灰度值
for i in names:
    img = Image.open('data//' + i)
    img = img.convert('L') # 灰度转换
    img = img.resize((32, 32))
    arr = np.array(img)
    X.append(arr.reshape(1, -1).flatten().tolist())
X = pd.DataFrame(X)

print('DataFrame数据框所有图片灰度值:')
print(X)
print('\n\n数据集形状:')
print(X.shape)

y = []
for i in names:
    img = Image.open('data//' + i)
    y.append(int(i.split('_')[0]))

print('\n\n目标变量列表:')
print(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建PCA
pca = PCA()
pca.fit(X_train)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)



# 选择解释方差比率累积达到95%时的主成分数量
# => 保留大部分原始数据信息的同时有效地降低数据维度
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print("\n最合适的PCA主成分数量:",n_components)
pca = PCA(n_components=n_components)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

def pca_explain():
    plt.plot(cumulative_variance_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance Ratio')
    plt.grid()
    plt.show()

pca_explain()  # 解释方差比率图，显示每个主成分的解释方差比率以及累积解释方差比率

"""    searching pca best parameter
# 定义PCA和KNN管道
pca = PCA(whiten=True)
pipe = Pipeline([('pca', pca),])
param_grid = {'pca__n_components': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140 ,150, 160, 170, 180, 190, 200]}
# 创建 GridSearchCV 对象
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
# 使用最佳参数重新进行PCA降维
best_n_components = grid_search.best_params_['pca__n_components']
pca = PCA(n_components=best_n_components, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
"""

print('\n降维后的训练集形状:')
print(X_train_pca.shape)
print('降维后的测试集形状:')
print(X_test_pca.shape)

# 特征向量绘制
V = pca.components_
print('\nV.shape:', V.shape)
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i, ax in enumerate(axes.flat):
    if i < len(V):  # 确保索引不超出数组长度
        ax.imshow(V[i, :].reshape(32, 32), cmap="gray")
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
    else:
        ax.axis('off')  # 关闭多余的子图
plt.show()

# 构建KNN
# 创建GridSearchCV对象执行网格搜索
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

best_knn_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_predict = best_knn_model.predict(X_test_pca)
print("\n最佳KNN参数:", best_params)


# 模型评估
print('准确率分值: {0:0.4f}'.format(accuracy_score(y_test, y_predict)))
print("查准率 :", round(precision_score(y_test, y_predict, average='weighted'), 4))
print("召回率 :", round(recall_score(y_test, y_predict, average='weighted'), 4))
print("F1分值:", round(f1_score(y_test, y_predict, average='weighted'), 4))

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
