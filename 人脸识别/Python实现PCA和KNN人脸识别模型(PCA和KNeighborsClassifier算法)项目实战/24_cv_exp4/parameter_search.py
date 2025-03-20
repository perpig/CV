from sklearn.datasets import fetch_lfw_people   # 使用lfw人脸库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 加载MIT人脸库数据集
lfw_dataset = fetch_lfw_people(min_faces_per_person=85, resize=0.4)

# 提取图像数据和标签
X = lfw_dataset.data
y = lfw_dataset.target
target_names = lfw_dataset.target_names
print("target names:", target_names)
print("分类个数:", len(target_names))

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 定义PCA和KNN管道
pca = PCA(whiten=True)
knn = KNeighborsClassifier()

# 定义管道
pipe = Pipeline([
    ('pca', pca),
    ('knn', knn)
])

# 定义参数空间
param_grid = {
    'pca__n_components': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140 ,150, 160, 170, 180, 190, 200],
    'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

# best_result: 6,70 with accuracy 0.693
# 6, 80, 100 with accuracy 0.70
# 6, 60, 85 with accuracy 0.75
# 3, 40, 150 with accuracy 0.875

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(pipe, param_grid, cv=5)

# 在训练数据上进行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数重新进行PCA降维
best_n_components = grid_search.best_params_['pca__n_components']
pca = PCA(n_components=best_n_components, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 使用最佳参数重新训练KNN分类器
best_n_neighbors = grid_search.best_params_['knn__n_neighbors']
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train_pca, y_train)

# 预测
y_pred = knn.predict(X_test_pca)

# 输出分类报告
print(classification_report(y_test, y_pred, target_names=target_names))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Average Accuracy:", accuracy)
