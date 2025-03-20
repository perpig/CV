from sklearn.datasets import fetch_lfw_people   # 使用lfw人脸库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
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

# 定义PCA和SVM管道
pca = PCA(whiten=True)
svm = SVC()

# 定义特征选择器
feature_selector = SelectKBest(f_classif)

# 定义管道
pipe = Pipeline([
    ('feature_selector', feature_selector),
    ('pca', pca),
    ('svm', svm)
])

# 定义参数空间
param_grid = {
    'feature_selector__k': [50, 100, 150],     # 选择的特征数量
    'pca__n_components': [50, 100, 150],       # PCA降维后的特征数量
    'svm__C': [0.1, 1, 10, 100],               # SVM正则化参数
    'svm__gamma': [0.001, 0.01, 0.1, 1]        # SVM核函数参数
}

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(pipe, param_grid, cv=5)

# 在训练数据上进行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数重新进行PCA降维和特征选择
best_k = grid_search.best_params_['feature_selector__k']
best_n_components = grid_search.best_params_['pca__n_components']

feature_selector = SelectKBest(f_classif, k=best_k)
pca = PCA(n_components=best_n_components, whiten=True)

X_train_selected = feature_selector.fit_transform(X_train, y_train)
X_train_pca = pca.fit_transform(X_train_selected)
X_test_selected = feature_selector.transform(X_test)
X_test_pca = pca.transform(X_test_selected)

# 使用最佳参数重新训练SVM分类器
best_C = grid_search.best_params_['svm__C']
best_gamma = grid_search.best_params_['svm__gamma']

svm = SVC(C=best_C, gamma=best_gamma)
svm.fit(X_train_pca, y_train)

# 预测
y_pred = svm.predict(X_test_pca)

# 输出分类报告
print(classification_report(y_test, y_pred, target_names=target_names))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Average Accuracy:", accuracy)
