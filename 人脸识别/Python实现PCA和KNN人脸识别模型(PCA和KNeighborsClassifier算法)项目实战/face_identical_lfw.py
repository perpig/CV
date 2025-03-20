# face_identical_program
# cv_exp_4

from sklearn.datasets import fetch_lfw_people   # 使用lfw人脸库
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

lfw_dataset = fetch_lfw_people(min_faces_per_person=85, resize=0.4)
X = lfw_dataset.data
y = lfw_dataset.target
target_names = lfw_dataset.target_names
print("target names:", target_names)
print("分类个数:", len(target_names))

# 分割训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# PCA降维
n_components = 60
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train_pca, y_train)

y_pred = knn.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))

# 可视化分类结果
num_images_to_plot = min(15, len(X_test))
fig, axes = plt.subplots(3, 5, figsize=(10, 7), subplot_kw={'xticks': (), 'yticks': ()})
for i, (image, prediction) in enumerate(zip(X_test[:num_images_to_plot], y_pred[:num_images_to_plot])):
    ax = axes[i // 5, i % 5]
    ax.imshow(image.reshape(50, 37), cmap='gray')
    ax.set_title(target_names[prediction].split()[-1], color='black' if prediction == y_test[i] else 'red')
plt.show()

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Average Accuracy:", accuracy)

