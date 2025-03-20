from PIL import Image
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# 提取图像的局部特征
def extract_local_features(image_path):
    image = Image.open(image_path)
    # 这里使用你选择的局部特征提取算法，比如SIFT
    keypoints, descriptors = your_local_feature_extraction_algorithm(image)
    return keypoints, descriptors

# 构建视觉词典
def build_visual_vocabulary(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    visual_vocabulary = kmeans.cluster_centers_
    return visual_vocabulary

# 计算图像的词袋表示
def compute_bovw_representation(image_descriptors, visual_vocabulary):
    bovw_representation = np.zeros(len(visual_vocabulary))
    nearest_neighbor = NearestNeighbors(n_neighbors=1)
    nearest_neighbor.fit(visual_vocabulary)
    distances, indices = nearest_neighbor.kneighbors(image_descriptors)
    for index in indices:
        bovw_representation[index] += 1
    return bovw_representation

# 计算两个词袋表示之间的相似度
def compute_similarity(bovw1, bovw2):
    similarity = np.dot(bovw1, bovw2) / (np.linalg.norm(bovw1) * np.linalg.norm(bovw2))
    return similarity
# 构建视觉词典
def build_visual_vocabulary(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    visual_vocabulary = kmeans.cluster_centers_
    return visual_vocabulary

# 计算图像的词袋表示
def compute_bovw_representation(image_descriptors, visual_vocabulary):
    bovw_representation = np.zeros(len(visual_vocabulary))
    nearest_neighbor = NearestNeighbors(n_neighbors=1)
    nearest_neighbor.fit(visual_vocabulary)
    distances, indices = nearest_neighbor.kneighbors(image_descriptors)
    for index in indices:
        bovw_representation[index] += 1
    return bovw_representation

# 计算两个词袋表示之间的相似度
def compute_similarity(bovw1, bovw2):
    similarity = np.dot(bovw1, bovw2) / (np.linalg.norm(bovw1) * np.linalg.norm(bovw2))
    return similarity
# 在数据集中找到与输入图像最相似的图像
def find_similar_images(query_image_path, dataset_image_paths, visual_vocabulary, num_similar_images=3):
    query_keypoints, query_descriptors = extract_local_features(query_image_path)
    query_bovw = compute_bovw_representation(query_descriptors, visual_vocabulary)
    similar_images = []
    for image_path in dataset_image_paths:
        keypoints, descriptors = extract_local_features(image_path)
        bovw = compute_bovw_representation(descriptors, visual_vocabulary)
        similarity = compute_similarity(query_bovw, bovw)
        similar_images.append((image_path, similarity))
    similar_images.sort(key=lambda x: x[1], reverse=True)
    return similar_images[:num_similar_images]

# 示例代码
def main():
    # 设置数据集路径和查询图像路径
    dataset_folder = 'data/'
    query_image_path = '0000.jpg'

    # 获取数据集图片列表
    dataset_image_paths = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]

    # 提取所有数据集图片的局部特征
    all_features = []
    for image_path in dataset_image_paths:
        _, descriptors = extract_local_features(image_path)
        all_features.extend(descriptors)

    # 构建视觉词典
    visual_vocabulary = build_visual_vocabulary(all_features, num_clusters=100)

    # 在数据集中寻找与查询图像最相似的图像
    similar_images = find_similar_images(query_image_path, dataset_image_paths, visual_vocabulary)

    # 打印最相似的图像路径和相似度
    for image_path, similarity in similar_images:
        print("Similar image:", image_path)
        print("Similarity:", similarity)

if __name__ == "__main__":
    main()
