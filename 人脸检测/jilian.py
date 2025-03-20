import cv2
import os
import numpy as np

# 加载人脸分类器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 加载AdaBoost分类器
boost = cv2.ml.Boost_create()
y
# 提取Haar特征
def extract_haar_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        # 只取第一个检测到的人脸区域的特征
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        # 使用cv2.resize()函数将图像大小调整为指定大小
        roi = cv2.resize(roi, (24, 24))
        # 使用cv2.HOGDescriptor()对象的compute方法提取特征
        hog = cv2.HOGDescriptor((24, 24), (8, 8), (4, 4), (8, 8), 9)
        feature = hog.compute(roi)
        return True, feature.flatten()  # 将特征展平成一维数组
    else:
        # 如果没有检测到人脸，则返回一个填充的特征数组
        return False, np.zeros((576,))  # 24x24的灰度图像，特征向量大小为576

# 提取正样本和负样本的特征
pos_features = []
neg_features = []

# 准备正样本特征
pos_samples_folder = 'faces'
for filename in os.listdir(pos_samples_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'JPG')):
        image_path = os.path.join(pos_samples_folder, filename)
        image = cv2.imread(image_path)
        has_face, feature = extract_haar_features(image)
        if has_face:
            pos_features.append(feature)

# 准备负样本特征
neg_samples_folder = 'non_faces'
for filename in os.listdir(neg_samples_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'JPG')):
        image_path = os.path.join(neg_samples_folder, filename)
        image = cv2.imread(image_path)
        has_face, feature = extract_haar_features(image)
        if not has_face:
            neg_features.append(feature)

# 准备训练数据
features = np.vstack((np.vstack(pos_features), np.vstack(neg_features)))
labels = np.hstack((np.ones(len(pos_features)), np.zeros(len(neg_features))))

# 训练AdaBoost模型
boost.train(features, cv2.ml.ROW_SAMPLE, labels)

folder_path = 'photo'
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'JPG')):
        # 读取图像
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)

        size = frame.shape[:2]
        minSize_1 = (size[1] // 10, size[0] // 10)  # 计算最小尺寸
        face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=2,
                                                   minSize=minSize_1)

        # 在图像上绘制检测到的人脸
        for (x, y, w, h) in face_rects:
            # 提取当前人脸区域的特征
            roi = frame[y:y+h, x:x+w]
            has_face, feature = extract_haar_features(roi)
            # 使用AdaBoost模型进行预测
            _, result = boost.predict(feature.reshape(1, -1))
            if result == 1:  # 1 表示检测到人脸
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imshow('detection', frame)
        cv2.waitKey(0)

cv2.destroyAllWindows()
