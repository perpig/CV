import cv2
import numpy as np
import os
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

[x, y, w, h] = [0, 0, 0, 0]

# 加载人脸分类器
face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

folder_path = 'photo'
for filename in os.listdir(folder_path):

    if filename.endswith(('.jpg', '.jpeg', '.png','JPG')):
        # 读取图像
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)

        # 将图像转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 直方图均衡
        gray = cv2.equalizeHist(gray)

        size = frame.shape[:2]
        image = np.zeros(size, dtype=np.float32)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    # image = cv2.equalizeHist(image)
        im_h, im_w = size
        minSize_1 = (im_w // 10, im_h // 10)
        faceRects = face_Cascade.detectMultiScale(gray, 1.05, 2, cv2.CASCADE_SCALE_IMAGE, minSize_1)
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x + w, y + h), [255, 255, 0], 2)

        cv2.imshow("detection", frame)
        cv2.waitKey(0)
        print('okay')
