import cv2
import os

# 加载人脸分类器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

folder_path = 'photo'
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'JPG')):
        # 读取图像
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)

        # 将图像转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算积分图
        integral_image = cv2.integral(gray)

        size = frame.shape[:2]
        minSize_1 = (size[1] // 10, size[0] // 10)  # 计算最小尺寸
        face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2,
                                                   minSize=minSize_1)

        # 在图像上绘制检测到的人脸
        for (x, y, w, h) in face_rects:
            # 计算Haar特征值
            # Haar特征值 = A - B - C + D
            A = integral_image[y + h, x + w]
            B = integral_image[y, x + w]
            C = integral_image[y + h, x]
            D = integral_image[y, x]
            haar_value = A - B - C + D

            # 在图像上绘制检测到的人脸
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # 显示图像
        cv2.imshow('detection', frame)
        cv2.waitKey(0)

# 关闭窗口
cv2.destroyAllWindows()
