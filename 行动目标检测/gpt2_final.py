import cv2
import numpy as np

# 读取视频文件
cap = cv2.VideoCapture('test.mov')

# 获取输入视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 初始化背景子tractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# 输出视频设置，使用输入视频的帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video2.mov', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用背景subtractor检测前景
    fgmask = fgbg.apply(frame)

    # 图像去噪
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # 画出前景的轮廓
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('frame', frame)

    # 写入帧
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
