import cv2
import numpy as np

# 读取视频文件
cap = cv2.VideoCapture('test.mov')

# 获取输入视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 初始化背景模型
background = None
alpha = 0.3  # 背景更新速率

# 输出视频设置，使用输入视频的帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mov', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if notimport cv2
import numpy as np

# 读取视频文件
cap = cv2.VideoCapture('test.mov')

# 获取输入视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 初始化背景模型
background = None
alpha = 0.3  # 背景更新速率

# 输出视频设置，使用输入视频的帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mov', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 初始化背景模型
    if background is None:
        background = gray.copy().astype(float)

    # 累加权重重构背景
    cv2.accumulateWeighted(gray, background, alpha)

    # 计算像素差
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))

    # 图像去噪
    blurred = cv2.medianBlur(diff, 5)
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 画出候选框
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('frame', frame)

    # 写入帧
    out.write(frame)

    # 等待一段时间，根据帧率设置延迟
    delay = int(1000 / fps)  # 计算延迟时间
    cv2.waitKey(delay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
 ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 初始化背景模型
    if background is None:
        background = gray.copy().astype(float)

    # 累加权重重构背景
    cv2.accumulateWeighted(gray, background, alpha)

    # 计算像素差
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))

    # 图像去噪
    blurred = cv2.medianBlur(diff, 5)
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 画出候选框
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('frame', frame)

    # 写入帧
    out.write(frame)

    # 等待一段时间，根据帧率设置延迟
    delay = int(1000 / fps)  # 计算延迟时间
    cv2.waitKey(delay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
