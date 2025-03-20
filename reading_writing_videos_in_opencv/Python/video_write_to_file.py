# Import packages
import cv2

# Create a video capture object
vid_capture = cv2.VideoCapture('Resources/Cars.mp4')

# Print error message if object is not in open state
if(vid_capture.isOpened() == False):
	print("Error opening video file")

# Get height and width of the frame
#CAP_PROP_FRAME_WIDTH = 3, CAP_PROP_FRAME_HEIGHT = 4
frame_width = int(vid_capture.get(3))
frame_height = int(vid_capture.get(4))
frame_size = (frame_width,frame_height)
fps = 20

# Create a video writer object
# 在 cv2.VideoWriter() 中，apiPreference 参数是用于指定视频编码的后端标识符。
# 这个参数通常不需要明确指定，因为 OpenCV 会自动选择合适的后端。但在某些情况下，您可能需要明确指定特定的后端标识符。

# 视频编码格式指的是将视频数据压缩和编码的方式。它决定了视频文件的存储格式以及解码后的播放方式。
# 视频编码格式通常由压缩算法和容器格式组成。

# 例如，常用的一些视频编码格式及其对应的四字符代码如下：
# H.264 编码器的四字符代码：'H264'
# XVID 编码器的四字符代码：'XVID'
# MPEG-4 编码器的四字符代码：'MP4V'
# Motion JPEG 编码器的四字符代码：'MJPG'
# Windows Media Video 编码器的四字符代码：'WMV1'、'WMV2'、'WMV3' 等

# cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('Resources/output_video_from_file.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

while(vid_capture.isOpened()):
	# vCapture.read() methods returns a tuple, first element is a bool 
	# and the second is frame
	ret, frame = vid_capture.read()

	if ret == True:
		output.write(frame)

		cv2.imshow("Frame",frame)
		# k == 113 is ASCII code for q key. You can try to replace that 
		# with any key with its corresponding ASCII code, try 27 which is for ESCAPE
		key = cv2.waitKey(20)
		if key == ord('q'):
			break
	else:
		print('Stream disconnected')
		break
# Release the video capture and output objects.
vid_capture.release()
output.release()
cv2.destroyAllWindows()