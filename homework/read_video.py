import cv2

# Create a video capture object, in this case we are reading the video from a file
vid = cv2.VideoCapture('Waymo.mp4')

if (vid.isOpened() == False):
    print("Error opening the video file")

while (vid.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is frame
    ret, frame = vid.read()
    if ret == True:
        cv2.imshow('Waymo', frame)
        # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(20)

        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
vid.release()
cv2.destroyAllWindows()



