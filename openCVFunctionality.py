import cv2 as cv


## ***************************** RANDOM OPENCV FUNCTIONALITIES ************************
# WEBCAM IMPORTING
# 0 is default webcam, 1 first camera connected, 2 is second, etc.
capture = cv.VideoCapture(0)



# IMPORTED VIDEOS -> PROBABLY NEED TO FIX RESOLUTION
capture = cv.VideoCapture("Video") # replace Video with filepath

# read video frame by frame
while True:
    isTrue, frame = capture.read()
    cv.imshow("Video", frame)

    # ensures video is not played infinitely
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

# come here and destroy window after while
capture.release()
cv.destroyAllWindows() # might give an error if we run out of frames. or wrong path file.


# CONVERT AN IMAGE TO GREYSCALE