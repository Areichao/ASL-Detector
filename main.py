import cv2 as cv
import numpy as np

## *************************** FUNCTION CALLS ************************************
def rescale_frame(frame: np.ndarray, scale: float = 0.75) -> None:
    """ Takes a frame and scales it by certain size (default 0.75). works for images, videos, and live videos"""
    width = int(frame.shape[1] * scale) # width of image
    height = int(frame.shape[0] * scale) # height of image
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)

def changeRes(width: int, height: int) -> None:
    """Only works for live videos (webcam)"""
    capture.set(3, width) # number is property. 3 is width, 4 is height
    capture.set(4, height)

def addText(frame: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int], scale: float = 1.0, thickness: int = 2) -> None:
    """Adds a text onto the frame/image in triplex font"""
    cv.putText(frame, text, origin, cv.FONT_HERSHEY_TRIPLEX, scale, colour, thickness)

## ************************** OTHER STUFF ****************************************
# gets a path to image and returns a matrix of pixels
# img = cv.imread("./nekosan.jpg")
capture = cv.VideoCapture(0)

if not capture.isOpened():
    print("Error: could not open webcam")
    exit()

changeRes(500, 500)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # rescale image
    # frames = rescale_frame(frame, 0.5)

    # add "Hello" onto the image
    addText(frame, "Hello", (255, 255), (0, 255, 0))

    # display image as new window
    cv.imshow('Neko', frame)

    # keyboard binding (ms)-> 0 means it waits infinite amount of time for key to be pressed
    # cv.waitKey(0)

    # Wait for a key press, exit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close windows
capture.release()
cv.destroyAllWindows()