# import sys
# print(sys.executable)
import cv2 as cv
import numpy as np

## *************************** FUNCTION CALLS ************************************
def rescale_frame(frame: np.ndarray, scale: float = 0.75) -> None:
    """ Takes a frame and scales it by certain size (default 0.75). works for images, videos, and live videos"""
    width = int(frame.shape[1] * scale) # width of image
    height = int(frame.shape[0] * scale) # height of image
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)

def changeRes(capture: cv.VideoCapture, width: int, height: int) -> None:
    """Only works for live videos (webcam)"""
    capture.set(3, width) # number is property. 3 is width, 4 is height
    capture.set(4, height)

def addText(frame: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int], scale: float = 1.0, thickness: int = 2) -> None:
    """Adds a text onto the frame/image in triplex font"""
    cv.putText(frame, text, origin, cv.FONT_HERSHEY_TRIPLEX, scale, colour, thickness)

## ************************** OTHER STUFF ****************************************

# Function to add text to the image
def addText(frame, text, position, color):
    cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)


# hello

def captureVideo() -> None:
    """ Capture webcam video """
    capture = cv.VideoCapture(0)
    # Start video capture
    capture = cv.VideoCapture(0)

    # Load Haar cascade for face detection
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # HSV range for skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    if not capture.isOpened():
        print("Error: could not open webcam")
        exit()
    
    changeRes(capture, 640, 480)  # Change resolution of the camera

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Add "Hello" onto the image
        addText(frame, "Hello", (50, 50), (0, 255, 0))

        # Detect face and create a mask to exclude it
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create a mask for the face(s)
        face_mask = np.zeros_like(gray)
        for (x, y, w, h) in faces:
            cv.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
 
        # Convert to HSV color space and apply skin mask
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        skin_mask = cv.inRange(hsv, lower_skin, upper_skin)

        # Remove the face from the skin mask
        skin_mask = cv.bitwise_and(skin_mask, cv.bitwise_not(face_mask))

        # Apply Gaussian blur and morphological operations
        blurred = cv.GaussianBlur(skin_mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv.morphologyEx(blurred, cv.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv.findContours(skin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter by contour area
            if cv.contourArea(contour) > 1000:
                # Get bounding box for the contour
                x, y, w, h = cv.boundingRect(contour)

                # Draw green rectangle around the hand
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display debugging windows
        cv.imshow('Skin Mask', skin_mask)
        cv.imshow('Hand Detection', frame)

        # Wait for a key press, exit on 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the capture object and close windows
    capture.release()
    cv.destroyAllWindows()


def printImage() -> None:
    """ prints out an image """
    # gets a path to image and returns a matrix of pixels
    img = cv.imread("./nekosan.jpg")

    # rescale image
    img = rescale_frame(img, 0.25) # try to change resolution instead

    # add "Hello" onto the image
    addText(img, "Hello", (255, 255), (0, 255, 0))

    # display image as new window
    cv.imshow('Neko', img)

    cv.waitKey(0)

# printImage()
captureVideo()