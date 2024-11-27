import cv2 as cv
import numpy as np
import tensorflow_hub as hub 

## *************************** MODEL IMPORTING ************************************
model = hub.KerasLayer("https://www.kaggle.com/models/sayannath235/american-sign-language/TensorFlow2/american-sign-language/1")

# CHECK IF THE MODEL WAS IMPORTED PROPERLY
try:
    print("Model loaded successfully!")
    
    # Check the type of the model (should be a Keras Layer)
    print("Model type:", type(model))
    
except Exception as e:
    print(f"Error loading the model: {e}")

## ***************************** GLOBAL VARIABLES AND CLASS DEFINITION *************************************
classes = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q',
    18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y',
    26: 'Z', 27: 'del', 28: 'space', 29: 'nothing'
}

## *************************** FUNCTION CALLS -> private & helper functions ************************************
def rescale_frame(frame: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    """ Takes a frame and scales it by certain size (default 0.75). works for images, videos, and live videos"""
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

def changeRes(capture: cv.VideoCapture, width: int, height: int) -> None:
    """Only works for live videos (webcam)"""
    capture.set(3, width) # number is property. 3 is width, 4 is height
    capture.set(4, height)

def addText(frame: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int], scale: float = 1.0, thickness: int = 2) -> None:
    """Adds a text onto the frame/image in triplex font"""
    cv.putText(frame, text, origin, cv.FONT_HERSHEY_TRIPLEX, scale, colour, thickness)

def normalizePixels(frame: np.ndarray) -> np.ndarray:
    """ Takes a frame or image and normalizes the pizels (value between 0 to 1)"""
    return frame.astype(np.float32) / 255.0

def addExtraDimension(frame: np.ndarray) -> np.ndarray:
    """ Takes a frame or image and adds an extra dimension to represent batch size"""
    return np.expand_dims(frame, axis=0)

## ************************** GETTING IMAGE OR VIDEO ****************************************
def captureVideo() -> None:
    """ Capture webcam video """
    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("Error: could not open webcam")
        exit()
    
    changeRes(capture, 640, 480) # change resolution of camera

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # add "Hello" onto the image
        addText(frame, "Hello", (255, 255), (0, 255, 0))

        # display image as new window
        cv.imshow('Camera', frame)

        # keyboard binding (ms)-> 0 means it waits infinite amount of time for key to be pressed
        # cv.waitKey(0)

        # Wait for a key press, exit on 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close windows
    capture.release()
    cv.destroyAllWindows()


def printImage() -> None:
    """ prints out an image """
    # gets a path to image and returns a matrix of pixels
    try:
        imgPath = './testImages/'
        img = cv.imread(imgPath + "A.png")
        if img is None:
            print("Error: image not found")
            return 
    except Exception as e:
        print(f"Error loading the image: {e}")  

    # rescale image
    img = rescale_frame(img, 224, 224)
    img = normalizePixels(img) 
    img = addExtraDimension(img)

    try:
        # get prediction from model
        prediction = model(img)
        predictionKey = np.argmax(prediction.numpy())
        # Get the class name
        predictedClass = classes[predictionKey + 1]
        print("Prediction done by model on image by percentage: ", prediction)
        print("Prediction done by model final result: ", predictedClass)
    except Exception as e:
        print(f"Error during prediction: {e}")

    # display image as new window
    cv.imshow('A ASL', (img[0] * 255).astype(np.uint8))  # Convert back to 0-255 range and remove batch dimension
    cv.waitKey(0)
    cv.destroyAllWindows()

printImage()
# captureVideo()