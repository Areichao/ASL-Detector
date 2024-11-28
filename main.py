import cv2 as cv
import numpy as np
import tensorflow_hub as hub 

## *************************** MAIN FUNCTION ************************************
def main() -> None:
    """ main function INIT and execution """

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

    ## **************************** APPLICATION CALL **********************************************
    printImage(model, classes)
# captureVideo()

## *************************** FUNCTION CALLS -> private & helper functions ************************************

def rescaleFrame(frame: np.ndarray, scale: float = 0.75) -> np.ndarray:
    """ Takes a frame and scales it by certain size (default 0.75). works for images, videos, and live videos"""
    width = int(frame.shape[1] * scale) # width of image
    height = int(frame.shape[0] * scale) # height of image
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)

# MIGHT NEED TO CHANGE THIS LATER INCASE VALUE IS NOT JUST A SQUARE
def modelFrameSize(frame: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    """ Takes a frame or image and scales it to be 224 by 224 (max) for the model to process """
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


def printImage(model: hub.KerasLayer, classes: dict) -> None:
    """ prints out an image and runs classification function on it"""
    # gets a path to image and returns a matrix of pixels
    try:
        imgPath = './testImages/'
        img = cv.imread(imgPath + "A.png")
        if img is None:
            print("Error: image not found")
            return 
    except Exception as e:
        print(f"Error loading the image: {e}")
        return   

    # rescale image -> create a frame version for just the model to test on
    imgModel = modelFrameSize(img, 224, 224) # change to 224 by 224 -> required by model
    imgModel = normalizePixels(imgModel) # normalize pixels (0 to 1 value)
    imgModel = addExtraDimension(imgModel) # add an extra dimension

    try:
        # get prediction from model -> using copy of image that is changed
        prediction = model(imgModel)
        predictionKey = np.argmax(prediction.numpy())
        predictedClass = classes[predictionKey + 1]
        print("Prediction done by model on image by percentage: ", prediction)
        print("Prediction done by model final result: ", predictedClass)

        # print original image
        img = rescaleFrame(img, scale=2.0)
        textCoordinates = (int(img.shape[1] * 0.05), int(img.shape[0] * 0.1))
        addText(img, predictedClass, textCoordinates, (0, 255, 0))
        # display image as new window
        cv.imshow('ASL', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
  
    except Exception as e:
        print(f"Error during prediction: {e}")


## ****************************** RUNNING MAIN FUNCTION ********************************
if __name__ == "__main__":
    main()