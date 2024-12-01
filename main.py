import cv2 as cv
import numpy as np
import tensorflow_hub as hub 
import mediapipe as mp

## for plotting class probabilities
# import matplotlib.pyplot as plt 
# from io import BytesIO

## *************************** MAIN FUNCTION ************************************
def main() -> None:
    """ main function INIT and execution """

    ## *************************** MODEL IMPORTING ************************************
    try:
        model = hub.KerasLayer("https://www.kaggle.com/models/sayannath235/american-sign-language/TensorFlow2/american-sign-language/1")
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

    # mediahands initialization
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    ## **************************** APPLICATION CALL **********************************************
    # print_image(model, classes)
    capture_video(model, classes, mp_hands, hands, mp_draw)

## *************************** FUNCTION CALLS -> private & helper functions ************************************

def rescale_frame(frame: np.ndarray, scale: float = 0.75) -> np.ndarray:
    """ Scales a frame by a certain size (default 0.75) """
    width = int(frame.shape[1] * scale) # width of image
    height = int(frame.shape[0] * scale) # height of image
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)

# MIGHT NEED TO CHANGE THIS LATER INCASE VALUE IS NOT JUST A SQUARE
def model_frame_size(frame: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    """ Takes a frame or image and scales it to be 224 by 224 (max) for the model to process """
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

def change_resolution(capture: cv.VideoCapture, width: int, height: int) -> None:
    """ Changes resolution for live videos (webcam) """
    capture.set(3, width) # number is property. 3 is width, 4 is height
    capture.set(4, height)

def add_text_on_frame(frame: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int], scale: float = 1.0, thickness: int = 2) -> None:
    """ Adds text onto the frame/image in triplex font """
    cv.putText(frame, text, origin, cv.FONT_HERSHEY_TRIPLEX, scale, colour, thickness)

def normalize_pixels(frame: np.ndarray) -> np.ndarray:
    """ Takes a frame/image and normalizes the pixels (value between 0 to 1) """
    return frame.astype(np.float32) / 255.0

def add_extra_dimension(frame: np.ndarray) -> np.ndarray:
    """ Takes a frame/image and adds an extra dimension to represent batch size """
    return np.expand_dims(frame, axis=0)

def draw_rectangle(width: int, height: int) -> tuple[int, int, int, int]:
    """ Draws a rectangle where the model can read best from """
    box_size = 224  # The size of the box (fit for model)
    top_left_X = (width - box_size) // 2
    top_left_Y = (height - box_size) // 2
    bottom_right_X = top_left_X + box_size
    bottom_right_Y = top_left_Y + box_size

    return (top_left_X, top_left_Y, bottom_right_X, bottom_right_Y)

def filter_frame(frame: np.ndarray) -> np.ndarray:
    """ Applies Gaussian filtering to the frame """
    blurred_frame = cv.GaussianBlur(frame, (5, 5), 0)
    return blurred_frame

# def plot_probabilities(probabilities: np.ndarray, classes: dict):
#     """ Chart of class probabilities """
#     class_labels = list(classes.values())
   
#     plt.figure(figsize=(6, 6))
#     y_pos = np.arange(len(class_labels))
#     plt.barh(y_pos, probabilities, color='skyblue')
#     plt.yticks(y_pos, class_labels)
#     plt.xlabel('Probability (%)')
#     plt.title('Class Probabilities')
#     plt.tight_layout()
    
#     # convert plot to image
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     buf.close()
#     plt.close()
#     img = cv.imdecode(img_array, 1)
#     return img

## ************************** GETTING IMAGE OR VIDEO ****************************************
def capture_video(model: hub.KerasLayer, classes: dict, mp_hands: mp.solutions.hands, hands: mp.solutions.hands.Hands, mp_draw: mp.solutions.drawing_utils) -> None:
    """ Capture webcam video """
    try:
        capture = cv.VideoCapture(0)
        # error if video is not opened
        if not capture.isOpened():
            print("Error: could not open webcam")
            return 

        height = 480
        width = 640
        change_resolution(capture, width, height) # change resolution of camera
        padding = 10 # padding for box around hand

        # drawing static 224 by 224 green square on frame (for model input)
        static_top_left_X, static_top_left_Y, static_bottom_right_X, static_bottom_right_Y = draw_rectangle(width, height)

        while True:
            ret, frame = capture.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # convert from BGR (cv default) to RGB (mediapipe default)
            hand_RGB = hands.process(rgb_frame)

            predicted_class = "nothing" # default prediction

            # get box and dots for hand in camera
            if hand_RGB.multi_hand_landmarks:
                for hand_points in hand_RGB.multi_hand_landmarks:
                    # dynamic sizing of box around hand based on landmarks
                    top_left_X, top_left_Y = width, height # right bottom corner -> reset each loop
                    bottom_right_X, bottom_right_Y = 0, 0 # top left corner -< reset each loop

                    for i in hand_points.landmark: # iterate 21 landmarks for the hand
                        x, y = int(i.x * width), int(i.y * height) # i in range [0,1] is converted into pixel value
                        # figure out box dimensions
                        top_left_X = min(top_left_X, x)
                        top_left_Y = min(top_left_Y, y)
                        bottom_right_X = max(bottom_right_X, x)
                        bottom_right_Y = max(bottom_right_Y, y)
                        # cv.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dots for landmark

                    # padding for boxes is added for clearer hand boxing
                    top_left_X = max(top_left_X - padding, 0)
                    top_left_Y = max(top_left_Y - padding, 0)
                    bottom_right_X = min(bottom_right_X + padding, width)
                    bottom_right_Y = min(bottom_right_Y + padding, height)

                    # mpDraw.draw_landmarks(frame, hand_points, mpHands.HAND_CONNECTIONS) # draw dots and lines on hand

                    # drawing box around hand
                    cv.rectangle(frame, (top_left_X, top_left_Y), (bottom_right_X, bottom_right_Y), (0, 255, 0), 2)
                    
                # gesture detection
                region = frame[static_top_left_Y:static_bottom_right_Y, static_top_left_X:static_bottom_right_X]
                filtered_frame = filter_frame(region)
                frame_model = model_frame_size(filtered_frame, 224, 224)  # change to 224 by 224 -> required by model
                frame_model = normalize_pixels(frame_model) # normalize pixels (0 to 1 value)
                frame_model = add_extra_dimension(frame_model) # add an extra dimension

                try:
                    # get prediction from model -> using copy of image that is changed
                    prediction = model(frame_model).numpy() # convert to numpy array
                    probabilities = prediction[0] # get array of probabilities for all classes
                    prediction_key = np.argmax(probabilities) # get index of highest probability
                    predicted_class = classes[prediction_key + 1] # get class label

                    # calculating percentage for each class
                    percentages = probabilities * 100
                    most_likely_percentage = percentages[prediction_key]
                    print(f"Predicted Class: {predicted_class} ({most_likely_percentage:.2f}%)")
                    print("All class probabilities:", percentages)

                    # display original frame & add text for most confident class
                    text_coordinates = (int(frame.shape[1] * 0.05), int(frame.shape[0] * 0.1))
                    add_text_on_frame(frame, f"{predicted_class} ({most_likely_percentage:.2f}%)", text_coordinates, (0, 255, 0))
                    
                    ## display probabilities chart
                    # probability_image = plot_probabilities(percentages, classes)
                    # probability_image = rescale_frame(probability_image, scale=0.5)
                    # x_offset, y_offset = 10, 10
                    # frame[y_offset:y_offset + probability_image.shape[0], x_offset:x_offset + probability_image.shape[1]] = probability_image

                except Exception as e:
                    print(f"Error during prediction: {e}")

            # draw static box on the frame
            cv.rectangle(frame, (static_top_left_X, static_top_left_Y), (static_bottom_right_X, static_bottom_right_Y), (0, 255, 0), 2)

            # display image as new window
            cv.imshow('ASL', frame)

            # keyboard binding (ms)-> 0 means it waits infinite amount of time for key to be pressed
            # cv.waitKey(0)
            # Wait for a key press, exit on 'q'
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except cv.error as e:
        print(f"An error occurred: {e}")

    finally:
        # Release the capture object and close windows
        capture.release()
        cv.destroyAllWindows()


def print_image(model: hub.KerasLayer, classes: dict) -> None:
    """ prints out an image and runs classification function on it"""
    # gets a path to image and returns a matrix of pixels
    try:
        img_path = './testImages/'
        img = cv.imread(img_path + "A.png")
        if img is None:
            print("Error: image not found")
            return 
    except Exception as e:
        print(f"Error loading the image: {e}")
        return   

    # rescale image -> create a frame version for just the model to test on
    img_model = model_frame_size(img, 224, 224) # change to 224 by 224 -> required by model
    img_model = normalize_pixels(img_model) # normalize pixels (0 to 1 value)
    img_model = add_extra_dimension(img_model) # add an extra dimension

    try:
        # get prediction from model -> using copy of image that is changed
        prediction = model(img_model)
        prediction_key = np.argmax(prediction.numpy())
        predicted_class = classes[prediction_key + 1]
        print("Prediction done by model on image by percentage: ", prediction)
        print("Prediction done by model final result: ", predicted_class)

        # print original image
        img = rescale_frame(img, scale=2.0)
        text_coordinates = (int(img.shape[1] * 0.05), int(img.shape[0] * 0.1))
        add_text_on_frame(img, predicted_class, text_coordinates, (0, 255, 0))
        # display image as new window
        cv.imshow('ASL', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
  
    except Exception as e:
        print(f"Error during prediction: {e}")


## ****************************** RUNNING MAIN FUNCTION ********************************
if __name__ == "__main__":
    main()