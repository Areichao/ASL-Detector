import cv2 as cv
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import mediapipe as mp
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# *************************** MODEL IMPORTING ************************************
def main() -> None:
    # Load the model and processor
    try:
        model_name = "RavenOnur/Sign-Language"
        model = AutoModelForImageClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        print("Model and processor loaded successfully!")
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit()

    # ***************************** GLOBAL VARIABLES AND CLASS DEFINITION *************************************
    # Use the model's config to get the label mapping
    if hasattr(model.config, 'id2label'):
        classes = model.config.id2label
    else:
        # If id2label is not available, define classes manually
        classes = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
            9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
            17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
            25: 'Z', 26: 'del', 27: 'space', 28: 'nothing'
        }

    # **************************** APPLICATION CALL **********************************************
    captureVideo(model, processor, classes)

# *************************** FUNCTION CALLS -> private & helper functions ************************************

def rescaleFrame(frame: np.ndarray, scale: float = 0.75) -> np.ndarray:
    """Rescales the frame by a given scale factor."""
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def modelFrameSize(frame: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    """Rescales the frame to the specified size for the model."""
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(capture: cv.VideoCapture, width: int, height: int) -> None:
    """Changes the resolution of the video capture."""
    capture.set(3, width)  # Property ID 3: Width
    capture.set(4, height)  # Property ID 4: Height

def addText(frame: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int], scale: float = 1.0, thickness: int = 2) -> None:
    """Adds text to the frame."""
    cv.putText(frame, text, origin, cv.FONT_HERSHEY_TRIPLEX, scale, colour, thickness)

def drawRectangle(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Defines a rectangle in the center of the frame."""
    height, width, _ = frame.shape
    box_size = 224
    top_left_x = (width - box_size) // 2
    top_left_y = (height - box_size) // 2
    bottom_right_x = top_left_x + box_size
    bottom_right_y = top_left_y + box_size
    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

# ************************** GETTING IMAGE OR VIDEO ****************************************

def captureVideo(model, processor, classes) -> None:
    """Capture webcam video and classify ASL letters with optimized MediaPipe Hands."""
    try:
        capture = cv.VideoCapture(0)
        if not capture.isOpened():
            print("Error: Could not open webcam.")
            return
        
        changeRes(capture, 640, 480)  # Set webcam resolution

        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        mp_draw = mp.solutions.drawing_utils

        while True:
            ret, frame = capture.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            hand_region = None
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    h, w, c = frame.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    padding = 20
                    x_min = max(x_min - padding, 0)
                    y_min = max(y_min - padding, 0)
                    x_max = min(x_max + padding, w)
                    y_max = min(y_max + padding, h)

                    hand_region = frame[y_min:y_max, x_min:x_max]

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if hand_region is not None:
                hand_image = Image.fromarray(cv.cvtColor(hand_region, cv.COLOR_BGR2RGB))
                inputs = processor(images=hand_image, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    predicted_label = classes.get(predicted_class_idx, "Unknown")

                text_coordinates = (int(frame.shape[1] * 0.05), int(frame.shape[0] * 0.1))
                addText(frame, predicted_label, text_coordinates, (0, 255, 0))

            cv.imshow('ASL Detection', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        capture.release()
        cv.destroyAllWindows()

# ****************************** RUNNING MAIN FUNCTION ********************************
if __name__ == "__main__":
    main()
