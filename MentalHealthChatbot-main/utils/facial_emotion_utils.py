import cv2
import numpy as np

def preprocess_face_image(face_image):
    """
    Preprocess the face image for emotion recognition.
    Converts the image to grayscale, resizes it, and normalizes the pixel values.
    
    Args:
    - face_image (ndarray): The face image to be processed.
    
    Returns:
    - ndarray: Preprocessed image ready for model input.
    """
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_image = cv2.resize(face_image, (48, 48)) / 255.0      # Resize and normalize
    return np.reshape(face_image, (1, 48, 48, 1))  # Reshape for the model

def draw_face_rectangle(frame, x, y, w, h, emotion):
    """
    Draw a rectangle around the detected face and put the predicted emotion text.
    
    Args:
    - frame (ndarray): The video frame.
    - x, y (int): Top left corner coordinates of the rectangle.
    - w, h (int): Width and height of the rectangle.
    - emotion (str): Predicted emotion to display.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Additional utility functions can be added here
