import os
import numpy as np
from keras.models import load_model
import cv2
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the pre-trained CNN model
model_path = os.path.join('models', 'cv_model.h5')
assert os.path.exists(model_path), f"Model file not found at {model_path}"
model = load_model(model_path)

# Emotion categories
categories = ['angry', 'happy', 'neutral', 'sad', 'surprised']

def test_predict_emotion():
    """
    This test function checks if the CNN model is able to predict emotions 
    from a synthetic test image.
    """
    # Create a synthetic test image (48x48 grayscale)
    test_image = np.random.rand(48, 48, 1).astype(np.float32)  # Random grayscale image
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    
    # Make a prediction using the loaded model
    prediction = model.predict(test_image)
    predicted_label = np.argmax(prediction)
    predicted_emotion = categories[predicted_label]

    assert predicted_emotion in categories, "Predicted emotion is not valid"
    print(f"Test passed: Model predicted {predicted_emotion}")

def test_predict_on_real_image():
    """
    This test function loads a real image, processes it, and checks the model prediction.
    """
    image_path = os.path.join('data', 'processed', 'images', 'happy', 'happy_1.jpg')  # Update with actual test image path
    assert os.path.exists(image_path), f"Image file not found at {image_path}"

    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (48, 48)) / 255.0  # Resize to 48x48 and normalize
    image_reshaped = np.reshape(image_resized, (1, 48, 48, 1))

    # Make a prediction using the loaded model
    prediction = model.predict(image_reshaped)
    predicted_label = np.argmax(prediction)
    predicted_emotion = categories[predicted_label]

    assert predicted_emotion == 'happy', f"Test failed: Expected 'happy', but got {predicted_emotion}"
    print(f"Test passed: Model correctly predicted 'happy'")

if __name__ == "__main__":
    print("Running CNN model tests...")
    test_predict_emotion()
    test_predict_on_real_image()
    print("All tests passed successfully!")
