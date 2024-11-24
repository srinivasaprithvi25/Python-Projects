import os
from keras.models import load_model

def load_cnn_model(model_path):
    """
    Load a pre-trained CNN model for facial emotion recognition.
    
    Args:
    - model_path (str): The file path to the model.
    
    Returns:
    - model: The loaded Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return load_model(model_path)

def save_model(model, model_path):
    """
    Save the model to the specified path.
    
    Args:
    - model: The Keras model to save.
    - model_path (str): The path where to save the model.
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Additional utility functions for managing models can be added here
