import json
import pandas as pd
import numpy as np
import cv2  # Import cv2 for image processing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os

# Load the SpaCy model for text preprocessing
import spacy
nlp = spacy.load('en_core_web_sm')

# Load pre-trained models and tokenizer
nlp_model = load_model('models/nlp_model.h5')
lstm_model = load_model('models/lstm_model.h5')
cv_model = load_model('models/cv_model.h5')

with open('models/tokenizer.json') as f:
    data = json.load(f)
    tokenizer_json_string = json.dumps(data)  # Convert dict back to JSON string
    tokenizer = tokenizer_from_json(tokenizer_json_string)

# Define the maximum sequence length
max_seq_len = 100

# Function to preprocess text using SpaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Function to predict sentiment from text
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_len)
    prediction = nlp_model.predict(padded_sequence)
    return np.argmax(prediction, axis=1)[0]

# Function to process image and predict emotion
def process_image_and_predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unable to read at path: {image_path}")
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 1))
    prediction = cv_model.predict(img)
    return np.argmax(prediction, axis=1)[0]

# Example usage
if __name__ == "__main__":
    text = "I am feeling happy today!"
    sentiment = predict_sentiment(text)
    print(f"Predicted Sentiment: {sentiment}")

    # Ensure these paths point to valid image files
    image_dir = 'data/processed/images'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_path in image_files:
        try:
            emotion = process_image_and_predict(image_path)
            print(f"Predicted Emotion for {image_path}: {emotion}")
        except ValueError as e:
            print(e)
