import unittest
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load the SpaCy model for text preprocessing
import spacy
nlp = spacy.load('en_core_web_sm')

# Preprocess text using SpaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Define the test class
class TestNLPModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the trained model
        cls.model = load_model('models/nlp_model.h5')

        # Load the tokenizer
        with open('models/tokenizer.json') as f:
            data = json.load(f)
            tokenizer_json_string = json.dumps(data)  # Convert dict back to JSON string
            cls.tokenizer = tokenizer_from_json(tokenizer_json_string)

        # Define the maximum sequence length
        cls.max_seq_len = 100

    def test_preprocess_text(self):
        # Test the text preprocessing function
        sample_text = "I'm feeling great today!"
        expected_output = "feel great today"
        self.assertEqual(preprocess_text(sample_text), expected_output)

    def test_model_prediction(self):
        # Test the model's prediction
        sample_text = "I am very happy today!"
        preprocessed_text = preprocess_text(sample_text)
        sequence = self.tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_seq_len)

        # Predict the sentiment
        predictions = self.model.predict(padded_sequence)
        predicted_label = np.argmax(predictions, axis=1)[0]

        # Load the label mapping
        label_to_int = {'happy': 0, 'sad': 1, 'anxious': 2}
        int_to_label = {v: k for k, v in label_to_int.items()}

        self.assertIn(predicted_label, int_to_label)

# Run the tests
if __name__ == '__main__':
    unittest.main()
