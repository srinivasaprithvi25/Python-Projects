import spacy
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load SpaCy model for text preprocessing
nlp = spacy.load('en_core_web_sm')

# Text preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Function to preprocess text dataset
def preprocess_text_dataset(df, text_column, label_column):
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    X = df['cleaned_text'].values
    y = pd.get_dummies(df[label_column]).values
    return X, y

# Tokenize and pad sequences
def tokenize_and_pad_sequences(X_train, X_test, max_words=5000, max_seq_len=100):
    tokenizer = Tokenizer(num_words=max_words, lower=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_len)

    return X_train_pad, X_test_pad, tokenizer

# Load and preprocess images
def load_and_preprocess_images(image_paths, labels, img_size=(48, 48)):
    images = []
    processed_labels = []
    for img_path, label in zip(image_paths, labels):
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                processed_labels.append(label)
            else:
                print(f"Error: Image at {img_path} could not be loaded.")
        else:
            print(f"Error: Image path {img_path} does not exist.")
    if not images:
        raise ValueError("No valid images found.")
    images = np.array(images) / 255.0  # Normalize pixel values
    return images, processed_labels

# Convert labels to one-hot encoding
def convert_labels_to_one_hot(labels):
    label_to_int = {label: idx for idx, label in enumerate(np.unique(labels))}
    integer_labels = [label_to_int[label] for label in labels]
    one_hot_labels = to_categorical(integer_labels, num_classes=len(label_to_int))
    return one_hot_labels, label_to_int

