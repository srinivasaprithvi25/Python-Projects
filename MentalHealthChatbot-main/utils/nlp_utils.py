import spacy
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load SpaCy model for text preprocessing
nlp = spacy.load('en_core_web_sm')

# Function to preprocess a single text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Function to preprocess a DataFrame of text data
def preprocess_text_dataframe(df, text_column):
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    return df

# Function to tokenize and pad sequences
def tokenize_and_pad_sequences(texts, max_words=5000, max_seq_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post')
    return padded_sequences, tokenizer

# Function to one-hot encode labels
def one_hot_encode_labels(labels):
    label_to_int = {label: idx for idx, label in enumerate(pd.Series(labels).unique())}
    int_labels = [label_to_int[label] for label in labels]
    one_hot_labels = to_categorical(int_labels, num_classes=len(label_to_int))
    return one_hot_labels, label_to_int

# Function to split data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to prepare text data for model input
def prepare_text_data(df, text_column, label_column, max_words=5000, max_seq_len=100):
    df = preprocess_text_dataframe(df, text_column)
    X, tokenizer = tokenize_and_pad_sequences(df['cleaned_text'], max_words, max_seq_len)
    y, label_to_int = one_hot_encode_labels(df[label_column])
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, tokenizer, label_to_int

