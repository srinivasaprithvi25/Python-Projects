import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the training data
train_data_path = 'data/processed/mental_health_train.csv'

# Load the dataset
try:
    df_train = pd.read_csv(train_data_path)
    print("Training data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{train_data_path}' was not found.")
    exit(1)

# Assuming the DataFrame has 'text' and 'label' columns
texts = df_train['text'].values
labels = df_train['label'].values

# Set parameters
max_words = 10000  # Maximum number of words to consider
max_length = 100   # Maximum length of each input sequence
embedding_dim = 100

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

# Save the tokenizer for later use
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_length)

# Convert labels to categorical (if needed)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(len(set(labels)), activation='softmax'))  # Adjust the output layer based on the number of classes

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10

try:
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error during training: {e}")

# Save the trained model
model.save('models/lstm_model.h5')
print("Model saved as 'lstm_model.h5'.")
