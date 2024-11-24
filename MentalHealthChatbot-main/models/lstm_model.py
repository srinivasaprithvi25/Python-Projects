import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(file_path, max_words=5000, max_seq_len=100):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post')
    
    # Convert labels to one-hot encoding
    labels = pd.get_dummies(df['label']).values
    
    return padded_sequences, labels, tokenizer

def train():
    # Load training and testing data
    X_train, y_train, tokenizer = load_and_preprocess_data('data/processed/mental_health_train.csv')
    X_test, y_test, _ = load_and_preprocess_data('data/processed/mental_health_test.csv')

    # Ensure labels are correctly one-hot encoded
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("y_train classes:", y_train[0])
    print("y_test classes:", y_test[0])

    # Aligning test labels to match the training labels
    num_classes = y_train.shape[1]
    y_test = to_categorical([label.argmax() for label in y_test], num_classes=num_classes)

    # Print adjusted shapes
    print("Adjusted y_test shape:", y_test.shape)

    # Build LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=X_train.shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Save the model and tokenizer
    model.save('models/lstm_model.h5')
    tokenizer_json = tokenizer.to_json()
    with open('models/tokenizer.json', 'w') as f:
        f.write(tokenizer_json)

    print("LSTM model training complete and saved successfully.")

# Allow this script to be run directly
if __name__ == "__main__":
    train()
