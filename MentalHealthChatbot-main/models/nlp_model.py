def train():
    import spacy
    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical

    # Load SpaCy model for text preprocessing
    nlp = spacy.load('en_core_web_sm')  # Ensure to have spaCy installed and the model downloaded

    # Function to preprocess text data using SpaCy
    def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    # Load train and test datasets
    train_df = pd.read_csv('data/processed/mental_health_train.csv')
    test_df = pd.read_csv('data/processed/mental_health_test.csv')

    # Apply preprocessing to the datasets
    train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
    test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)

    # Ensure labels are consistent across train and test sets
    all_classes = list(set(train_df['label']).union(set(test_df['label'])))
    print("All classes:", all_classes)

    # Create label to integer mappings
    label_to_int = {label: i for i, label in enumerate(all_classes)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    # Map labels to integers
    train_df['label'] = train_df['label'].map(label_to_int)
    test_df['label'] = test_df['label'].map(label_to_int)

    # Prepare training and testing data
    X_train = train_df['cleaned_text'].values
    y_train = to_categorical(train_df['label'].values, num_classes=len(all_classes))
    X_test = test_df['cleaned_text'].values
    y_test = to_categorical(test_df['label'].values, num_classes=len(all_classes))

    # Print shapes for debugging
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=5000, lower=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    max_seq_len = 100
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_len)

    # Build LSTM model
    num_classes = y_train.shape[1]
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_seq_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Save the model
    model.save('models/nlp_model.h5')
    tokenizer_json = tokenizer.to_json()
    with open('models/tokenizer.json', 'w') as f:
        f.write(tokenizer_json)

    print("Model training complete and saved successfully.")

# Ensure this function is called when running as a standalone script
if __name__ == "__main__":
    train()
