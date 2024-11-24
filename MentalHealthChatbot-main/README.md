# MentalHealthChatbot

## Overview

MentalHealthChatbot is a project designed to provide mental health support using Natural Language Processing (NLP) and Computer Vision. This chatbot can understand and respond to text input and analyze facial expressions to offer a more personalized experience.

## Features

- **NLP Model**: Understands and responds to user text input.
- **LSTM Model**: Analyzes user text to detect emotions and sentiments.
- **Computer Vision Model**: Analyzes facial expressions to detect emotions.
- **Data Preprocessing**: Includes utilities for preprocessing text and image data.

## Usage

### Data Preparation

1. **Capture Images**:
    Run the script to capture images for various expressions:
    ```sh
    python data/capture_images.py
    ```

2. **Create Dataset**:
**Create Dataset**:
    Create the dataset for text and image analysis:
    ```sh
    python data/dataset_creation.py
    python data/create_facial_expressions_dataset.py
    ```

### Model Training

1. **Train All Models**:
    Train the NLP, LSTM, and CV models:
    ```sh
    python models/model_training.py
    ```

### Testing

1. **Run Tests**:
    Run the tests to ensure everything is working:
    ```sh
    python -m unittest discover tests
    ```