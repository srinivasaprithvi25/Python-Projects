import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Psychological questions to assess mental health
questions = [
    "How would you describe your current mood?",
    "What emotions do you feel most often?",
    "Are you generally optimistic or pessimistic?",
    "How do you cope with stress or anxiety?",
    "Do you feel connected to friends and family?",
    "How satisfied are you with your life?",
    "How often do you experience feelings of sadness?",
    "What activities bring you joy?",
    "How do you typically respond to challenges?",
    "How would you rate your overall mental well-being?",
    "Do you find it easy to express your feelings?",
    "How often do you feel overwhelmed?",
    "Do you have a support system in place?",
    "How do you handle negative thoughts?",
    "Do you find it easy to relax?",
    "How would you describe your self-esteem?",
    "How do you feel about your future?",
    "Do you often feel anxious in social situations?",
    "How do you prioritize your mental health?",
    "What do you do to unwind after a stressful day?"
]

# Step 1: Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)

# Step 2: Save the tokenizer to a file
with open('models/tokenizer.pkl', 'wb') as f:  # Make sure the 'models' directory exists
    pickle.dump(tokenizer, f)

print("Tokenizer saved successfully to models/tokenizer.pkl.")
