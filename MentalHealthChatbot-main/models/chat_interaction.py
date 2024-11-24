
import pickle
import random

# Load the sentiment tokenizer
try:
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print("Error: Tokenizer file not found. Please ensure the tokenizer is saved.")
    exit(1)

def ask_question_and_get_response(question):
    """
    Asks a question to the user and returns the user's response.
    """
    print(question)
    response = input("Your response: ")  # Get the response from the user
    return response

def analyze_response(response):
    """
    Analyze the user's response for sentiment.
    This is a placeholder for actual sentiment analysis logic.
    """
    # Simulate sentiment analysis by randomly categorizing.
    sentiments = ['happy', 'sad', 'angry', 'neutral', 'anxious']
    predicted_sentiment = random.choice(sentiments)  # Simulate prediction
    print(f"Analyzing response: '{response}'")
    print(f"Predicted sentiment: {predicted_sentiment}")
    return predicted_sentiment

# Example usage of the function
if __name__ == "__main__":
    question = "How are you feeling today? (happy, sad, anxious, etc.)"
    user_response = ask_question_and_get_response(question)
    sentiment = analyze_response(user_response)
    print(f"Final analysis: You seem to be feeling {sentiment}.")