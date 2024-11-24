import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import combined_analysis from models
import models.combined_analysis as combined_analysis

def test_combined_emotion_analysis():
    facial_emotion = "happy"
    text_sentiment = "happy"

    overall_mood = combined_analysis.combine_emotions(facial_emotion, text_sentiment)
    
    assert overall_mood == "Your overall mood seems to be: happy", "Test failed: Mood should be happy"

if __name__ == "__main__":
    test_combined_emotion_analysis()
    print("Test passed: Combined emotion analysis works correctly.")
