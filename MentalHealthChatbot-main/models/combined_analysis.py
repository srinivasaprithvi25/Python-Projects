import sys
import os
import cv2


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.real_time_emotion import detect_emotion  # Now it should work
from models.chat_interaction import ask_question_and_get_response
def combine_emotions(facial_emotion, text_sentiment):
    """
    Combines the facial emotion and text sentiment and returns the overall mood.
    """
    if facial_emotion == text_sentiment:
        return f"Your overall mood seems to be: {facial_emotion}"
    else:
        return f"Facial emotion shows {facial_emotion}, but your text indicates {text_sentiment}. You might have mixed feelings."

def main():
    cap = cv2.VideoCapture(0)
    print("Press 'a' to start chatbot interaction and analyze text sentiment.")
    print("Press 'q' to quit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access the camera.")
            break

        # Detect facial emotion from the camera feed
        emotion_frame, facial_emotion = detect_emotion(frame)
        if facial_emotion:
            cv2.imshow('Real-time Emotion Detection', emotion_frame)

        # Press 'a' to start chatbot interaction and analyze text sentiment
        if cv2.waitKey(1) & 0xFF == ord('a'):
            print("Chatbot interaction started...")
            text_sentiment = ask_question_and_get_response()

            # Combine facial emotion and text sentiment analysis
            if facial_emotion and text_sentiment:
                overall_mood = combine_emotions(facial_emotion, text_sentiment)
                print(overall_mood)
            else:
                print("Error: Unable to detect facial emotion or text sentiment.")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting application.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
