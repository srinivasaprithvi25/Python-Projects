import os
import pandas as pd
import cv2
import sys
from flask import Flask, render_template, jsonify, request, Response
from textblob import TextBlob

# Add the parent directory to the sys.path to enable absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models using absolute imports
from models.real_time_emotion import detect_emotion
from models.chat_interaction import ask_question_and_get_response

app = Flask(__name__)

# Load questions from the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')
questions_df = pd.read_csv(os.path.join(DATA_DIR, 'mental_health_train.csv'))
questions = questions_df['text'].tolist()  # Get questions from the dataset
user_responses = []  # List to store user answers and detected emotions

# Initialize video capture for real-time emotion detection
cap = cv2.VideoCapture(0)

def gen_frames():
    """Generate frames from the webcam for real-time emotion detection."""
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect emotion from the current frame
            emotion_frame, detected_emotion = detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', emotion_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main chatbot page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the webcam feed to the frontend."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_question', methods=['GET'])
def get_question():
    """Get the next question from the dataset."""
    global user_responses
    if len(user_responses) < len(questions):
        current_question = questions[len(user_responses)]
        return jsonify(question=current_question)
    else:
        return jsonify(question=None)

@app.route('/answer_question', methods=['POST'])
def answer_question():
    """Handle the user's response and detected emotion."""
    global user_responses
    user_input = request.json.get('user_input', '')  # Default to an empty string if no input
    detected_emotion = request.json.get('detected_emotion', 'Neutral')  # Default emotion

    # Append the user's answer and the detected emotion to the response list
    user_responses.append({'response': user_input, 'emotion': detected_emotion})

    # Get the next question, or return None if we've reached the end
    next_question = None
    if len(user_responses) < len(questions):
        next_question = questions[len(user_responses)]  # Get the next question

    # Return both the next question and the detected emotion
    return jsonify(next_question=next_question, detected_emotion=detected_emotion)

@app.route('/predict_sentiment', methods=['GET'])
def predict_sentiment():
    """Predict the overall sentiment after all questions are answered."""
    if user_responses:
        text_sentiment_score = 0
        emotion_sentiment_score = 0
        response_count = len(user_responses)

        # Calculate text sentiment
        for response in user_responses:
            blob = TextBlob(response['response'])
            text_sentiment_score += blob.sentiment.polarity

            # Assign sentiment score based on detected emotion
            if response['emotion'] == "Happy":
                emotion_sentiment_score += 1
            elif response['emotion'] == "Sad":
                emotion_sentiment_score -= 1
            elif response['emotion'] == "Angry":
                emotion_sentiment_score -= 1
            elif response['emotion'] == "Neutral":
                emotion_sentiment_score += 0

        # Calculate average sentiment scores
        avg_text_sentiment = text_sentiment_score / response_count
        avg_emotion_sentiment = emotion_sentiment_score / response_count

        # Combine text and emotion sentiment for overall sentiment
        overall_sentiment_score = (avg_text_sentiment + avg_emotion_sentiment) / 2

        # Determine sentiment label based on overall sentiment score
        if overall_sentiment_score > 0:
            overall_sentiment = "Positive"
        elif overall_sentiment_score < 0:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
    else:
        overall_sentiment = "Neutral"

    return jsonify(sentiment=overall_sentiment)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shut down the video capture and release resources."""
    cap.release()
    cv2.destroyAllWindows()
    return "Video capture released"

if __name__ == '__main__':
    app.run(debug=True)
