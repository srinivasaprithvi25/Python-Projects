import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained CNN model for emotion recognition
model = load_model('models/cv_model.h5')

# Load Haar Cascade for face detection (OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion categories (these should match the labels in your trained model)
categories = ['angry', 'happy', 'neutral', 'sad', 'surprised']

def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48)) / 255.0
        face_reshaped = np.reshape(face_resized, (1, 48, 48, 1))
        
        prediction = model.predict(face_reshaped)
        emotion = categories[np.argmax(prediction)]
        
        # Draw rectangle around face and put the predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame, emotion if len(faces) > 0 else None

def main():
    cap = cv2.VideoCapture(0)
    print("Starting real-time emotion detection. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to detect emotion
        emotion_frame, facial_emotion = detect_emotion(frame)
        cv2.imshow('Real-time Emotion Detection', emotion_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
