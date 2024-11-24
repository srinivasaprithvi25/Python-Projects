from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import pickle
import sys

# Force the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

# Load the model from the pickle file
with open('hybrid_model.pkl', 'rb') as f:
    hybrid_model = pickle.load(f)

# Define the input shape for the model
input_shape = (150, 150, 3)

# Define the class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image):
    # Resize and preprocess the image
    image = image.resize(input_shape[:2])  # Resize to the input shape required by the model
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pituitary')
def pituitary():
    return render_template('pituitary.html')

@app.route('/giloma')
def glioma():
    return render_template('giloma.html')

@app.route('/meningioma')
def meningioma():
    return render_template('meningioma.html')

@app.route('/model')
def model_ui():
    # return render_template('index1.html')
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('results_new.html', error='No file part'), 400
    file = request.files['file']
    if file.filename == '':
        return render_template('results_new.html', error='No selected file'), 400
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file.stream).convert('RGB')
        image_array = preprocess_image(image)
        
        # For InceptionV3, duplicate the image_array to match the model's input requirements
        image_array_dup = np.copy(image_array)
        
        # Use the hybrid model to make predictions
        prediction = hybrid_model.predict([image_array, image_array_dup])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]
        
        return render_template('results_new.html', predicted_label=predicted_label)
    return render_template('results_new.html', error='Invalid file type'), 400


if __name__ == '__main__':
    app.run(debug=True)
