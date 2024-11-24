import cv2
import numpy as np

# Function to load an image
def load_image(image_path, grayscale=True):
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Error: Image at {image_path} could not be loaded.")
    return img

# Function to resize an image
def resize_image(image, size=(48, 48)):
    resized_img = cv2.resize(image, size)
    return resized_img

# Function to normalize an image
def normalize_image(image):
    normalized_img = image / 255.0
    return normalized_img

# Function to preprocess a single image
def preprocess_image(image_path, size=(48, 48), grayscale=True):
    img = load_image(image_path, grayscale)
    img = resize_image(img, size)
    img = normalize_image(img)
    return img

# Function to load and preprocess a batch of images
def load_and_preprocess_images(image_paths, size=(48, 48), grayscale=True):
    images = []
    for image_path in image_paths:
        img = preprocess_image(image_path, size, grayscale)
        images.append(img)
    images = np.array(images)
    return images

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    return image

# Function to detect faces in an image using Haar cascades
def detect_faces(image, face_cascade_path='haarcascade_frontalface_default.xml'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to apply a filter to an image
def apply_filter(image, filter_type='blur'):
    if filter_type == 'blur':
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'edge':
        filtered_image = cv2.Canny(image, 100, 200)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    return filtered_image
