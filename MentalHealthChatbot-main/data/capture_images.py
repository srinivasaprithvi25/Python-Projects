import cv2
import os
import threading
import logging
import json
import argparse
import time
import tkinter as tk
from tkinter import messagebox

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration settings for the application."""
    
    def __init__(self, config_file='config.json'):
        self.load_config(config_file)

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.expressions = config.get("expressions", [])
            self.num_images_per_expression = config.get("num_images_per_expression", 100)  # Reduced
            self.base_dir = config.get("base_dir", "data/processed/images")
            self.camera_width = config.get("camera_width", 160)  # Lower resolution
            self.camera_height = config.get("camera_height", 120)
            self.image_quality = config.get("image_quality", 95)  # JPEG quality percentage

            # Validate config values
            if not self.expressions:
                raise ValueError("No expressions found in the config file.")
            if self.num_images_per_expression <= 0:
                raise ValueError("Number of images per expression must be positive.")
            if not os.path.exists(self.base_dir):
                raise ValueError("Base directory for images does not exist.")
            logging.info("Configuration loaded successfully.")

        except FileNotFoundError:
            logging.error(f"Configuration file '{config_file}' not found.")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from the file '{config_file}'.")
            raise

class ImageCapture:
    """Class to handle the image capturing process."""

    def __init__(self, config):
        self.config = config
        self.capture_event = threading.Event()
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.is_capturing = False
        self.current_frame = None
        self.lock = threading.Lock()

    def create_directories(self):
        """Create directories for each expression."""
        os.makedirs(self.config.base_dir, exist_ok=True)
        for expression in self.config.expressions:
            os.makedirs(os.path.join(self.config.base_dir, expression), exist_ok=True)
        logging.info("Directories created for expressions.")

    def capture_frame(self):
        """Continuously capture frames from the webcam."""
        frame_counter = 0
        while self.capture_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if frame_counter % 5 == 0:  # Process every 5th frame
                    with self.lock:
                        self.current_frame = frame
                frame_counter += 1

    def capture_face_images(self, expression):
        """Capture images for a given expression."""
        count = 0
        self.capture_event.set()  # Start capturing images

        while count < self.config.num_images_per_expression and self.capture_event.is_set():
            with self.lock:
                if self.current_frame is None:
                    continue
                frame_to_process = self.current_frame.copy()

            # Convert to grayscale and detect faces
            gray_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_image = gray_frame[y:y + h, x:x + w]
                img_path = os.path.join(self.config.base_dir, expression, f"{expression}_{count + 1}.jpg")
                cv2.imwrite(img_path, face_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.image_quality])
                logging.info(f'Captured image {count + 1} for: {expression}')
                count += 1
                time.sleep(0.1)  # Brief pause to avoid overwhelming

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exiting...")
                break

        if count >= self.config.num_images_per_expression:
            logging.info(f"Captured {count} images for: {expression}")

    def start_capture(self):
        """Start the capturing process."""
        self.create_directories()
        if not self.cap.isOpened():
            logging.error("Error: Could not open webcam.")
            return

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        self.capture_event.set()  # Allow frame capturing
        capture_thread = threading.Thread(target=self.capture_frame)
        capture_thread.start()

        for expression in self.config.expressions:
            self.is_capturing = True
            self.capture_face_images(expression)

        self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        if self.cap.isOpened():
            self.capture_event.clear()  # Stop frame capturing
            self.cap.release()
        cv2.destroyAllWindows()

class App:
    """Main application GUI for capturing images."""

    def __init__(self, config):
        self.root = tk.Tk()
        self.root.title("Emotion Image Capture")
        self.image_capture = ImageCapture(config)

        self.label = tk.Label(self.root, text="Select Expressions:")
        self.label.pack()

        self.expression_var = tk.StringVar(value=self.image_capture.config.expressions[0])
        self.expression_menu = tk.OptionMenu(self.root, self.expression_var, *self.image_capture.config.expressions)
        self.expression_menu.pack()

        self.capture_button = tk.Button(self.root, text="Start Capture", command=self.start_capture)
        self.capture_button.pack()

        self.exit_button = tk.Button(self.root, text="Exit", command=self.on_exit)
        self.exit_button.pack()

        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack()

    def start_capture(self):
        """Start the capture process."""
        expression = self.expression_var.get()
        self.capture_button.config(state=tk.DISABLED)
        self.image_capture.start_capture()
        self.update_progress_label()

    def update_progress_label(self):
        """Update the progress label."""
        while self.image_capture.is_capturing:
            self.progress_label.config(text=f"Captured: {self.image_capture.progress}/{self.image_capture.config.num_images_per_expression}")
            self.root.update_idletasks()
            time.sleep(0.1)

    def on_exit(self):
        """Handle exit confirmation."""
        if messagebox.askokcancel("Exit", "Do you really want to exit?"):
            self.image_capture.cleanup()
            self.root.quit()

    def run(self):
        """Run the GUI application."""
        self.root.mainloop()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Face Image Capture for Emotions")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration JSON file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    try:
        config = Config(args.config)
        app = App(config)
        app.run()
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error: {e}")
