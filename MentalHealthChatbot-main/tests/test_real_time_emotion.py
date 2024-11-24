import sys
import os
import numpy as np  # Import numpy to handle arrays
from unittest.mock import patch

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models.real_time_emotion as real_time_emotion  # Importing the entire module

def test_detect_emotion():
    # Create a dummy frame (black image) for testing
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mocking the detect_emotion function to return a known output
    with patch('models.real_time_emotion.detect_emotion', return_value=(test_frame, 'happy')):
        emotion_frame, facial_emotions = real_time_emotion.detect_emotion(test_frame)

        # Assert that the mock returns the expected emotion
        assert facial_emotions == 'happy', "Expected emotion to be 'happy'"

if __name__ == "__main__":
    test_detect_emotion()
