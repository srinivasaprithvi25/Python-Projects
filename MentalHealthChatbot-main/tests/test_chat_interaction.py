import sys
import os
from unittest.mock import patch

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models.chat_interaction as chat_interaction  # Import the chat interaction module

def test_ask_question_and_get_response():
    # Mock the input and response
    with patch('builtins.input', return_value="I'm feeling anxious"):
        response = chat_interaction.ask_question_and_get_response("How are you feeling?")
        
        # You can assert based on the expected processing of the response
        assert isinstance(response, str), "Expected a string response from the chatbot"
        assert "anxious" in response.lower(), "Expected the chatbot to identify 'anxious' in the response"

if __name__ == "__main__":
    test_ask_question_and_get_response()
