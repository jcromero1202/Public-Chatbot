import os
import streamlit as st
import requests
import json
from dotenv import load_dotenv

# For local development, load environment variables from the .env file.
# In Streamlit Cloud, st.secrets will be available.
load_dotenv()

def get_config_value(key: str) -> str:
    """
    Retrieve configuration value from st.secrets (if available)
    or fallback to environment variables.
    """
    return st.secrets.get(key, os.getenv(key))

# --------------------------
# 1. Configuration Constants
#    Retrieve settings and sensitive data from environment variables.
# --------------------------
BASE_API_URL = os.getenv("BASE_API_URL")
LANGFLOW_ID = os.getenv("LANGFLOW_ID")
FLOW_ID = os.getenv("FLOW_ID")
APPLICATION_TOKEN = os.getenv("APPLICATION_TOKEN")  # Must be set in the .env file

if not APPLICATION_TOKEN:
    raise ValueError("APPLICATION_TOKEN environment variable not set. Please set it in your .env file.")

# --------------------------
# 2. Function to Run the Langflow Flow
#    This function sends the user's message to the Langflow API and returns the bot's response.
# --------------------------
def run_flow(user_message: str) -> str:
    """
    Sends the user's message to the Langflow flow and returns the LLM's response.
    """
    endpoint = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{FLOW_ID}"
    # Setup the headers with the authorization token and content type
    headers = {
        "Authorization": f"Bearer {APPLICATION_TOKEN}",
        "Content-Type": "application/json"
    }
    # Prepare the payload to send to the API
    payload = {
        "input_value": user_message,
        "output_type": "chat",   # Expected output type
        "input_type": "chat"     # Expected input type
    }

    # Send the POST request to the Langflow API
    response = requests.post(endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Return the 'output' from the API response, or a default message if key is missing
        return data.get("output", "No output found in response.")
    else:
        # Return an error message if the request failed
        return f"Error: {response.status_code} - {response.text}"

# --------------------------
# 3. Streamlit UI Interface
#    Provides a simple user interface to send messages and display the chatbot response.
# --------------------------
def main():
    # Display the title of the app
    st.title("My Conversational Chatbot")

    # Input field for the user to type a message
    user_input = st.text_input("Enter your message:")

    # When the "Send" button is clicked, process the input
    if st.button("Send"):
        if user_input.strip():
            answer = run_flow(user_input)
            # Display the chatbot's response
            st.write(f"**Bot:** {answer}")
        else:
            # Warn the user if the input is empty
            st.warning("Please enter a message first.")

if __name__ == "__main__":
    main()
