import os
import streamlit as st
import requests
import json
import re
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
    headers = {
        "Authorization": f"Bearer {APPLICATION_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "input_value": user_message,
        "output_type": "chat",
        "input_type": "chat"
    }

    response = requests.post(endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # First, check for a top-level key.
        output = data.get("Text") or data.get("text")
        if output:
            return output
        # If not found, try to extract from the nested structure.
        outputs = data.get("outputs")
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            nested_outputs = outputs[0].get("outputs")
            if nested_outputs and isinstance(nested_outputs, list) and len(nested_outputs) > 0:
                results = nested_outputs[0].get("results", {})
                message = results.get("message", {})
                # Check if the text is in the "data" sub-dictionary.
                if "data" in message and isinstance(message["data"], dict):
                    text = message["data"].get("text")
                    if text:
                        return text
                # Fallback to check directly in message.
                if message.get("text"):
                    return message.get("text")

        st.error("No output found. API response structure is unexpected:")
        st.json(data)
        return "Error: Unexpected API response structure. Please check the debug output above."
    else:
        error_msg = f"Error: {response.status_code} - {response.text}"
        st.error(error_msg)
        return error_msg

# --------------------------
# 3. Streamlit UI Interface
#    Provides a simple user interface to send messages and display the chatbot response.
# --------------------------
def highlight_think(text: str) -> str:
    """
    Wraps any text between <think> and </think> tags in a styled div
    with a dark gray background and white text.
    """
    pattern = re.compile(r'(<think>)(.*?)(</think>)', re.DOTALL)
    highlighted = pattern.sub(r'<div style="background-color: #444; color: #fff; padding: 5px; border-radius: 5px;">\2</div>', text)
    return highlighted

def main():
    # Initialize conversation history in session state if not already set.
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Custom header with two centered lines.
    st.markdown(
        """
        <div style="text-align: center;">
            <h3 style="margin: 0;">Conversational Chatbot</h3>
            <p style="font-size: 16px; margin: 0;">enabled by DeepSeek-R1-distill-Qwen32B with Inference via Groq</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Add a space between the title and conversation history.
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Conversation history header with intermediate font size.
    st.markdown(
        "<h4 style='font-size:20px;'>Conversation History:</h4>",
        unsafe_allow_html=True,
    )
    
    # Display conversation history.
    for chat in st.session_state.history:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['text']}")
        elif chat["role"] == "bot":
            # Highlight any <think> blocks in the bot response.
            bot_text = highlight_think(chat["text"])
            st.markdown(f"**Bot:** {bot_text}", unsafe_allow_html=True)

    # Multi-line text area for long messages.
    user_input = st.text_area("Enter your message:", height=150)
    
    # When the "Send" button is clicked, process the input.
    if st.button("Send"):
        if user_input.strip():
            # Append user message to conversation history.
            st.session_state.history.append({"role": "user", "text": user_input})
            
            # Get bot response.
            answer = run_flow(user_input)
            st.session_state.history.append({"role": "bot", "text": answer})
            
            # Clear the text area after processing.
            st.experimental_rerun()
        else:
            st.warning("Please enter a message first.")

if __name__ == "__main__":
    main()
