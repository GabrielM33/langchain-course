import os

from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

# Firebase Configuration
PROJECT_ID = "langchain-42a7b"
SESSION_ID = "user1_session"
COLLECTION_NAME = "chat_history"
CREDENTIALS_PATH = (
    "firebase-credentials.json"  # Path to the JSON key file you downloaded
)

# Initialize Firebase Client
print("Initializing Firebase Chat Message History...")

client = firestore.Client.from_service_account_json(CREDENTIALS_PATH)

# Initialize Firebase Chat History
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

print("Chat History Initialized")
print("Current Chat History:", chat_history.messages)

# Initialize Models
openai_model = ChatOpenAI(model="gpt-4o-mini")
google_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

print("Start chatting with the AI. Type 'exit' to stop.")

while True:
    human_input = input("You: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = google_model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print("AI:", ai_response.content)

# Check Firebase for chat history
