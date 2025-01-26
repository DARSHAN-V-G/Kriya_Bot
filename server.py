from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os
import pickle
import time
from datetime import datetime
from dotenv import load_dotenv
from threading import Lock

load_dotenv()

# Load all Groq API keys from environment variables
groq_api_keys = os.environ.get("GROQ_API_KEYS").split(",")
current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Lock to ensure thread-safe access to shared resources
lock = Lock()

# Tracking requests sent per API key
api_key_usage = {key: 0 for key in groq_api_keys}
last_reset_time = time.time()

# Store chat sessions globally
chat_sessions = {}

def reset_api_key_usage():
    global last_reset_time
    with lock:
        current_time = time.time()
        if current_time - last_reset_time >= 60:  # Reset every 60 seconds
            for key in api_key_usage:
                api_key_usage[key] = 0
            last_reset_time = current_time

def get_available_api_key():
    reset_api_key_usage()
    with lock:
        for key, usage in api_key_usage.items():
            if usage < 30:  # Check if usage is under the limit
                return key
        return None  # No available keys

def increment_api_key_usage(api_key):
    with lock:
        if api_key in api_key_usage:
            api_key_usage[api_key] += 1

def store_data_in_faiss(pdf_path, index_file):
    # Load and split the PDF document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)

    # Create embeddings and store them in FAISS
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(documents[:30], embeddings)
    with open(index_file, 'wb') as f:
        pickle.dump(db, f)
    return db

def answer_query(query, index_file, pdf_path, session_id):
    try:
        # Load the FAISS index from the pickle file
        try:
            with open(index_file, 'rb') as f:
                db = pickle.load(f)
                print("FAISS vector store loaded from .pkl file.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            db = store_data_in_faiss(pdf_path, index_file)
            print("FAISS vector store created and stored.")

        # Perform a similarity search
        result = db.similarity_search(query)
        context = result[0].page_content

        # Prepare the prompt for Groq Cloud API with session history
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        # Add the current question to the conversation history
        chat_sessions[session_id].append({"role": "user", "content": query})
        chat_sessions[session_id].append({"role": "assistant", "content": context})

        # Combine all previous interactions into the prompt
        conversation_history = "\n".join([f"{message['role']}: {message['content']}" for message in chat_sessions[session_id]])

        prompt = f"""
        You are an AI assistant specialized in providing precise information about events. Follow these guidelines strictly:

        1. Use ONLY the given context to answer the question.
        2. If the query is unrelated to the context, respond with: "I apologize, but I can only provide information based on the events in my context."
        3. Provide clear, concise, and easy-to-understand answers.
        4. Focus on extracting relevant event details such as:
        - Event name
        - Date and time
        - Location
        - Key participants or speakers
        - Any specific event-related information
        5. You can use emojis to enhance your answers, depending on the sentiment. For example:
   - 😀 for positive answers
   - 😔 for apologies or unavailability
   - 📅 for dates or events
   - 📍 for locations
   - 🗣️ for speakers or participants
    - 📝 for event-related information
       6. Keep it short and precise if possible(not too congested)
       7. Generate in markdown format
   

        Respond in a straightforward manner, using simple language that anyone can understand.


        Current system date and time: {current_date_time}

        Context: {context}

        Conversation History:
        {conversation_history}

        Question: {query}

        Your response should directly address the question using only the information available in the context.
        """
        
        print(prompt)

        # Get an available API key
        api_key = get_available_api_key()
        if not api_key:
            return "All API keys have reached their request limit. Please wait a moment and try again."

        # Initialize Groq client with the API key
        client = Groq(api_key=api_key)

        # Perform the chat completion using Groq Cloud
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model="llama-3.1-8b-instant"  # Use the model ID that you want to use in Groq
        )

        # Increment the usage count for the API key
        increment_api_key_usage(api_key)

        # Add assistant's response to the conversation history
        assistant_response = chat_completion.choices[0].message.content
        chat_sessions[session_id].append({"role": "assistant", "content": assistant_response})

        # Return the response
        return assistant_response
    except Exception as e:
        return f"An error occurred: {str(e)}"
