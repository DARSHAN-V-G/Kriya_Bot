from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
from dotenv import load_dotenv
from server import answer_query, store_data_in_faiss

load_dotenv()

# Initialize PDF and index file paths
pdf_path = "events.pdf"
index_file = "vector_db.pkl"

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Change "*" to your frontend domain for security

# Store user sessions in memory (chat history per session)
chat_sessions = {}

# Helper function to manage session
def get_session_id():
    return str(uuid.uuid4())  # Generates a unique session ID for each user

@app.route('/query', methods=['POST'])
def query():
    try:
        query_text = request.json.get('query', None)
        session_id = request.json.get('session_id', None)

        if not query_text:
            return jsonify({"error": "No query provided"}), 400

        if not session_id:
            session_id = get_session_id()

        # Make sure each session has a chat history
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        # Add the current query to the session's chat history
        chat_sessions[session_id].append({'role': 'user', 'content': query_text})

        # Pass the session's chat history to the answer_query function
        results = answer_query(query_text, index_file, pdf_path, session_id)

        # Append the response to the session's chat history
        chat_sessions[session_id].append({'role': 'assistant', 'content': results})

        # Return the response as JSON
        return jsonify({"response": results, "session_id": session_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/store', methods=['GET'])
def store():
    try:
        # Store the text data in Faiss only if not already stored
        store_data_in_faiss(pdf_path, index_file)
        return jsonify({"message": "Text data successfully stored!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Ensure the app listens on all interfaces
