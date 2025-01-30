import eventlet
eventlet.monkey_patch()
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import uuid
from dotenv import load_dotenv
from server import answer_query, store_data_in_faiss

load_dotenv()

# Initialize Flask app, CORS, and SocketIO
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Store user sessions in memory
chat_sessions = {}

# Generate a unique session ID
def get_session_id():
    return str(uuid.uuid4())

@socketio.on('query')
def handle_query(data):
    try:
        query_text = data.get('query', None)
        session_id = data.get('session_id', None)

        if not query_text:
            emit('response', {"error": "No query provided"})
            return

        if not session_id:
            session_id = get_session_id()

        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        chat_sessions[session_id].append({'role': 'user', 'content': query_text})

        # Process query and get results
        results = answer_query(query_text, "vector_db.pkl", "kriya_events.pdf", session_id)

        # Store response in chat history
        chat_sessions[session_id].append({'role': 'assistant', 'content': results})

        # Send response to client
        emit('response', {"response": results, "session_id": session_id})

    except Exception as e:
        emit('response', {"error": str(e)})

@socketio.on('store')
def handle_store():
    try:
        
        store_data_in_faiss("kriya_events.pdf", "vector_db.pkl")
        emit('store_response', {"message": "Text data successfully stored!"})
    except Exception as e:
        emit('store_response', {"error": str(e)})



if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
