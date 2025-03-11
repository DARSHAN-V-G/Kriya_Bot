import eventlet
eventlet.monkey_patch()
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import uuid
import requests
import os
from dotenv import load_dotenv
from server import answer_query, store_data_in_faiss

load_dotenv()

# Initialize Flask app, CORS, and SocketIO
app = Flask(__name__)
CORS(app, resources={r"/": {"origins": ""}})
socketio = SocketIO(app, cors_allowed_origins="*")

# WhatsApp API Credentials (Replace with actual values)
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")

# Store user sessions in memory
chat_sessions = {}

# Handle WhatsApp Webhook (Receiving Messages)
@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        return request.args.get("hub.challenge")

    data = request.json
    if "messages" in data["entry"][0]["changes"][0]["value"]:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender_id = message["from"]
        text = message["text"]["body"]

        # Emit message via WebSocket to connected clients
        socketio.emit("whatsapp_message", {"sender": sender_id, "message": text})

        # Process message with chatbot
        response = answer_query(text, "vector_db.pkl", "kriya_events.pdf", sender_id)

        # Send chatbot response to WhatsApp
        send_whatsapp_message(sender_id, response)

    return jsonify({"status": "received"}), 200

# Send Messages to WhatsApp
def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"WhatsApp API Error: {response.text}")
    return response.json()



@socketio.on("connect")
def on_connect():
    print("Client connected to WebSocket")




# WebSocket Event: Receive messages from the frontend
@socketio.on("send_whatsapp_response")
def handle_send_whatsapp(data):
    sender_id = data.get("sender_id")
    message = data.get("message")

    if sender_id and message:
        send_whatsapp_message(sender_id, message)

# WebSocket Event: Handle website queries



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