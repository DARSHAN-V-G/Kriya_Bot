import pickle

# Load the existing kriya_data.pkl file
with open("vector_db.pkl", "rb") as f:
    kriya_data = pickle.load(f)

# Extract required data
event_names = kriya_data.get("event_names", [])
workshops = kriya_data.get("workshops", [])
paper_presentations = kriya_data.get("paper_presentations", [])
event_details = kriya_data.get("event_details", [])
print(paper_presentations)

# Function to chunk text manually
def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks of a given size with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks

# Initialize chunks list
chunks = []

# Add event names, workshops, and paper presentation names as separate chunks
chunks.append({"text": "List of Event Names: " + ", ".join(event_names), "metadata": {"type": "event_names"}})
chunks.append({"text": "List of Workshops: " + ", ".join(workshops), "metadata": {"type": "workshops"}})
chunks.append({"text": "List of Paper Presentations: " + ", ".join(paper_presentations), "metadata": {"type": "paper_presentations"}})

# Process detailed event descriptions
for event in event_details:
    text = (
        f"Event Name: {event['eventName']}\nCategory: {event['category']}\nDescription: {event['description']}"
        f"\nLocation: {event['location']}\nDate: {event['date']}\nTiming: {event['timing']}"
    )
    chunked_texts = chunk_text(text)

    for chunk in chunked_texts:
        chunks.append({"text": chunk, "metadata": {"type": "event_details", "event_name": event['eventName']}})

# Save the processed chunks into a new .pkl file for FAISS
with open("kriya_faiss.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Chunked data saved to kriya_faiss.pkl")