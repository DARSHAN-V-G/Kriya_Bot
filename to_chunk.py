import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # or use a similar embedding class

# Step 1: Load the manually parsed data
with open("manual.pkl", "rb") as f:
    manual_data = pickle.load(f)

# Assume your manual_data dictionary has keys like "event_details" or "event_names"
# and you want to build a vector store over the detailed event descriptions.
# For example, if "event_details" is a list of dicts with a "description" field:
documents = []
for event in manual_data.get("event_details", []):
    # Create a document string from the event's details (you can customize this)
    doc = f"{event.get('eventName', '')}. {event.get('description', '')}"
    documents.append(doc)

# Step 2: Create embeddings (adjust model as necessary)
embeddings = HuggingFaceEmbeddings()  # or use SentenceTransformer embeddings wrapper

# Step 3: Build the FAISS vector store from your documents
faiss_store = FAISS.from_texts(documents, embeddings)

# Step 4: Save the FAISS vector store for later use
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(faiss_store, f)

print("FAISS vector store built and saved to faiss_store.pkl")