from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pickle
load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")
def store_data_in_faiss(pdf_path,index_file):
    # Load and split the PDF document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)

    # Create embeddings and store them in FAISS
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(documents[:30], embeddings)
    with open(index_file, 'wb') as f:
        pickle.dump(db, f)
    return db


def answer_query(query,index_file,pdf_path):
    try:
        try:
            with open(index_file, 'rb') as f:
                db = pickle.load(f)
                print("FAISS vector store loaded from .pkl file.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            db = store_data_in_faiss(pdf_path,index_file)
            print("FAISS vector store loaded from .pkl file.")
        # Perform a similarity search
        result = db.similarity_search(query)
        context = result[0].page_content
        prompt = f"""
        You are an AI assistant specialized in providing precise information about events. Follow these guidelines strictly:

        1. Use ONLY the given context to answer the question
        2. If the query is unrelated to the context, respond with: "I apologize, but I can only provide information based on the events in my context."
        3. Provide clear, concise, and easy-to-understand answers
        4. Focus on extracting relevant event details such as:
        - Event name
        - Date and time
        - Location
        - Key participants or speakers
        - Any specific event-related information

        Respond in a straightforward manner, using simple language that anyone can understand.

        Context: {context}

        Question: {query}

        Your response should directly address the question using only the information available in the context.
        """
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"
