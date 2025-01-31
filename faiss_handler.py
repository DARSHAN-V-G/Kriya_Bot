import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")

def store_data_in_faiss(pdf_path, index_file):
    # Load and split the PDF document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    documents = text_splitter.split_documents(docs)
    
    # Create embeddings and store them in FAISS
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    with open(index_file, 'wb') as f:
        pickle.dump(db, f)
    return db

def answer_query(query, index_file, pdf_path):
    try:
        try:
            with open(index_file, 'rb') as f:
                db = pickle.load(f)
                print("FAISS vector store loaded from .pkl file.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            db = store_data_in_faiss(pdf_path, index_file)
            print("FAISS vector store loaded from .pkl file.")

        # Perform a similarity search
        result = db.similarity_search(query)
        print(result[0].page_content)
        
        if not result:
            return "I couldn't find any relevant information in the documents to answer your query."

        # Extract context from top 3 similar documents
        context = "\n".join([r.page_content for r in result[:3]])

        prompt = f"""
        You are an AI assistant specialized in providing precise information about events. Please follow these instructions strictly:

        1. Use ONLY the given context to answer the question.
        2. If the query is unrelated to the context, respond with: "I apologize, but I can only provide information based on the events in my context."
        3. Provide clear, concise, and easy-to-understand answers.
        4. If multiple events are mentioned in the context, extract the most relevant event details. 
        5. Focus on the following event details:
           - Event name
           - Date and time
           - Location
           - Key participants or speakers
           - Event-related information such as agenda or topics covered.

        Context: {context}

        Question: {query}

        Please give a direct response based on the context, using only the information available in the context.
        """
        print(prompt)
        # Initialize Gemini model
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"An error occurred: {str(e)}"
