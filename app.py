from flask import Flask, request, jsonify 
from faiss_handler import store_data_in_faiss, answer_query 
from dotenv import load_dotenv 

load_dotenv() 
pdf_path = "events.pdf"
index_file = "vector_db.pkl"


app = Flask(__name__) 
 
@app.route('/query', methods=['POST']) 
def query(): 
    try: 
        query_text = request.json.get('query', None) 
         
        if not query_text: 
            return jsonify({"error": "No query provided"}), 400 
    
        results = answer_query(query_text,index_file,pdf_path) 
        print(f"Results : {results}") 
        # Return the search results as JSON response 
        return jsonify(results), 200 
    except Exception as e: 
        return jsonify({"error": str(e)}), 500 
 
@app.route('/store', methods=['GET']) 
def store(): 
    try: 
        # Store the text data in Faiss only if not already stored
        store_data_in_faiss(pdf_path,index_file)
        return jsonify({"message": "Text data successfully stored!"}), 200 
    except Exception as e: 
        return jsonify({"error": str(e)}), 500 
     
 
if __name__ == '__main__': 
    app.run(debug=True)
