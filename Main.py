from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text
import os
import pickle
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = Flask(__name__)
vectorstore_openai = None
file_path = "faiss_store_openai.pkl"

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

@app.route('/upload', methods=['POST'])
def upload_file():
    global vectorstore_openai
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    uploaded_file = request.files['file']
    uploaded_file.save(uploaded_file.filename)
    try:
        extracted_text = extract_text(uploaded_file.filename)
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
    finally:
        os.remove(uploaded_file.filename)
    if not extracted_text:
        return jsonify({"error": "No text extracted from the PDF"}), 400
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(extracted_text)
    if not text_chunks:
        return jsonify({"error": "No text chunks generated from the document"}), 400
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vectorstore_openai = FAISS.from_texts(text_chunks, embeddings)
    except Exception as e:
        return jsonify({"error": f"Failed to create embeddings: {str(e)}"}), 500
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    return jsonify({"message": "File uploaded and processed successfully."})

@app.route('/ask', methods=['POST'])
def ask_question():
    global vectorstore_openai
    if not vectorstore_openai:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore_openai = pickle.load(f)
        else:
            return jsonify({"error": "No processed documents found. Please upload a file first."}), 400
    query = request.json.get('question', None)
    if not query:
        return jsonify({"error": "No question provided"}), 400
    chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore_openai.as_retriever())
    try:
        answer = chain.run(query)
    except Exception as e:
        return jsonify({"error": f"Failed to process the query: {str(e)}"}), 500
    return jsonify({"question": query, "answer": answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
