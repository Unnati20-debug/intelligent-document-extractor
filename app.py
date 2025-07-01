from flask import Flask, render_template, request, jsonify 
import google.generativeai as genai
import os
from prompts import get_prompt_template  # 👈 Import the prompt selector
from extractor import extract_text_from_file  # OCR/Text extraction logic
from rag_engine import store_embeddings, retrieve_relevant_chunks, chunk_text # LangChain RAG logic

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Configure Gemini API Key
genai.configure(api_key="")  # 🔐 Replace with your actual key
model = genai.GenerativeModel("gemini-2.0-flash")

# ---------------------------
# 📄 Home Page
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------------------
# 💬 Chat Endpoint (RAG-based)
# ---------------------------
@app.route('/chat', methods=['POST'])
def chat_response():
    user_input = request.json['message']
    try:
        # 🔍 Get context from FAISS vector store
        docs = retrieve_relevant_chunks(user_input)
        context = "\n\n".join([doc.page_content for doc in docs]) or "No relevant context found."

        # 🧠 Use RAG to generate final response
        prompt = f"Use the following document context to answer:\n\n{context}\n\nQuestion: {user_input}"
        response = model.generate_content(prompt)

        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'response': f"❌ Error: {str(e)}"})

# ---------------------------
# 📤 Upload + Process File
# ---------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'response': '❌ No file uploaded.'})

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 1️⃣ Extract raw text
        text = extract_text_from_file(filepath)

        if not text.strip():
            return jsonify({'response': '⚠️ The uploaded document appears empty or unreadable.'})

        # 2️⃣ Create vector embeddings + store in FAISS
        print("Extracted text (preview):", text[:300])

        store_embeddings(text)
        docs = chunk_text(text)
        print(f"🧩 Chunks created: {len(docs)}")
        print(f"🧩 First chunk preview: {docs[0].page_content[:200]}")

        print("Embedding stored ✅")

        # 3️⃣ Optional: Ask Gemini for a summary
        doc_type = request.form.get("doc_type", "general").lower()  # get type from frontend
        prompt = get_prompt_template(doc_type, text)
        #prompt = f"Here is the uploaded document content:\n\n{text[:4000]}\n\nPlease summarize or extract key insights."
        response = model.generate_content(prompt)
        

        return jsonify({'response': response.text})

    except Exception as e:
        return jsonify({'response': f"❌ Error processing file: {str(e)}"})

# ---------------------------
# 🚀 Start the Server
# ---------------------------
if __name__ == '__main__':
    print("✅ Starting Flask App with LangChain RAG...")
    app.run(debug=True)
