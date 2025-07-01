from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil

VECTOR_STORE_DIR = "vector_store"

# ✅ Load SentenceTransformer model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Split text into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": "uploaded"}) for chunk in chunks]
    return docs

# ✅ Store document embeddings
def store_embeddings(text, vector_dir=VECTOR_STORE_DIR):
    docs = chunk_text(text)

    # 🔄 Remove old vector store if exists (to avoid errors/duplication)
    if os.path.exists(vector_dir):
        shutil.rmtree(vector_dir)

    # 🧠 Build new FAISS index and save
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(vector_dir)
    print(f"✅ FAISS index saved to: {vector_dir}")

# ✅ Load existing FAISS vector DB
def load_vector_db(vector_dir=VECTOR_STORE_DIR):
    index_path = os.path.join(vector_dir, "index.faiss")
    if not os.path.exists(index_path):
        print("⚠️ No FAISS vector DB found.")
        return None
    return FAISS.load_local(vector_dir, embeddings, allow_dangerous_deserialization=True)

# ✅ Retrieve most relevant chunks based on query
def retrieve_relevant_chunks(query, k=3):
    db = load_vector_db()
    if not db:
        return []
    return db.similarity_search(query, k=k)
