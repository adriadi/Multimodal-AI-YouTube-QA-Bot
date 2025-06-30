import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Wraps each chunk in LangChain Document
def convert_to_documents(chunk_docs: list) -> list:
    return [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunk_docs
    ]

# Embeds with OpenAI
def embed_documents_with_openai(documents: list, model: str = "text-embedding-3-small"):
    embedding_model = OpenAIEmbeddings(model=model)
    db = FAISS.from_documents(documents, embedding_model)
    return db

# Saves FAISS db
def save_vectorstore_faiss(db, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    db.save_local(path)

# Loads FAISS db later
def load_vectorstore_faiss(path: str):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)