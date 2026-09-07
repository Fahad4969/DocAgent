import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DATA_DIR = "data"
DB_DIR = "chroma_db"

def load_documents():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} document page(s)")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks

def build_vectorstore(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    db.persist()
    print(f"Vector store built and saved to '{DB_DIR}'")
    return db

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)