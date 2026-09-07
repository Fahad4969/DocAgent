from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DB_DIR = "chroma_db"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

questions = [
    "What programming languages does Fahad know?",
    "What ML frameworks and tools does Fahad use?",

]

for question in questions:
    results = db.similarity_search(question, k=6)
    print(f"Question: {question}\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(doc.page_content[:400])  # truncate if long
        print()
    print("-" * 40)