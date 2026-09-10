import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DB_DIR = "chroma_db"
MODEL = "openai/gpt-oss-120b"
K = 6

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

question = "What is Fahad's favourite programming language?"

docs = db.similarity_search(question, k=K)

blocks = []
for i, doc in enumerate(docs, start=1):
    blocks.append(f"[{i}] {doc.page_content}")
context = "\n\n".join(blocks)

system_prompt = (
    "You answer questions using only the numbered context passages provided. "
    "Do not use any outside knowledge. "
    "If the passages do not contain the answer, say so plainly instead of guessing. "
    "Cite the passages you used like [1] or [2][4]. "
    "Keep the answer short."
)
client = Groq()

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ],
)

print(response.choices[0].message.content)