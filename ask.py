import argparse
import os

from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
# Import the constants from ingest.py rather than retyping them.
# If the embedding model in ingest.py ever changes, this file follows
# automatically — searching one model's vectors with another model's
# query returns nonsense, silently, with no error.
from ingest import DB_DIR, EMBED_MODEL

MODEL = "openai/gpt-oss-120b"
K = 6

SYSTEM_PROMPT = (
    "You are a study assistant. Answer using only the numbered context passages, "
    "which are slides from the student's university courses. "
    "Do not use outside knowledge. "
    "If the passages do not contain the answer, say so plainly instead of guessing. "
    "Cite the passages you used like [1] or [2][4]. "
    "Be concise and precise with technical terms."
)


def describe(doc):
    """Short source label: 'Machine Perception / lecture_26.pdf, slide 11'."""
    m = doc.metadata
    return f"{m['course']} / {m['source']}, slide {m['page']}"


def main():
    parser = argparse.ArgumentParser(description="Ask a question about your course slides.")
    parser.add_argument("question", nargs="+", help="the question to ask")
    parser.add_argument("-k", type=int, default=K, help=f"slides to retrieve (default {K})")
    parser.add_argument("--course", help="restrict the search to one course (partial name is enough)")
    parser.add_argument("--show-context", action="store_true", help="print the retrieved slides too")
    args = parser.parse_args()

    question = " ".join(args.question)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Metadata filtering: this is what the course tag on every chunk buys you.
    # Chroma only matches exact values, so resolve a partial name first.
    where = None
    if args.course:
        courses = {m["course"] for m in db.get(include=["metadatas"])["metadatas"]}
        matches = [c for c in courses if args.course.lower() in c.lower()]
        if len(matches) != 1:
            raise SystemExit(f"--course matched {matches or 'nothing'}; be more specific. Available: {sorted(courses)}")
        where = {"course": matches[0]}

    docs = db.similarity_search(question, k=args.k, filter=where)
    if not docs:
        raise SystemExit("Nothing matched that question.")

    context = "\n\n".join(
        f"[{i}] ({describe(d)})\n{d.page_content}" for i, d in enumerate(docs, start=1)
    )

    if args.show_context:
        print(context, "\n" + "-" * 60)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    print(response.choices[0].message.content)
    print("\nSources:")
    for i, d in enumerate(docs, start=1):
        print(f"  [{i}] {describe(d)}")


if __name__ == "__main__":
    main()