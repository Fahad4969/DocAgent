"""
Measure retrieval quality: for each question in eval_questions.csv, is the
slide that answers it among the top-k search results, and at which rank?

No LLM is called, so this is free, fast and gives the same result every run.

Usage:
    python eval_retrieval.py              # print metrics + misses
    python eval_retrieval.py --verbose    # also print the rank for every question
"""

import argparse
import csv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Same constants as ingest.py and ask.py, so we test exactly what ask.py uses.
from ingest import DB_DIR, EMBED_MODEL

EVAL_FILE = "eval_questions.csv"
MAX_K = 10                   # one search with k=10 gives hit@1/3/6/10 at once
REPORT_KS = [1, 3, 6, 10]    # 6 is ask.py's default


def load_questions(path):
    """
    Each row: question, gold.
    gold lists every copy of the answer slide, e.g.
        "lecture_26.pdf:86;lecture_27.pdf:37"
    Split on the LAST ':' so only the page number is cut off the file name.
    """
    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold = set()
            for item in row["gold"].split(";"):
                source, page = item.strip().rsplit(":", 1)
                gold.add((source, int(page)))
            questions.append((row["question"], gold))
    return questions


def first_hit_rank(docs, gold):
    """1-based rank of the first retrieved slide that is in gold, or None."""
    for rank, doc in enumerate(docs, start=1):
        if (doc.metadata["source"], doc.metadata["page"]) in gold:
            return rank
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print every question's rank")
    args = parser.parse_args()

    questions = load_questions(EVAL_FILE)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    ranks = []
    misses = []
    distinct_in_top6 = []

    for question, gold in questions:
        docs = db.similarity_search(question, k=MAX_K)
        rank = first_hit_rank(docs, gold)
        ranks.append(rank)

        # How many DIFFERENT slide texts did the top 6 contain?
        # Duplicate slides in the corpus waste top-k slots.
        distinct_in_top6.append(len({d.page_content for d in docs[:6]}))

        if args.verbose:
            print(f"{'miss' if rank is None else f'rank {rank}':>7}  {question}")
        if rank is None:
            misses.append((question, gold, docs))

    n = len(questions)
    print(f"\nRetrieval evaluation on {n} questions ({EMBED_MODEL})\n")
    for k in REPORT_KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  hit@{k:<2}  {hits:>2}/{n}  = {hits / n:.0%}")

    # MRR: average of 1/rank, counting a miss as 0.
    # Rewards finding the right slide near the top, not just somewhere in the top 10.
    mrr = sum(1 / r for r in ranks if r is not None) / n
    print(f"  MRR      {mrr:.3f}")

    avg_distinct = sum(distinct_in_top6) / n
    print(f"  distinct slides in top 6: {avg_distinct:.1f} on average")

    if misses:
        print(f"\n--- {len(misses)} miss(es): what was retrieved instead (top 3) ---")
        for question, gold, docs in misses:
            print(f"\nQ: {question}")
            expected = " or ".join(f"{s}, slide {p}" for s, p in sorted(gold))
            print(f"   expected: {expected}")
            for d in docs[:3]:
                first_line = d.page_content.splitlines()[0][:70]
                print(f"   got: {d.metadata['source']}, slide {d.metadata['page']}  | {first_line}")


if __name__ == "__main__":
    main()
