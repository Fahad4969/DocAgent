import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

MIN_CHARS = 50          
MAX_CHARS = 1500        
BATCH_SIZE = 128        



def read_pdfs():
    pages = []
    for course_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        pdfs = sorted(course_dir.glob("*.pdf"))
        print(f"{course_dir.name}: {len(pdfs)} file(s)")
        for pdf in pdfs:
            doc = pymupdf.open(pdf)
            for page_number, page in enumerate(doc, start=1):
                pages.append((course_dir.name, pdf.name, page_number, page.get_text()))
            doc.close()
    return pages



def find_boilerplate(pages, page_fraction=0.5, min_files=3):

    pages_per_file = Counter()                 
    line_pages = defaultdict(Counter)        

    for _course, filename, _page_number, text in pages:
        pages_per_file[filename] += 1
        for line in {l.strip() for l in text.splitlines() if l.strip()}:
            if len(line) > 3:
                line_pages[line][filename] += 1

    boilerplate = set()
    for line, per_file in line_pages.items():
        dominant_in = sum(
            1 for filename, count in per_file.items()
            if count >= page_fraction * pages_per_file[filename]
        )
        if dominant_in >= min_files:
            boilerplate.add(line)
    return boilerplate

def clean_page(text, boilerplate):
    """Drop boilerplate lines, bare page numbers and blank lines."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in boilerplate:
            continue
        if stripped.isdigit():            
            continue
        kept.append(stripped)
    return "\n".join(kept)


def build_documents(pages, boilerplate):
    docs = []
    skipped = 0
    duplicates = 0
    seen = set()

    for course, filename, page_number, raw_text in pages:
        text = clean_page(raw_text, boilerplate)
        if len(text) < MIN_CHARS:          
            skipped += 1
            continue

        if text in seen:
            duplicates += 1
            continue
        seen.add(text)

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "course": course,
                    "source": filename,
                    "page": page_number,
                },
            )
        )

    print(f"Skipped {skipped} slide(s) with under {MIN_CHARS} characters")
    print(f"Removed {duplicates} exact duplicate slide(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    small = [d for d in docs if len(d.page_content) <= MAX_CHARS]
    large = [d for d in docs if len(d.page_content) > MAX_CHARS]
    if large:
        print(f"Splitting {len(large)} oversized slide(s)")
        small.extend(splitter.split_documents(large))

    return small


def build_vectorstore(docs):
    
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    if Path(DB_DIR).exists():
        print(f"Removing old '{DB_DIR}'")
        shutil.rmtree(DB_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        db.add_documents(batch)
        print(f"  embedded {min(start + BATCH_SIZE, len(docs))}/{len(docs)}")

    print(f"Vector store built at '{DB_DIR}'")
    return db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and report, but don't build the vector store")
    args = parser.parse_args()

    pages = read_pdfs()
    print(f"\nRead {len(pages)} slide(s) total")

    boilerplate = find_boilerplate(pages)
    print(f"\nStripping {len(boilerplate)} boilerplate line(s) — CHECK THESE:")
    for line in sorted(boilerplate)[:20]:
        print(f"  - {line[:100]!r}")

    docs = build_documents(pages, boilerplate)
    print(f"\n{len(docs)} chunk(s) ready")

    per_course = Counter(d.metadata["course"] for d in docs)
    for course, count in per_course.items():
        print(f"  {course}: {count}")

    lengths = sorted(len(d.page_content) for d in docs)
    print(f"chunk length: min={lengths[0]} median={lengths[len(lengths)//2]} max={lengths[-1]}")

    print("\n--- example chunk ---")
    example = docs[len(docs) // 2]
    print(example.metadata)
    print(example.page_content[:400])

    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    build_vectorstore(docs)


if __name__ == "__main__":
    main()
