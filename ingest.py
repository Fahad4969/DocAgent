"""
Build the vector store from course slide PDFs.

Expected layout:
    data/
      Machine Perception/
        lecture_01.pdf
      Sensor Network Technologies and Sensor Data Fusion/
        2025WS SDF13 KF.pdf

Each SLIDE becomes one chunk, tagged with course / file / page, so answers
can cite "Machine Perception, lecture_26.pdf, slide 8".

Usage:
    python ingest.py --dry-run     # parse and report, write nothing
    python ingest.py               # parse and build the vector store
"""

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

MIN_CHARS = 50          # slides with less text than this carry no signal
MAX_CHARS = 1500        # above this, split the slide into smaller chunks
BATCH_SIZE = 128        # how many chunks to embed per batch


# ---------------------------------------------------------------------------
# 1. Read every PDF, one entry per page
# ---------------------------------------------------------------------------

def read_pdfs():
    """Return [(course, filename, page_number, raw_text), ...] for all slides."""
    pages = []
    # Only look inside course sub-folders, so stray PDFs in data/ are ignored.
    for course_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        pdfs = sorted(course_dir.glob("*.pdf"))
        print(f"{course_dir.name}: {len(pdfs)} file(s)")
        for pdf in pdfs:
            doc = pymupdf.open(pdf)
            for page_number, page in enumerate(doc, start=1):
                pages.append((course_dir.name, pdf.name, page_number, page.get_text()))
            doc.close()
    return pages


# ---------------------------------------------------------------------------
# 2. Find boilerplate: the university header, the course footer, etc.
# ---------------------------------------------------------------------------

def find_boilerplate(pages, page_fraction=0.5, min_files=3):
    """
    Furniture = a line that is on MOST PAGES of a deck, in SEVERAL decks.

    Two conditions, both required. "Appears in many files" alone is not
    enough: your recap decks restate lecture content, so words like
    "Encoder" show up across many files and would be deleted. But a real
    header sits on nearly every page of its deck, while "Encoder" sits on
    a handful.
    """
    pages_per_file = Counter()                 # filename -> total pages
    line_pages = defaultdict(Counter)          # line -> (filename -> pages containing it)

    for _course, filename, _page_number, text in pages:
        pages_per_file[filename] += 1
        for line in {l.strip() for l in text.splitlines() if l.strip()}:
            if len(line) > 3:
                line_pages[line][filename] += 1

    boilerplate = set()
    for line, per_file in line_pages.items():
        # in how many files does this line cover at least half the pages?
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
        if stripped.isdigit():              # slide numbers
            continue
        kept.append(stripped)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# 3. Turn pages into chunks carrying metadata
# ---------------------------------------------------------------------------

def build_documents(pages, boilerplate):
    docs = []
    skipped = 0
    duplicates = 0
    seen = set()

    for course, filename, page_number, raw_text in pages:
        text = clean_page(raw_text, boilerplate)
        if len(text) < MIN_CHARS:           # title slides, pure images
            skipped += 1
            continue

        if text in seen:
            duplicates += 1
            continue
        seen.add(text)

        docs.append(
            Document(
                page_content=text,
                # This metadata is what lets an answer cite its source, and
                # what would let you filter to a single course later.
                metadata={
                    "course": course,
                    "source": filename,
                    "page": page_number,
                },
            )
        )

    print(f"Skipped {skipped} slide(s) with under {MIN_CHARS} characters")
    print(f"Removed {duplicates} exact duplicate slide(s)")

    # Most slides are already the right size, so we do NOT split by default.
    # Only the rare wall-of-text slide gets cut up; split_documents copies
    # the metadata onto every piece it produces.
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


# ---------------------------------------------------------------------------
# 4. Embed and store
# ---------------------------------------------------------------------------

def build_vectorstore(docs):
    # Imported here, not at the top of the file, so that --dry-run starts
    # instantly instead of loading the whole embedding stack first.
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    # Chroma APPENDS. Rebuilding from scratch is the only way to avoid
    # silently accumulating duplicate copies of every chunk.
    if Path(DB_DIR).exists():
        print(f"Removing old '{DB_DIR}'")
        shutil.rmtree(DB_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Embedding runs on your CPU and takes a while, so go in batches and
    # print progress rather than staring at a frozen terminal.
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        db.add_documents(batch)
        print(f"  embedded {min(start + BATCH_SIZE, len(docs))}/{len(docs)}")

    print(f"Vector store built at '{DB_DIR}'")
    return db


# ---------------------------------------------------------------------------

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
