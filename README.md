# docuagent

A question-answering assistant over my own university lecture slides. You ask a question in plain English, it finds the relevant slides with vector search and has an LLM answer from those slides only, citing the file and slide number for every claim.

I built it to study for two Master's courses at TH Ingolstadt (Machine Perception, and Sensor Network Technologies and Sensor Data Fusion). The interesting part turned out not to be the LLM call but the data: getting clean, well-cited chunks out of 57 lecture PDFs and measuring whether retrieval actually finds the right slide.

```
python ask.py "Why does a masked autoencoder hide most of the image patches?" --course machine
```

## How it works

```
57 PDFs ──► one chunk per slide ──► strip headers/footers ──► drop duplicates ──► embed ──► Chroma
                                                                                            │
question ──► embed ──► top-6 similar slides (optionally filtered by course) ──► LLM ──► answer + [1][2] citations
```

| Step | Choice | Why |
|---|---|---|
| Parsing | PyMuPDF, one page = one chunk | A slide is already one topic. Fixed 800-character chunks (my first version) cut slides in half and mixed topics. It also gives exact citations for free. |
| Cleaning | Remove lines that sit on ≥50% of a deck's pages in ≥3 decks | Strips repeated headers and lecturer names without deleting real terms that happen to appear in several lectures. |
| Short slides | Skip slides under 50 characters | Title and image-only slides carry no searchable text (235 slides, about 9%). |
| Deduplication | Drop slides whose cleaned text is an exact copy | Recap decks and re-uploaded lectures repeated 569 slides (23%). See evaluation below. |
| Embeddings | `all-MiniLM-L6-v2`, runs locally on CPU | Free and fast. It truncates at 256 tokens (about 1,000 characters); the longest slide is 998, so nothing is cut. |
| Vector store | Chroma, rebuilt from scratch on each ingest | Chroma appends, so rebuilding is the simplest way to avoid duplicate vectors. |
| Metadata | `course`, `source`, `page` on every chunk | Used for citations and for the `--course` filter. |
| Generation | `openai/gpt-oss-120b` via Groq | The system prompt restricts the model to the numbered passages, asks it to cite them and to say so when the slides don't contain the answer. |

## Corpus

| | |
|---|---|
| Courses | 2 |
| PDF files | 57 |
| Slides read | 2,678 |
| Skipped (under 50 characters) | 235 |
| Exact duplicates removed | 569 |
| Chunks in the index | 1,874 (Machine Perception 1,152, Sensor Fusion 722) |
| Chunk length | min 50, median 330, max 998 characters |

The slides themselves are not in this repository because they belong to the lecturers.

## Evaluation

`eval_retrieval.py` tests retrieval on its own, without calling the LLM: if the right slide is not retrieved, no prompt can produce a grounded answer.

- `eval_questions.csv` has 20 questions, 10 per course, written in student wording rather than copied from the slides. Each answer slide was checked by hand.
- When a slide exists in several files, every copy is listed as correct. That keeps before/after comparisons fair when deduplication changes which copy is stored.
- Metrics: **hit@k** (is a correct slide in the top k?) and **MRR** (average of 1/rank, rewarding correct slides near the top). One search with k=10 gives all of them, and no LLM is involved, so runs are free and repeatable.

### Results

| | Baseline | After deduplication |
|---|---|---|
| Chunks | 2,443 | 1,874 |
| hit@1 | 40% | 40% |
| hit@3 | 50% | 50% |
| hit@6 (used by `ask.py`) | 70% | 75% |
| hit@10 | 85% | 90% |
| MRR | 0.507 | 0.524 |
| Distinct slides in top 6 | 5.0 | 6.0 |

With 20 questions, one question is 5 percentage points, so the hit@k gain is one more question found. The clearer effect is the last row: before deduplication, the top 6 contained on average one repeated slide, so the LLM saw five different slides instead of six. In one baseline miss, the top 3 results were the same slide from three different files.

### What still fails

- **Similar topics.** "How does a decoder stop a position from looking at later tokens?" retrieves the neighbouring self-attention and cross-attention slides instead of the masked self-attention one. A cross-encoder reranker or hybrid keyword + vector search should help here.
- **Formulas are lost.** PyMuPDF only reads the text layer, so equations disappear. The Mahalanobis-distance slide is mostly equations, and too little text is left to retrieve it.
- **Ranking.** The right slide is in the top 10 for 90% of questions but first for only 40%. That gap is what a reranker addresses.

## Limitations

- The system prompt asks the model to stay grounded, but that is an instruction, not a guarantee. Answers can still blend in outside knowledge or cite a passage that doesn't fully support the claim.
- 20 evaluation questions is enough to spot problems, not to make fine-grained claims.
- Only exact duplicates are removed. Near-copies, such as slides that were lightly edited between two versions of a lecture, remain.
- Every ingest re-embeds the whole corpus.

## What I would do next

1. Add a cross-encoder reranker over the top 20 and measure hit@1 again.
2. Try hybrid search (BM25 + vectors) for exact technical terms.
3. Read formulas with OCR or a vision-language model looking at the slide image.
4. Evaluate the answers too, not just retrieval: is each claim supported by the cited slide?
5. Index incrementally by hashing files, so only changed PDFs are re-embedded.

## Running it

```
pip install -r requirements.txt
echo GROQ_API_KEY=your_key > .env

# put PDFs in data/<Course name>/*.pdf
python ingest.py --dry-run     # parse and report, write nothing
python ingest.py               # build the vector store
python eval_retrieval.py       # retrieval metrics + misses
python ask.py "your question" [--course NAME] [-k 6] [--show-context]
```

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Read PDFs, clean, deduplicate, embed, store |
| `ask.py` | Retrieve slides and generate a cited answer |
| `eval_retrieval.py` | Measure hit@k and MRR on the question set |
| `eval_questions.csv` | 20 questions with their answer slides |
| `results_baseline.txt`, `results_dedup.txt` | Evaluation output before and after deduplication |
