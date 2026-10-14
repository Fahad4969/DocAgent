# docuagent

A study assistant for my university lecture slides. You ask a question, it finds the most relevant slides and an LLM answers using only those slides, with the file and slide number for each source.

Built for two Master's courses at TH Ingolstadt: Machine Perception, and Sensor Network Technologies and Sensor Data Fusion.

```
python ask.py "What is a masked autoencoder?"
```

## How it works

1. **Ingest** (`ingest.py`): reads 57 lecture PDFs, makes one chunk per slide, removes repeated headers, empty slides and duplicate slides, then embeds each slide with `all-MiniLM-L6-v2` and stores it in Chroma.
2. **Ask** (`ask.py`): embeds the question, retrieves the 6 closest slides (optionally from one course with `--course`), and sends them to an LLM via Groq with the instruction to answer only from those slides and cite them.
3. **Evaluate** (`eval_retrieval.py`): checks on 20 test questions whether the correct slide is retrieved.

One slide per chunk works better than fixed-size chunks: each slide is already one topic, and it gives exact citations.

## Data

57 PDFs, 2,678 slides. 235 nearly empty slides and 569 duplicate slides (from recap lectures) are removed, leaving **1,874 chunks**. The slides are not included in this repository because they belong to the lecturers.

## Evaluation

20 questions (10 per course) with the slide that answers each one. **hit@k** = how often the correct slide is in the top k results. **MRR** rewards finding it near the top.

| | Before deduplication | After |
|---|---|---|
| hit@1 | 40% | 40% |
| hit@6 (used by `ask.py`) | 70% | 75% |
| hit@10 | 85% | 90% |
| MRR | 0.507 | 0.524 |
| Different slides in top 6 | 5.0 | 6.0 |

Before deduplication, the same slide sometimes filled several of the top results. Removing duplicates means the LLM now sees 6 different slides instead of 5. With only 20 questions, one question equals 5 percentage points, so these numbers show direction, not precision.

## Limitations and next steps

- Formulas are lost when extracting text from PDFs, so equation-heavy slides are hard to find.
- The right slide is usually in the top 10 but often not first. A reranker could fix this.
- The prompt asks the model to stay grounded in the slides, but that is not a guarantee. Next step: check that each claim is supported by the slide it cites.

## Run it

```
pip install -r requirements.txt
echo GROQ_API_KEY=your_key > .env
# put PDFs in data/<Course name>/*.pdf

python ingest.py
python eval_retrieval.py
python ask.py "your question" --course machine
```
