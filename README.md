# Immigration RAG

Ask questions about USCIS immigration forms and get answers grounded in the
actual instruction documents, with citations.

I built this while going through the K-1 fiancé visa process. The USCIS
instructions are long, cross-referenced PDFs (the I-485 instructions alone run
40+ pages), and finding the one paragraph that answers your question means
reading all of them. This indexes the forms and answers the question directly,
pointing back to the source.

<!-- TODO: rewrite the two paragraphs above in your own words / fix any details
I got wrong about your situation. -->

**Live demo:** <!-- paste the Render URL once deployed -->

```bash
curl -X POST https://<your-app>.onrender.com/query \
  -H 'content-type: application/json' \
  -d '{"question": "Can the I-864 be waived?"}'
```

## Evaluation

The eval set is 24 questions scored with [RAGAS](https://docs.ragas.io/) on four
metrics: faithfulness, answer relevancy, context precision, context recall.

The two columns differ only in retrieval: the baseline is vector search alone,
the hybrid column adds BM25 keyword fusion. Chunking, the model, the prompt, and
reranking are identical across both (reranking is turned off for this table), so
the difference is attributable to retrieval.

| Metric            | Baseline (vector) | Hybrid (vector + BM25) | delta |
| ----------------- | ----------------- | ---------------------- | ----- |
| Faithfulness      | 0.95              | 0.97                   | +0.01 |
| Answer relevancy  | 0.78              | 0.83                   | +0.04 |
| Context precision | 0.92              | 0.94                   | +0.02 |
| Context recall    | 0.92              | 0.93                   | +0.02 |

The gains are modest but consistent. Breaking the set down by question type shows
where hybrid actually helps: on conceptual questions context precision rose +0.08
(0.91 -> 0.99) and recall +0.05, while on exact-token questions (form numbers
like "I-864") the two were within noise. The embedding model already handles form
numbers well; BM25's lift shows up on questions where keyword overlap helps. See
[docs/eval-optimization-log.md](docs/eval-optimization-log.md) for the full set
of experiments.

The deployed system adds an LLM reranker on top of hybrid retrieval. Against the
vector baseline, the full system (hybrid + reranking together) scores answer
relevancy 0.83 (up from 0.78), context precision 0.97 (up from 0.92), context
recall 0.96 (up from 0.92), and faithfulness 0.93 (down from 0.95). The reranker
changes individual rankings more than it changes the averages: on one question it
moved the answering chunk from rank 16 to rank 1 (see the optimization log).

## Architecture decisions

**pgvector instead of a dedicated vector DB.** The corpus is small (a few
hundred chunks) and already lives in Postgres-shaped infrastructure, so a
separate vector service would be one more thing to run and deploy for no real
gain at this scale.

**Hybrid search (vector + BM25).** The hypothesis was that immigration questions
hinge on exact form numbers like "I-864" or "I-129F", and that pure semantic
search would blur them together (I-864 vs I-864A vs I-864EZ), so BM25 would keep
the exact-match signal. The eval only partly bore this out: vector search already
handles form numbers fine, and BM25's measured lift was on conceptual questions
(context precision +0.08), not exact-token ones. Hybrid still wins overall, just
not for the reason expected.

**LLM reranking over a wider candidate pool.** The first-stage retriever fetches
25 candidates; an LLM reranker scores each against the question and keeps the top
5. This was added after the eval showed a chunk that answered a question was
retrievable but ranked too low to make the top 5 (the question said "waived", the
document said "exceptions"). Reranking moved it from rank 16 to rank 1.

**Chunking: SentenceSplitter, 512 / 50.** <!-- TODO: why this size, and why you
fixed it rather than tuning it. A sentence or two. -->

**Structured output with one retry.** The model returns a Pydantic
`Answer{answer, sources}` object that gets validated; a malformed response is
retried once before failing. Raw LLM JSON is not reliable enough to trust
unvalidated in a request path.

## What the eval caught

<!-- TODO: pick one question that scored badly and write 2-3 plain sentences:
what the question was, why it failed (bad chunk boundary? wrong retrieval?
missing context?), and what you'd change. This is worth more than a high
average score. -->

## Pipeline

Ingest is two stages: `extract` turns the PDFs in `data/` into cleaned `.txt`
(stripping watermarks, page headers, and tables of contents), and `ingest`
splits the `.txt` into 512-token chunks (50-token overlap), embeds them with
`text-embedding-3-small`, and stores the vectors in pgvector. The cleaned `.txt`
is committed as the source of truth; the raw PDFs are not. A query fetches a pool
of candidates (hybrid vector + BM25), an LLM reranker keeps the top 5, and
`gpt-4o-mini` generates a structured `Answer{answer, sources}` returned over
`POST /query`.

## Stack

Python, LlamaIndex, pgvector (Postgres), OpenAI `text-embedding-3-small`,
RAGAS, FastAPI, Docker, Render.

## Run locally

`docker compose` brings up the database (pgvector, extension auto-created) and
the app together. Set `OPENAI_API_KEY` first (in your shell or a `.env` file).

```bash
cp .env.example .env              # fill in OPENAI_API_KEY

docker compose up -d              # start db + app -> http://localhost:8000
docker compose run --rm app python ingest.py --extract   # one-time: PDFs -> .txt -> pgvector

curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "Can the I-864 be waived?"}'
```

To work on the code directly (no app container), run the database from compose
and the Python locally:

```bash
conda create -n immigration-rag python=3.13 && conda activate immigration-rag
pip install -r requirements.txt

docker compose up -d db                      # just the database
PYTHONPATH=src python src/ingest.py --extract  # PDFs -> .txt -> pgvector
python scripts/inspect_chunks.py             # sanity-check the chunks (counts, junk, samples)
PYTHONPATH=src python eval/run_eval.py        # baseline vs hybrid comparison
uvicorn src.api:app --reload                 # serve POST /query
```

`run_eval.py` flags: `--rerank` scores with the production reranker on,
`--serial` forces single-threaded (for timing), `--limit N` runs the first N
items.
