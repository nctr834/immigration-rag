# Immigration RAG

[![tests](https://github.com/nctr834/immigration-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/nctr834/immigration-rag/actions/workflows/tests.yml)

Ask questions about USCIS immigration forms and get answers grounded in the
actual instruction documents, with citations.

I built this while going through the K-1 fiancé visa process. The instructions
are long, cross-referenced PDFs (the I-485 instructions alone run 40+ pages), and
answering one question could mean skimming through all of them to find the
paragraph that applies. This indexes the forms and answers directly, pointing
back to the source so the answer is checkable.

**Live demo:** https://immigration-rag.onrender.com/ui
(free tier; first request after idle takes ~30s to wake)

```bash
curl -X POST https://immigration-rag.onrender.com/query \
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
| Faithfulness      | 0.95              | 0.97                   | +0.02 |
| Answer relevancy  | 0.78              | 0.83                   | +0.05 |
| Context precision | 0.92              | 0.94                   | +0.02 |
| Context recall    | 0.92              | 0.93                   | +0.01 |

The gains are modest but consistent. Breaking the set down by question type shows
where hybrid actually helps: on conceptual questions context precision rose +0.08
(0.91 -> 0.99) and recall +0.04 (0.92 -> 0.96), while on exact-token questions (form numbers
like "I-864") the two were within noise. The embedding model already handles form
numbers well; BM25's lift shows up on questions where keyword overlap helps. See
[docs/eval-optimization-log.md](docs/eval-optimization-log.md) for the full set
of experiments.

The deployed system adds an LLM reranker on top of hybrid retrieval. Against the
vector baseline, the full system (hybrid + reranking together) scores answer
relevancy 0.83 (from 0.78), context precision 0.97 (from 0.92), context recall
0.96 (from 0.92), and faithfulness 0.93 (from 0.95). The faithfulness move is
within run-to-run noise on 24 items, so I don't read it as a real trade. The
reranker's value shows in individual rankings rather than the averages: on one
question it moved the answering chunk from rank 16 to rank 1 (see the
optimization log).

Per-query latency (warm caches): about 6s end to end, of which the retrieval
stage (embed + BM25 + a 25-candidate LLM rerank) is ~4s and generation ~2s. The
reranker is the dominant cost; on gpt-4o-mini a query is a fraction of a cent.

Two things keep the eval honest. The RAGAS judge is Claude (Anthropic), a
different model family from the gpt-4o-mini generator, so the scores aren't a
model grading its own output (`eval/run_eval.py`, needs `ANTHROPIC_API_KEY`). And
a separate out-of-scope set (`eval/eval_set_oos.json`, run with `--oos`) measures
what matters most for a grounded assistant: whether it refuses questions the
documents don't cover. It refuses all 8 without fabricating — including a fee
question, where it points to the USCIS fee schedule rather than inventing a
number (redirecting to the right source counts as a refusal, since it gives no
unsupported answer).

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
25 candidates; an LLM reranker scores each against the question and keeps the top 5. This was added after the eval showed a chunk that answered a question was
retrievable but ranked too low to make the top 5 (the question said "waived", the
document said "exceptions"). Reranking moved it from rank 16 to rank 1.

**Chunking: SentenceSplitter, 512 / 50.** 512 tokens is large enough to keep a
form's instruction (a requirement plus its conditions) in one chunk, and small
enough that retrieval stays specific; the 50-token overlap keeps a sentence
that straddles a boundary from being cut in half. The size was fixed rather than
tuned: with a corpus this small, chunk-size sweeps overfit the eval set, and the
retrieval gains came from hybrid search and reranking, not chunk size.

**Structured output with one retry.** The model returns a validated Pydantic
`Answer` (the prose plus a disclaimer); a malformed response is retried once
before failing. Raw LLM JSON is not reliable enough to trust unvalidated in a
request path.

**Citations are built, not generated.** The model produces only the prose. Each
source citation (file + a verbatim quoted passage) is attached from the chunks
that were actually retrieved, so a citation can't be hallucinated and the quote
lets a reader check the answer against the source text.

## What the eval caught

The question "Can the Form I-864 be waived?" kept scoring badly: context recall
0.20, and the answer hedged. The chunk that answers it does exist in the corpus
(the "Are There Exceptions to Who Needs to Submit Form I-864?" section), but it
was ranked 16th, below the top 5 that reach the model. The cause was a vocabulary
mismatch: the question says "waived", the document says "exceptions" / "do not
need to file", so neither keyword nor embedding search ranked it highly. Adding an
LLM reranker over a wider candidate pool moved that chunk from rank 16 to rank 1
and recall to 1.00. The fix came from reading one bad score, not the average; the
full set of experiments is in
[docs/eval-optimization-log.md](docs/eval-optimization-log.md).

## Pipeline

Ingest is two stages: `extract` turns the PDFs in `data/` into cleaned `.txt`
(stripping watermarks, page headers, and tables of contents), and `ingest`
splits the `.txt` into 512-token chunks (50-token overlap), embeds them with
`text-embedding-3-small`, and stores the vectors in pgvector. The cleaned `.txt`
is committed as the source of truth; the raw PDFs are not. A query fetches a pool
of candidates (hybrid vector + BM25), an LLM reranker keeps the top 5, and
`gpt-4o-mini` generates a structured `Answer` (prose + citations with quoted
passages + a disclaimer) returned over `POST /query`.

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

## The Gradio UI

A one-page UI for asking questions without curl. It is mounted onto the same
FastAPI app at `/ui` and calls `generate()` in-process, so it ships with the API
as one service. With the app running, open `http://localhost:8000/ui`.

## Deploy (Render)

`render.yaml` provisions one Dockerized web service (REST API at `/query`, UI at
`/ui`) and a managed Postgres with pgvector.

1. On Render: New -> Blueprint, point it at this repo. It reads `render.yaml`.
2. Set `OPENAI_API_KEY` when prompted (it is not committed). `DATABASE_URL` is
   wired to the database automatically.
3. After the first deploy, open the web service's Shell and run the one-time
   ingest against the live DB: `PYTHONPATH=src python ingest.py --extract`.
   This creates the `vector` extension, embeds the committed `data/*.txt`, and
   fills pgvector.
4. Visit `https://<your-app>.onrender.com/ui` for the UI, or POST to `/query`.
   Paste the URL into the live-demo line at the top of this README.

Cold start: the Render free tier sleeps after inactivity, so the first request
after idle takes ~30s to wake the service. A periodic ping to `/health` keeps it
warm.
