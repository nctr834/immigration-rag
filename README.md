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

The eval set is 50 questions scored with [RAGAS](https://docs.ragas.io/) on four
metrics: faithfulness, answer relevancy, context precision, context recall.

The two columns differ only in retrieval: the baseline is vector search alone,
the hybrid column adds BM25 keyword fusion. Chunking, the model, and the prompt
are identical across both, so the difference is attributable to retrieval.

<!-- TODO: fill in the real numbers once you've run both configs. -->

| Metric            | Baseline (vector) | Hybrid (vector + BM25) | delta |
| ----------------- | ----------------- | ---------------------- | ----- |
| Faithfulness      |                   |                        |       |
| Answer relevancy  |                   |                        |       |
| Context precision |                   |                        |       |
| Context recall    |                   |                        |       |

## Architecture decisions

**pgvector instead of a dedicated vector DB.** The corpus is small (a few
hundred chunks) and already lives in Postgres-shaped infrastructure, so a
separate vector service would be one more thing to run and deploy for no real
gain at this scale.

**Hybrid search (vector + BM25).** Immigration questions hinge on exact form
numbers like "I-864" or "I-129F". Those are precise tokens, and pure semantic
search tends to blur them together (I-864 vs I-864A vs I-864EZ). BM25 keeps the
exact-match signal that vector search loses.

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

Ingest reads the PDFs in `data/`, splits them into 512-token chunks (50-token
overlap), embeds them with `text-embedding-3-small`, and stores the vectors in
pgvector. A query retrieves the top 5 chunks (hybrid vector + BM25), and
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
docker compose run --rm app python ingest.py   # one-time: PDFs -> pgvector

curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "Can the I-864 be waived?"}'
```

To work on the code directly (no app container), run the database from compose
and the Python locally:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

docker compose up -d db           # just the database
python src/ingest.py              # PDFs in data/ -> pgvector
python scripts/inspect_chunks.py  # sanity-check the chunks (counts, junk, samples)
python eval/run_eval.py           # baseline scores
uvicorn src.api:app --reload      # serve POST /query
```
