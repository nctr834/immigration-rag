# Immigration RAG

A single-pipeline RAG system over USCIS immigration documents, with a real
evaluation harness, deployed to a public URL.

**Stack:** Python · LlamaIndex · pgvector · OpenAI `text-embedding-3-small` ·
RAGAS · FastAPI · Docker · Render

## Structure

```
src/
  ingest.py      # PDF -> chunks -> pgvector
  retrieve.py    # query -> retrieved chunks
  generate.py    # chunks + question -> answer
  api.py         # FastAPI app
eval/
  eval_set.json  # 50 questions
  run_eval.py    # RAGAS harness
data/            # raw PDFs (gitignored)
```
