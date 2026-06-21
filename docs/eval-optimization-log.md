# Eval Optimization Log

What was tried to improve the RAG eval scores and whether it worked. Metrics are
RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) from
eval/run_eval.py against eval/eval_set.json.

## Summary

| Change | Goal | Result |
|---|---|---|
| Pin generation + judge LLMs to temperature 0 | Reproducible scores | Worked |
| Cache get_retriever (lru_cache) | Stop rebuilding BM25 per query | Worked |
| use_async=False on fusion retriever | Avoid nested-loop crash in eval | Worked, no behavior change |
| Expand eval set 3 -> 8 items | Stop one item swinging the average | Worked |
| Exp 1: raise top_k | Surface the missing chunk | No (needs top_k>=16, floods context) |
| Exp 2: query expansion (num_queries=4) | Bridge waived vs exceptions vocab gap | Partial (rank 16 -> 10, still not top-5) |
| Exp 3: LLM rerank (pool 25 -> top 5) | Fix ranking quality directly | Worked (rank 18 -> 1, recall 0.20 -> 1.00) |

## Reproducibility

Re-running the eval gave different scores each time. Two LLMs are involved: the
one that generates answers and the RAGAS judge that grades them. Pinning only the
judge did not help; the generation LLM ran at default temperature, so the answer
being graded changed every run. Setting temperature 0 on both made consecutive
runs identical on well-grounded items. Residual wobble remained only on items
where retrieval is borderline (q001/q002), which is itself a signal that those
items have a retrieval problem, not an eval problem.

Temperature 0 also makes product answers deterministic, which is a product
decision, not just an eval knob.

## Harness

- Retriever caching: get_retriever reads all chunks and builds an in-memory BM25
  index once per process via lru_cache, not per query.
- Concurrency: retrieve/generate run serially (they share one pgvector async
  connection that cannot run concurrent operations); grading runs concurrently,
  since grading is the slow part.
- use_async=False: the fusion async path tried to start an event loop inside the
  eval's running loop. With num_queries=1 there is nothing to fan out, so
  disabling async removes the crash at no cost.

## Eval set expansion

With 3 items, one hard item moved the average by ~0.33. Added q004-q008 grounded
in i-129finstr, i-864ainstr, the I-485 245(i) supplement, and the public charge
rule, each written from actual chunk text. The new items score near 1.00, which
validates them and stops the two hard items from dominating the average.

## The q002 retrieval gap

q002 ("Can the Form I-864 requirement be waived?") scored worst: faithfulness
~0.50-0.75, recall 0.20, with a vague hedged answer. The chunk that answers it
("Are There Exceptions to Who Needs to Submit Form I-864?", listing all four
exception categories) exists in the corpus verbatim but never reached the top-5.

Root cause is a vocabulary gap: the question says "waived", the document says
"exceptions" / "do not need to file". BM25 gets no lexical help, and "waived"
embeds closer to the generic purpose/obligation chunks than to the exceptions
list, so the right chunk sits at rank 16 of 20, below a score cliff. This is a
ranking problem, not a recall-depth or chunking problem. (Fused scores are tiny,
~0.03, because RRF produces values around 1/(k+rank); magnitude is not relevance,
only rank matters.)

Exp 1, raise top_k: capturing rank 16 needs top_k>=16, which injects 15
irrelevant chunks first and tanks precision. Used only as a probe to confirm the
chunk is findable but badly ranked.

Exp 2, query expansion (num_queries=4): the retriever rewrites the query into
variants and fuses results. Moved the target chunk from rank 16 to 10, but still
not into the top-5, so not sufficient alone.

Exp 3, LLM rerank: retrieve a wide pool (25), then LLMRerank (gpt-4o-mini, temp
0) scores each candidate against the question and keeps the top 5. The target
chunk went from rank 18 in the pool to rank 1.

End-to-end with rerank patched in:

| q002 metric | baseline | with rerank |
|---|---|---|
| faithfulness | 0.50-0.75 | 0.89 |
| context_precision | 0.45-0.59 | 0.92 |
| context_recall | 0.20 | 1.00 |

Aggregate recall went 0.78 -> 1.00, precision 0.83 -> 0.94, nothing regressed.
Reranking is the fix because the failure was always ranking quality: the chunk
was retrievable the whole time, just buried.

Reranking is now wired into retrieve.py: get_retriever fetches RERANK_POOL=25
candidates, get_reranker (LLMRerank, gpt-4o-mini, temp 0) trims to TOP_K. Both
are lru_cached.

## The q001 corpus gap

After reranking went live, q001 ("Does the medical exam from the K-1 process
carry over to the I-485?") dropped to faithfulness 0.00 / relevancy 0.00. This
was not a regression: the answer was never in the corpus (its source was the
USCIS Policy Manual, which had not been ingested). Before reranking the system
guessed from tangential i-485 chunks; after reranking it correctly refused.

The original ground_truth was also factually wrong ("valid within one year").
Fixed by ingesting USCIS Policy Manual Vol 8 Part B Chapter 4 (medical exam
documentation) into data/ and rewriting q001's ground_truth from the actual text
(K nonimmigrants generally do not repeat the overseas exam, but may still need to
show vaccination compliance). q001 recovered to faithfulness 0.67 / recall 0.75 -
a genuine score, no longer a refusal.

Final aggregates (8 items, reranking live): faithfulness 0.90, answer_relevancy
0.86, context_precision 0.98, context_recall 0.97.
