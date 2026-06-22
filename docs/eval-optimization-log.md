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

## Corpus hygiene (no measured metric effect)

Not every change was about scores. Ingestion was split into extract (PDF ->
cleaned .txt) and ingest (.txt -> embeddings), so the cleaned corpus is committed
and inspectable instead of being produced invisibly at ingest time. Making the
text visible immediately surfaced noise: table-of-contents leader lines
("What Is the Purpose... ____ 2") were being embedded as chunks, so a strip for
those was added to the cleaning step.

These are reproducibility and maintainability wins, not optimizations. The eval
was flat across them (faithfulness 0.94 -> 0.92, within borderline-item noise),
so the TOC removal is justified on principle - less junk in the index - but did
not measurably move the metrics, and is not claimed to. The chunk count dropped
222 -> 191 because text now chunks across page boundaries instead of per page;
the eval was unchanged by that too.

## Baseline vs hybrid (the headline comparison)

The eval set was grown 8 -> 24 items so one hard item could not swing the
average, with each new item tagged exact-token (form numbers, fees, section
citations) or conceptual, to test where hybrid actually helps.

The comparison varies only retrieval mode (vector vs vector+BM25); chunking,
model, prompt, and reranking are held constant. Reranking is turned OFF for this
table: with it on, the reranker surfaces the same best chunk regardless of which
first-stage retriever fed the pool, so it masks the retrieval-mode difference
entirely (an earlier run with rerank on showed deltas of ~0.00 across the board).
Rerank off is the apples-to-apples retrieval comparison.

Overall (24 items, rerank off):

| Metric | Baseline (vector) | Hybrid (vec+BM25) | delta |
|---|---|---|---|
| Faithfulness | 0.95 | 0.97 | +0.02 |
| Answer relevancy | 0.78 | 0.83 | +0.05 |
| Context precision | 0.92 | 0.94 | +0.02 |
| Context recall | 0.92 | 0.93 | +0.01 |

Caveat on significance: this is one run of 24 self-authored items graded by an
LLM, so the scores carry run-to-run noise. Deltas below about 0.05 are
directional, not significant — read the overall faithfulness/precision/recall
gains as "hybrid is no worse and probably a bit better," not as firm
measurements. The conceptual context-precision gain (+0.08) is the one delta
clearly outside that noise band. Use `run_eval.py --repeat N` to see the spread
across runs.

The gains are modest but consistent. The per-type breakdown is the interesting
part, and it contradicted the original hypothesis:

| Type | Metric | Baseline | Hybrid | delta |
|---|---|---|---|---|
| exact-token (7) | context precision | 0.96 | 0.94 | -0.02 |
| exact-token (7) | context recall | 0.95 | 0.95 | +0.00 |
| conceptual (9) | context precision | 0.91 | 0.99 | +0.08 |
| conceptual (9) | context recall | 0.92 | 0.96 | +0.04 |

The hypothesis was that BM25 helps on exact form numbers ("I-864" vs "I-864A").
The data says the opposite: the embedding model already handles form numbers, so
hybrid adds nothing there, while BM25's lift shows up on conceptual questions
where keyword overlap helps. Hybrid still wins overall, just not for the reason
expected.

## Parallelizing the eval

The eval was reshaped so each item runs its whole pipeline (retrieve -> generate
-> grade) as one bounded-concurrency task. retrieve/generate are sync and are
offloaded to threads (each gets its own pgvector connection, which is safe);
previously they ran in a fully serial loop while only the grading was concurrent.

Measured on a 12-item slice (24 item-runs), rerank off:

| Concurrency | Wall clock |
|---|---|
| 1 (serial) | 449.8s |
| 4 (parallel) | 285.5s |

About 1.6x, not 4x: each item has an internal retrieve -> generate -> grade
critical path, and the four grading calls per item were already concurrent in
both runs, so only the cross-item dimension was newly parallelized. Concurrency 8
was faster still but triggered intermittent OpenAI timeouts, so 4 is the stable
setting; metric calls also retry transient timeouts, and a per-item failure is
skipped rather than aborting the whole run.
