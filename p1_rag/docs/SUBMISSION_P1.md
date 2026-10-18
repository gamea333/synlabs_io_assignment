# Submission draft — Problem 1 (copy into the template section by section)

> Every number below comes from a file in `p1_rag/results/` and is reproducible with the commands in `p1_rag/README.md`.
> Sections marked **✍️ WRITE YOURSELF** must be written by you without AI assistance (template rule).

## Candidate details

| Field | Your entry |
|---|---|
| Full name | Aditya Singh Mehra |
| Email | mehraaditya777@gmail.com |
| Date submitted | 18 October 2026 |
| Repository link | _(add after pushing to GitHub)_ |
| Total time spent | ~7–8 h across 2 sessions (17–18 Oct 2026) |
| Problem 1 — vector store chosen | LanceDB (embedded, file-based; FAISS benchmarked as a second store) |
| Problem 2 — judge model / generator model | judge: gpt-oss-120b (OpenAI family) · generator: qwen3.8-27b (Alibaba family) |

---

## 1.1 Architecture Diagram / Flowchart

Paste the image `p1_rag/docs/screenshots/architecture.png`. Its source is `p1_rag/docs/architecture.html`, also included as Mermaid in the README.
The red boxes are the two "no relevant context" branches.

---

## 1.2 Setup & Run Instructions

**Prerequisites:** Python 3.11+ (tested on 3.13.15). Windows, macOS or Linux. About 1 GB of disk. Internet access for the first model and corpus download. No GPU, Docker or database server is needed. You need a free Groq API key, or any OpenAI-compatible endpoint such as a local Ollama.

**Environment variables** (names only; see `.env.example`): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `JUDGE_MODEL`, `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SCORE`, `LANCEDB_URI`, `LANCEDB_TABLE`, `LOG_PATH`

**Install:**
```bash
git clone <repo-url> && cd <repo>/p1_rag
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # set LLM_API_KEY
python scripts/fetch_corpus.py                        # 5 arXiv PDFs (HTML/MD snapshots are in the repo)
```

**Ingest a corpus:**
```bash
python -m app.cli ingest data/corpus      # default chunk size 256 tokens, overlap 32 (cl100k tokens)
python -m app.cli stats                   # vector count + duplicate check (re-running ingest is a no-op)
```

**Run a query (HTTP endpoint):**
```bash
uvicorn app.api:app --port 8000
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How many layers does the Transformer encoder have?", "k": 5, "doc_type": "pdf"}'
# -> {"answer": "The Transformer encoder stack contains six identical layers. [1]",
#     "citations": [{"chunk_id": "...", "source": "pdf/attention_is_all_you_need.pdf", "page": 2, "score": 0.75}], ...}
```

**Vector store chosen + one-line why:** LanceDB. It's an embedded, columnar, on-disk store with no server to keep running, it can be backed by S3, and it supports SQL metadata pre-filters and an IVF_PQ index for scale. At 10M vectors it costs $84/mo, against $1,053 for Pinecone p1 pods.

---

## 1.3 Evaluation Results

Fixed set: `eval/gold.json`, **27 questions**. 23 are answerable: 22 single-fact and 1 needing two documents. 4 are unanswerable: 1 off-topic and 3 on-topic but absent from the corpus.
Corpus: 15 documents (5 PDF, 5 HTML, 5 MD), giving **463 chunks**.

### 1.3.1 Retrieval metrics — k used: **k = 5**

| Metric | Value | How computed / notes |
|---|---|---|
| Recall@k / Hit Rate | **Hit@5 = 0.870** (20/23); Recall@5 = 0.665 | Hit: ≥1 relevant chunk in the top 5. Recall: relevant chunks in the top 5 ÷ all relevant chunks in the index. Overlap puts some facts in 2–3 chunks, which is why recall is lower than hit rate. |
| MRR | **0.750** | Mean over questions of 1 / rank of the first relevant chunk, or 0 if none is retrieved |
| nDCG@k | **0.649** | Binary gains with a log2(rank+1) discount, normalized by the ideal DCG at min(#relevant, 5) |
| Context precision | **0.733** (rank-weighted); P@5 = 0.252 | RAGAS definition: the mean of precision@i over ranks i that hold a relevant chunk. **A chunk counts as relevant** if it comes from the fact's source document and contains a verbatim answer-bearing quote (after NFKC, whitespace and case normalization). Quotes came from searching the whole source document, so the labels don't depend on chunk size. |

### 1.3.2 Answer quality

| Metric | Score | Method | Notes |
|---|---|---|---|
| Faithfulness / groundedness | **1.000** | LLM judge (qwen3.8-27b, a different family from the gpt-oss generator) | The answer is split into atomic claims and each is checked against the exact context the generator saw. **Validated with a corruption probe:** swapped numbers were caught 14/14 and fabricated claims 22/22, with 0/22 false alarms (`results/judge_probe.json`). |
| Answer relevance | **0.957** | LLM judge | A 1–5 scale mapped to 0–1, over all 23 answerable questions. A refusal scores 0; the only refusal was q13. |
| EM | 0.000 | gold | SQuAD normalization plus number words. The system answers in full sentences, so EM against short gold answers is structurally about 0. Judge correctness vs gold = **0.870**, and contains-gold = 0.478. |
| F1 | 0.217 | gold | SQuAD token F1, which penalizes full-sentence answers for their extra words. |
| *Refusals* | **4/4** correct; 1/23 false | — | All 4 unanswerable questions were refused (gate 2). One answerable question was wrongly refused (q13). |

### 1.3.3 Cost comparison across scale

| Vectors | Your store ($/mo) | Managed DB ($/mo) | Savings / notes |
|---|---|---|---|
| 100K | **$26.14** (t4g.medium + 20 GB gp3 + S3 backup) | $81.03 Pinecone p1 pod · $50.00 Pinecone serverless (plan minimum) | 68% vs pods · 48% vs serverless |
| 1M | **$26.21** (same box; IVF_PQ index) | $162.06 (2 p1 pods) · $50.00 serverless (minimum) | 84% vs pods · 48% vs serverless |
| 10M | **$83.52** (r7g.large + 56 GB gp3) | $1,053.39 (13 p1 pods) · $243.09 (3 s1 pods) · $101.83 serverless | 92% vs p1 pods · 18% vs serverless. **Serverless is cheaper below about 69K queries/month** at this size. |

**Pricing assumptions:** All prices are list prices checked 2026-10-17, in USD, AWS us-east-1, on-demand, 730 h/month.
- **Record size:** **3,730 B, measured** from our table. That's a 384-d float32 vector (1,536 B) plus chunk text (about 1,983 B) plus metadata (about 211 B). The same record is assumed in Pinecone.
- **Pinecone serverless (Standard plan):** $50/mo minimum, $0.33/GB-mo storage, $16 per million read units (RU), $4 per million write units (WU).
  - A query costs **1 RU per GB of namespace** (minimum 0.25).
  - An upsert costs 1 WU per KB, and an overwrite also pays for the old record's KB.
  - 10% of vectors are assumed to be re-upserted each month.
  - Query volume is **100K/month**, with a sensitivity table from 10K to 10M in the README.
- **Pinecone pods (legacy):** $0.111/h for p1.x1 or s1.x1. Pod capacity is converted to bytes from Pinecone's sizing of 1M (p1) or 5M (s1) 768-d vectors with no metadata. 1 replica, i.e. no HA.
- **Our side:**
  - The **whole VM** is charged to the vector store. This overstates our cost, since a Pinecone setup also needs an app server.
  - t4g.medium at $0.0336/h; r7g.large at $0.1071/h.
  - gp3 at $0.08/GB-mo, provisioned at 1.5× the data size for index files and versions.
  - S3 backup at $0.023/GB-mo.
  - An IVF_PQ index above about 100K vectors, which our benchmark showed is required.
- **Not modelled:** egress, operations time, HA replicas on either side, and reserved-instance discounts.

### 1.3.4 Latency

| Metric | Value (ms) | Notes / conditions |
|---|---|---|
| Retrieval p50 | **16.0** search only (30.3 with query embedding) | k = 5, 463 vectors, LanceDB exact cosine search; 540 warm queries (27 × 20); AMD Ryzen 7 5800H, 15.4 GB RAM, NVMe, Windows 11 |
| Retrieval p95 | **21.7** search only (41.4 with embedding) | Same conditions. **At scale (bonus benchmark):** LanceDB IVF_PQ at 1M vectors gives 19.1 / 60.5 ms p50/p95 with recall@10 = 0.915 |
| End-to-end p95 (optional) | **687** (p50 529) | Includes LLM generation (gpt-oss-20b via Groq, about 1.6K prompt tokens). Provider rate-limit waits are excluded; wall-clock p95 including them is 1,755 ms. |

---

## 1.4 Design Decisions & Trade-offs

**Why this vector store over the others on the list?**
I wanted the cheapest thing that still behaves like a database. FAISS is the fastest in RAM (12 ms exact at 100K in our benchmark), but it's a library. It has no metadata filtering, persistence or upserts, so I'd have had to build all three, and its RAM footprint scales linearly with vectors. Chroma and Qdrant are good but are normally run as servers, which brings back the always-on cost the brief is about. pgvector is attractive if Postgres is already there, but it means running a database server. sqlite-vec is pre-v1 and brute-force only. LanceDB is embedded: it runs in-process with no server, stores data as columnar files that can live on S3, does SQL pre-filtering on metadata, upserts by key with `merge_insert`, and has an IVF_PQ index. In our benchmark that index gave 19 ms p50 at 1M vectors with 0.915 recall. FAISS was benchmarked as the second store.

**Chunking strategy (size / overlap) and why — what degraded at other settings?**
256-token windows with 32-token overlap (12.5%), measured in cl100k tokens so sizes match the LLM's token budget. Picked from a 6-configuration sweep on the gold set. 256/32 had the best Hit@5 (0.870), MRR (0.750), nDCG (0.649) and context precision (0.733), while sending half the context tokens of 512. **128/16** split facts away from their context, so MRR fell to 0.601 and four questions were missed. **512 and 1024** diluted each chunk with surrounding text, so the specific chunk ranked lower (MRR 0.63–0.68 at 512, 0.565 at 1024), and they cost 2–4× more prompt tokens per query. The three 512 variants differed by only 1–2 questions, which is within noise for a 23-question set, so I didn't read anything into overlap at that size. **k = 5:** going from 5 to 10 adds one hit (0.870 → 0.913) for twice the prompt tokens.

**Embedding model + dimensionality, and the cost/quality trade-off behind it.**
`BAAI/bge-small-en-v1.5`, **384 dimensions**, run locally with fastembed (ONNX on CPU). Embedding costs nothing per vector, has no API dependency, and takes about 7 ms per query. Vector storage is 1,536 B per record, half that of a 768-d model and a quarter of 1536-d OpenAI embeddings. That directly halves or quarters the storage and pod lines of the cost model, and makes IVF_PQ codes smaller. The cost is quality. Three of 23 facts were never retrieved, and the similarity scores are compressed (in-scope and out-of-scope top-1 scores overlap at 0.52–0.56), which weakens score thresholding. The next upgrade I'd try is bge-base (768-d), together with hybrid BM25 search, which targets the exact-string misses (q05 "RAG-Token", q07 "9%-19%").

**How you handle "no relevant context" without hallucinating.**
There are two gates. **Gate 1, retrieval:** if no retrieved chunk has cosine ≥ `MIN_SCORE`, the service returns a fixed refusal and **never calls the LLM** (0 tokens). **Gate 2, generation:** the prompt numbers each passage, requires a `[n]` citation on every sentence, and requires the exact string "I don't know based on the provided documents." when the passages don't contain the answer. The code detects that string and returns `answerable: false`. Answers without any valid citation are flagged `answer_without_citations`. I first calibrated `MIN_SCORE = 0.58` on 6 hand-picked questions. **The gold set showed that was wrong:** an answerable question ("authors of the HNSW paper", 0.529) scored inside the off-topic range (0.518–0.563), and on-topic-but-absent questions scored 0.61–0.66, so no threshold could separate them. I lowered gate 1 to 0.50 so it can't cause false refusals, and gate 2 does the real work. Result: **4/4** unanswerable questions refused, **0** hallucinated answers (faithfulness 1.0, validated by the corruption probe), and 1/23 over-refusal (q13).

**Idempotent re-ingest: how do you guarantee no duplicate vectors?**
Three layers:
1. **Document-hash skip.** The SHA-256 of the file bytes, the chunk size/overlap and the embedding model are stored on every row. If all of them match, the file is skipped and not re-embedded. A plain re-run took 0.11 s and wrote 0 chunks.
2. **Replace on change.** If a file or the chunk config changes, all of that source's rows are deleted before the new chunks are inserted. Switching from 512 to 256 tokens took the table from 233 to **463, not 696**.
3. **Deterministic `chunk_id`** = sha256(source, index, text), written with LanceDB `merge_insert` (upsert on key). Even `--force`, which re-embeds everything, leaves 233 → 233 vectors.

`stats` compares the row count with the distinct `chunk_id` count and reported `duplicates: 0` after every run (screenshots attached). `--prune` removes vectors for deleted files.

**Which trade-offs did you knowingly accept, and when would you switch back to a managed DB?**
**Accepted:**
- A single node with no HA. Recovery is from an S3 backup, so minutes rather than seconds.
- One writer process at a time.
- Above about 100K vectors, building and periodically rebuilding an IVF_PQ index, and accepting about 0.92 ANN recall instead of exact search.
- Owning compaction, upgrades, monitoring and capacity planning ourselves.
- A small embedding model that costs some retrieval recall.

**I'd switch back to managed when:**
1. The index is **huge and almost never queried**. At 10M vectors and fewer than about 69K queries/month, Pinecone serverless ($50–102) is cheaper than our always-on box ($84). LanceDB-on-S3 behind a scale-to-zero function could compete again, but with cold-start latency.
2. We need an **SLA, HA, multi-region, or many concurrent writers**.
3. Nobody wants to own the operational work. One engineer-day a month costs more than the whole 10M-vector bill.

For the brief's case (large index, lightly but steadily queried), self-hosted LanceDB is 84–92% cheaper than pods.

---

## 1.5 Subtask Evidence & Reflection

**Screenshot — end-to-end query:** paste `docs/screenshots/query_1_log_latency_chunks_tokens.png`. It's the per-query log record from `logs/queries.jsonl` for the 13:42:51 query: latency embed 8.1 / search 10.0 / llm 634.8 / total 653.0 ms, `chunks_retrieved: 5`, `chunks_used: 1`, tokens 1,656 / 61. Next to it, paste `query_2_cited_answer.png`, the HTTP response for the same query, showing the cited answer with identical latency and token numbers.

**Screenshot — idempotent re-ingest:** paste `docs/screenshots/reingest_1_stats_before.png`, `reingest_2_ingest_skipped.png` and `reingest_3_stats_after.png`. They show 463 vectors, then a re-ingest that skips all 15 files (`chunks_written: 0`, 463 → 463), then 463 vectors and `duplicates: 0`.

**Walk through one query (your own words):** ✍️ WRITE YOURSELF.
*Raw facts you can use (not prose):* FastAPI validates `QueryIn` → `rag.answer()` → bge embeds the question with the query prefix (about 7–15 ms) → LanceDB cosine top-k with a `doc_type` / `source` pre-filter (about 10–16 ms) → gate 1 `MIN_SCORE` → numbered-passage prompt → Groq gpt-oss-20b (about 500–900 ms) → gate 2 refusal-string check → regex `[n]` → chunk_id, source, page, score → JSONL log line → JSON response.

**Something that broke first time:** ✍️ WRITE YOURSELF. Pick one:
- **(a)** The first answer was correct but had no `[n]` citations, so `citations: []`. Fixed with a few-shot example plus repeating the rule at the end of the user turn.
- **(b)** LLM latency jumped from 0.7 s to 21 s. The Groq headers showed an 8K tokens/minute limit, and the OpenAI client was silently retrying 429s. I moved to manual retries and logged `throttle_ms` separately.
- **(c)** The 0.58 threshold looked clean on 6 questions but would have refused a real question (0.529) on the gold set.
- **(d)** First retrieval score was Hit@5 0.61, but half of the "misses" were labels that were too narrow. I fixed them by searching the whole document for every passage stating each fact.
- **(e)** Wikipedia returned 403, then 429. I switched to the REST API and added backoff that honours `Retry-After`.

**AI usage disclosure:** ✍️ WRITE YOURSELF. Be accurate: Claude Code (Claude Opus) wrote most of the code, the eval harness, the cost model and these drafts under your direction. State what you decided, reviewed, ran and verified yourself.
