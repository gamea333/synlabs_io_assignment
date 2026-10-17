# Applied AI / ML Engineering — Take-Home

| Problem | Folder | Start here |
|---|---|---|
| **P1 — Cost-efficient RAG** (LanceDB, FAISS benchmarked as a second store) | [`p1_rag/`](p1_rag/) | [`p1_rag/README.md`](p1_rag/README.md) |
| **P2 — LLM-as-Judge pipeline** (judge gpt-oss-120b ≠ generator Qwen 3.8) | [`p2_judge/`](p2_judge/) | [`p2_judge/README.md`](p2_judge/README.md) |

Each folder is self-contained: its own `requirements.txt`, `.env.example` (no secrets committed), `results/` with every
number quoted in its README, and `docs/` with the architecture diagram, evidence screenshots and a submission-template draft.

Quick start (either problem):

```bash
cd p1_rag   # or p2_judge
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # add your API key(s)
```
