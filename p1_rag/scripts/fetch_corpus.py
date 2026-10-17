"""Download the evaluation corpus (PDF + HTML + MD) into data/corpus/.

PDFs use versioned arXiv URLs so content is immutable. HTML/MD snapshots are
committed to the repo so gold evidence spans stay valid; re-running this
script only fills in files that are missing.
"""
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1] / "data" / "corpus"

CORPUS = {
    "pdf": {
        "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762v7",
        "rag_lewis_2020.pdf": "https://arxiv.org/pdf/2005.11401v4",
        "dense_passage_retrieval.pdf": "https://arxiv.org/pdf/2004.04906v3",
        "lost_in_the_middle.pdf": "https://arxiv.org/pdf/2307.03172v3",
        "hnsw_malkov.pdf": "https://arxiv.org/pdf/1603.09320v4",
    },
    "html": {
        "wiki_vector_database.html": "https://en.wikipedia.org/api/rest_v1/page/html/Vector_database",
        "wiki_nearest_neighbor_search.html": "https://en.wikipedia.org/api/rest_v1/page/html/Nearest_neighbor_search",
        "wiki_retrieval_augmented_generation.html": "https://en.wikipedia.org/api/rest_v1/page/html/Retrieval-augmented_generation",
        "wiki_product_quantization.html": "https://en.wikipedia.org/api/rest_v1/page/html/Vector_quantization",
        "wiki_tf_idf.html": "https://en.wikipedia.org/api/rest_v1/page/html/Tf%E2%80%93idf",
    },
    "md": {
        "pgvector_readme.md": "https://raw.githubusercontent.com/pgvector/pgvector/master/README.md",
        "faiss_readme.md": "https://raw.githubusercontent.com/facebookresearch/faiss/main/README.md",
        "sqlite_vec_readme.md": "https://raw.githubusercontent.com/asg017/sqlite-vec/main/README.md",
        "qdrant_readme.md": "https://raw.githubusercontent.com/qdrant/qdrant/master/README.md",
        "chroma_readme.md": "https://raw.githubusercontent.com/chroma-core/chroma/main/README.md",
    },
}


def get_with_backoff(client: httpx.Client, url: str, tries: int = 5) -> httpx.Response:
    for attempt in range(tries):
        r = client.get(url)
        if r.status_code != 429 or attempt == tries - 1:
            r.raise_for_status()
            return r
        wait = int(r.headers.get("Retry-After", 2 ** (attempt + 2)))
        print(f"      429 on {url}, retrying in {wait}s")
        time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> int:
    headers = {"User-Agent": "p1-rag-eval/1.0 (https://github.com/; educational RAG evaluation corpus) httpx"}
    failed = 0
    with httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client:
        for sub, files in CORPUS.items():
            (ROOT / sub).mkdir(parents=True, exist_ok=True)
            for name, url in files.items():
                dest = ROOT / sub / name
                if dest.exists():
                    print(f"skip  {sub}/{name} (exists)")
                    continue
                try:
                    r = get_with_backoff(client, url)
                    dest.write_bytes(r.content)
                    print(f"ok    {sub}/{name}  {len(r.content) // 1024} KB")
                    time.sleep(1)  # be polite to Wikipedia / GitHub
                except httpx.HTTPError as e:
                    failed += 1
                    print(f"FAIL  {sub}/{name}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
