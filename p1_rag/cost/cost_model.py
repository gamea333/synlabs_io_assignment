"""Monthly cost: self-hosted LanceDB vs Pinecone (pods and serverless) at 100K / 1M / 10M vectors.

  python -m cost.cost_model   -> results/cost_comparison.json + results/cost_comparison.md

Every number below is an explicit, named assumption. Prices are list prices (USD) checked 2026-10-17:
  - Pinecone serverless Standard: https://www.pinecone.io/pricing/   (storage, RU, WU, $50 minimum)
  - Pinecone RU/WU rules:         https://docs.pinecone.io/guides/manage-cost/understanding-cost
  - Pinecone pods (legacy):       https://www.pinecone.io/pricing/pods/
  - AWS EC2 on-demand us-east-1 (third-party aggregators holori.com / doit.com), EBS gp3, S3 Standard list prices
"""
import json
import math
from pathlib import Path

HOURS_PER_MONTH = 730
GB = 1e9

# ---- workload ---------------------------------------------------------------------------------
DIM = 384                     # bge-small-en-v1.5
BYTES_VECTOR = DIM * 4        # float32
BYTES_RECORD = 3730           # MEASURED from our LanceDB table: vector 1536 + chunk text ~1983 + metadata ~211
QUERIES_PER_MONTH = 100_000   # "large but lightly-queried": ~2.3 queries/minute on average
MONTHLY_CHURN = 0.10          # fraction of vectors re-upserted per month (document updates)
SCALES = [100_000, 1_000_000, 10_000_000]

# ---- Pinecone serverless (Standard plan, low end of the published ranges) ---------------------
PC_MIN_MONTHLY = 50.0
PC_STORAGE_PER_GB = 0.33
PC_PER_M_RU = 16.0            # $/million read units ($16-18 depending on region)
PC_PER_M_WU = 4.0             # $/million write units ($4-4.50)
PC_RU_PER_GB_PER_QUERY = 1.0  # "1 RU for every 1 GB of namespace size"
PC_RU_MIN_PER_QUERY = 0.25
PC_WU_PER_KB = 1.0            # 1 WU per KB of request; overwrite also pays for the existing record's KB

# ---- Pinecone pods (legacy, the "always-on pod" model the brief describes) ---------------------
POD_PRICE_HR = {"s1.x1": 0.1110, "p1.x1": 0.1110}
# Pinecone sizes pods as N 768-d vectors WITHOUT metadata; convert to a byte budget per pod.
POD_CAPACITY_BYTES = {"s1.x1": 5_000_000 * 768 * 4, "p1.x1": 1_000_000 * 768 * 4}
POD_REPLICAS = 1              # no HA, to match our single-node setup

# ---- Self-hosted LanceDB (embedded in the API process, data on EBS gp3) ------------------------
# The VM runs the FastAPI app + ONNX embedder + LanceDB. We charge the WHOLE VM to the vector store,
# which overstates our cost (a Pinecone deployment still needs an app server).
EC2_PRICE_HR = {"t4g.medium": 0.0336, "t4g.large": 0.0672, "r7g.large": 0.1071}
EBS_GP3_PER_GB = 0.08
EBS_MIN_GB = 20
STORAGE_OVERHEAD = 1.5        # ANN index files + LanceDB version history (compacted periodically)
S3_PER_GB = 0.023


def lancedb_instance(n: int) -> str:
    # Measured (results/store_benchmark.json): LanceDB exact search is 133 ms p50 at 100K and
    # 1.2 s at 1M, so above ~100K we build an IVF_PQ index (1M: 19 ms p50, recall@10 0.915).
    # PQ codes are 48 B/vec (~48 MB at 1M), so a 4 GB box is enough up to 1M.
    # At 10M: ~0.5 GB PQ codes + page cache for re-ranking reads -> 16 GB memory-optimised box.
    if n <= 1_000_000:
        return "t4g.medium"
    return "r7g.large"


def lancedb_cost(n: int) -> dict:
    inst = lancedb_instance(n)
    data_gb = n * BYTES_RECORD / GB
    disk_gb = max(EBS_MIN_GB, math.ceil(data_gb * STORAGE_OVERHEAD))
    compute = EC2_PRICE_HR[inst] * HOURS_PER_MONTH
    storage = disk_gb * EBS_GP3_PER_GB
    backup = data_gb * S3_PER_GB                     # nightly copy of the table to S3
    return {"instance": inst, "data_gb": round(data_gb, 2), "disk_gb": disk_gb,
            "compute": round(compute, 2), "storage": round(storage, 2), "s3_backup": round(backup, 2),
            "total": round(compute + storage + backup, 2)}


def pinecone_serverless_cost(n: int, queries: int = QUERIES_PER_MONTH) -> dict:
    data_gb = n * BYTES_RECORD / GB
    storage = data_gb * PC_STORAGE_PER_GB
    ru_per_query = max(PC_RU_MIN_PER_QUERY, data_gb * PC_RU_PER_GB_PER_QUERY)
    reads = queries * ru_per_query / 1e6 * PC_PER_M_RU
    # churned records: pay for the new record's KB + the overwritten record's KB
    wu = n * MONTHLY_CHURN * 2 * (BYTES_RECORD / 1000) * PC_WU_PER_KB
    writes = wu / 1e6 * PC_PER_M_WU
    usage = storage + reads + writes
    return {"data_gb": round(data_gb, 2), "ru_per_query": round(ru_per_query, 2),
            "storage": round(storage, 2), "reads": round(reads, 2), "writes": round(writes, 2),
            "usage": round(usage, 2), "total": round(max(PC_MIN_MONTHLY, usage), 2)}


def pinecone_pods_cost(n: int, pod: str) -> dict:
    pods = max(1, math.ceil(n * BYTES_RECORD / POD_CAPACITY_BYTES[pod])) * POD_REPLICAS
    return {"pod": pod, "pods": pods, "total": round(pods * POD_PRICE_HR[pod] * HOURS_PER_MONTH, 2)}


def breakeven_queries(n: int) -> int | None:
    """Monthly query volume at which Pinecone serverless costs the same as our LanceDB box."""
    target = lancedb_cost(n)["total"]
    base = pinecone_serverless_cost(n, queries=0)["usage"]
    ru_per_query = max(PC_RU_MIN_PER_QUERY, n * BYTES_RECORD / GB * PC_RU_PER_GB_PER_QUERY)
    per_query = ru_per_query / 1e6 * PC_PER_M_RU
    if target <= PC_MIN_MONTHLY and base <= PC_MIN_MONTHLY:
        # serverless never drops below $50; we are cheaper at any volume if our box < $50
        return 0
    return max(0, math.ceil((target - base) / per_query))


def main() -> None:
    rows = []
    for n in SCALES:
        ours = lancedb_cost(n)
        sl = pinecone_serverless_cost(n)
        p1 = pinecone_pods_cost(n, "p1.x1")
        s1 = pinecone_pods_cost(n, "s1.x1")
        sens = {q: pinecone_serverless_cost(n, q)["total"] for q in (10_000, 100_000, 1_000_000, 10_000_000)}
        rows.append({"vectors": n, "lancedb": ours, "pinecone_serverless": sl,
                     "pinecone_p1_pods": p1, "pinecone_s1_pods": s1,
                     "serverless_by_query_volume": sens, "breakeven_queries_per_month": breakeven_queries(n)})

    assumptions = {k: v for k, v in globals().items() if k.isupper()}
    Path("results").mkdir(exist_ok=True)
    Path("results/cost_comparison.json").write_text(
        json.dumps({"assumptions": assumptions, "rows": rows}, indent=2), encoding="utf-8")

    md = ["| Vectors | LanceDB self-hosted | Pinecone p1 pods | Pinecone s1 pods | Pinecone serverless (100K q/mo) | Savings vs p1 pods | Savings vs serverless |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        o, p1, s1, sl = (r["lancedb"]["total"], r["pinecone_p1_pods"]["total"],
                         r["pinecone_s1_pods"]["total"], r["pinecone_serverless"]["total"])
        md.append(f"| {r['vectors']:,} | ${o:,.2f} ({r['lancedb']['instance']}, {r['lancedb']['disk_gb']} GB gp3) "
                  f"| ${p1:,.2f} ({r['pinecone_p1_pods']['pods']} pod) | ${s1:,.2f} ({r['pinecone_s1_pods']['pods']} pod) "
                  f"| ${sl:,.2f} ({r['pinecone_serverless']['data_gb']} GB, {r['pinecone_serverless']['ru_per_query']} RU/q) "
                  f"| {100 * (1 - o / p1):.0f}% | {100 * (1 - o / sl):.0f}% |")
    md += ["", "Pinecone serverless monthly cost by query volume (LanceDB cost is flat in query volume until the box saturates):", "",
           "| Vectors | 10K q/mo | 100K q/mo | 1M q/mo | 10M q/mo | LanceDB | Serverless is cheaper below |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["serverless_by_query_volume"]
        b = r["breakeven_queries_per_month"]
        be = "never ($50/mo floor > LanceDB)" if b == 0 else f"{b:,} queries/mo"
        md.append(f"| {r['vectors']:,} | ${s[10_000]:,.2f} | ${s[100_000]:,.2f} | ${s[1_000_000]:,.2f} | "
                  f"${s[10_000_000]:,.2f} | ${r['lancedb']['total']:,.2f} | {be} |")
    Path("results/cost_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
