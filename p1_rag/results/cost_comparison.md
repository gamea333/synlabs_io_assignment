| Vectors | LanceDB self-hosted | Pinecone p1 pods | Pinecone s1 pods | Pinecone serverless (100K q/mo) | Savings vs p1 pods | Savings vs serverless |
|---|---|---|---|---|---|---|
| 100,000 | $26.14 (t4g.medium, 20 GB gp3) | $81.03 (1 pod) | $81.03 (1 pod) | $50.00 (0.37 GB, 0.37 RU/q) | 68% | 48% |
| 1,000,000 | $26.21 (t4g.medium, 20 GB gp3) | $162.06 (2 pod) | $81.03 (1 pod) | $50.00 (3.73 GB, 3.73 RU/q) | 84% | 48% |
| 10,000,000 | $83.52 (r7g.large, 56 GB gp3) | $1,053.39 (13 pod) | $243.09 (3 pod) | $101.83 (37.3 GB, 37.3 RU/q) | 92% | 18% |

Pinecone serverless monthly cost by query volume (LanceDB cost is flat in query volume until the box saturates):

| Vectors | 10K q/mo | 100K q/mo | 1M q/mo | 10M q/mo | LanceDB | Serverless is cheaper below |
|---|---|---|---|---|---|---|
| 100,000 | $50.00 | $50.00 | $50.00 | $60.10 | $26.14 | never ($50/mo floor > LanceDB) |
| 1,000,000 | $50.00 | $50.00 | $63.89 | $601.01 | $26.21 | never ($50/mo floor > LanceDB) |
| 10,000,000 | $50.00 | $101.83 | $638.95 | $6,010.15 | $83.52 | 69,320 queries/mo |
