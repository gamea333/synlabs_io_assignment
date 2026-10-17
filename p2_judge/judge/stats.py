"""Small, dependency-free statistics used in reports."""
import math
import random
from collections import Counter


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stdev(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def entropy(xs) -> float:
    """Shannon entropy (bits) of a discrete score distribution — low = clustered."""
    c = Counter(xs)
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def cohen_kappa(a: list, b: list) -> float | None:
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial (sign) test, ties dropped. H0: P(win) = 0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(diffs: list[float], iters: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iters))
    return means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters) - 1]
