from __future__ import annotations

import math
from typing import Literal


def _log_binom_pmf(n: int, k: int, p: float) -> float:
    """log( Binomial(n,k) * p^k * (1-p)^(n-k) ) computed stably."""
    if k < 0 or k > n:
        return float("-inf")
    if p == 0.0:
        return 0.0 if k == 0 else float("-inf")
    if p == 1.0:
        return 0.0 if k == n else float("-inf")
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binom_sf(n: int, p: float, t: int) -> float:
    """
    Survival function: P[Binomial(n,p) >= t], computed without scipy.
    Uses log-sum-exp over the smaller tail for stability.
    """
    if t <= 0:
        return 1.0
    if t > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0 if t <= n else 0.0

    # Decide which tail to sum: lower tail (0..t-1) or upper tail (t..n).
    # Then convert to survival prob.
    lower_count = t
    upper_count = n - t + 1

    def logsumexp(log_terms):
        m = max(log_terms)
        return m + math.log(sum(math.exp(x - m) for x in log_terms))

    if lower_count <= upper_count:
        # Compute F(t-1) then return 1 - F
        logs = [_log_binom_pmf(n, k, p) for k in range(0, t)]
        logF = logsumexp(logs)
        F = math.exp(logF)
        # Guard against tiny negative from rounding
        return max(0.0, 1.0 - F)
    else:
        # Compute upper tail directly
        logs = [_log_binom_pmf(n, k, p) for k in range(t, n + 1)]
        logU = logsumexp(logs)
        return min(1.0, math.exp(logU))


def expected_num_queries_ge_T(
    R: int,
    C: int,
    D: int,
    T: int,
    tail: Literal["binom"] = "binom",
) -> float:
    """
    Model B (uniform random distinct rows; binomial approximation):
      E[#queries with count >= T] =
        sum_{k=0..C} (C choose k) * D^k * P[Binomial(R, 1/D^k) >= T]

    Returns the expected number of qualifying queries.
    """
    if any(x < 0 for x in (R, C, D, T)):
        raise ValueError("R, C, D, T must be nonnegative (and D>0).")
    if D == 0:
        raise ValueError("D must be > 0.")
    if C == 0:
        # Only one query: empty constraint. Count is always R.
        return 1.0 if R >= T else 0.0
    if T <= 0:
        # Every query qualifies
        return float((1 + D) ** C)
    if T > R:
        return 0.0

    total = 0.0
    for k in range(0, C + 1):
        num_queries_k = math.comb(C, k) * (D ** k)
        p = 1.0 / (D ** k) if k > 0 else 1.0  # k=0 matches all rows
        prob = binom_sf(R, p, T)
        total += num_queries_k * prob
    return total


# Example:
if __name__ == "__main__":
    R, C, D, T = 1000, 6, 10, 3
    print(expected_num_queries_ge_T(R, C, D, T))
