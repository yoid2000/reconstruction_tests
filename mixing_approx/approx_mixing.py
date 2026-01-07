from __future__ import annotations

import math
from typing import Literal


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_hypergeom_pmf(N: int, K: int, n: int, x: int) -> float:
    # PMF: [C(K,x) * C(N-K, n-x)] / C(N,n)
    if x < 0 or x > n or x > K or n - x > N - K:
        return float("-inf")
    return _log_choose(K, x) + _log_choose(N - K, n - x) - _log_choose(N, n)


def _logsumexp(log_terms: list[float]) -> float:
    m = max(log_terms)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(t - m) for t in log_terms))


def hypergeom_sf(N: int, K: int, n: int, t: int) -> float:
    """P[X >= t] for X ~ Hypergeom(N,K,n), computed without scipy."""
    if t <= 0:
        return 1.0
    # Support bounds
    lo = max(0, n - (N - K))
    hi = min(n, K)
    if t > hi:
        return 0.0
    if t <= lo:
        return 1.0

    # Sum the smaller tail for stability
    lower_count = t - lo           # terms for x = lo .. t-1
    upper_count = hi - t + 1       # terms for x = t .. hi

    if lower_count <= upper_count:
        logs = [_log_hypergeom_pmf(N, K, n, x) for x in range(lo, t)]
        logF = _logsumexp(logs)
        F = math.exp(logF)
        return max(0.0, 1.0 - F)
    else:
        logs = [_log_hypergeom_pmf(N, K, n, x) for x in range(t, hi + 1)]
        logU = _logsumexp(logs)
        return min(1.0, math.exp(logU))


def binom_sf(n: int, p: float, t: int) -> float:
    """P[Binomial(n,p) >= t] without scipy (used as fallback)."""
    if t <= 0:
        return 1.0
    if t > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    def log_binom_pmf(n_: int, k_: int, p_: float) -> float:
        return (
            math.lgamma(n_ + 1) - math.lgamma(k_ + 1) - math.lgamma(n_ - k_ + 1)
            + k_ * math.log(p_)
            + (n_ - k_) * math.log1p(-p_)
        )

    # sum smaller tail
    lower_count = t
    upper_count = n - t + 1
    if lower_count <= upper_count:
        logs = [log_binom_pmf(n, k, p) for k in range(0, t)]
        logF = _logsumexp(logs)
        return max(0.0, 1.0 - math.exp(logF))
    else:
        logs = [log_binom_pmf(n, k, p) for k in range(t, n + 1)]
        logU = _logsumexp(logs)
        return min(1.0, math.exp(logU))


def expected_mixing_M(
    R: int, C: int, D: int, T: int,
    method: Literal["auto", "hypergeom", "binom"] = "auto",
) -> float:
    """
    Expected mixing:
      E[M] = sum_{k=0..C} comb(C,k) * p_match(k) * P(Y_k >= T-2),
    where p_match(k) = (D^(C-k)-1)/(D^C-1),
    and Y_k is Hypergeom(D^C-2, D^(C-k)-2, R-2) (exact),
    with binomial fallback when D^C too large for hypergeom via math.lgamma.
    """
    if any(x < 0 for x in (R, C, D, T)):
        raise ValueError("R, C, D, T must be nonnegative (and D>0).")
    if D <= 0:
        raise ValueError("D must be > 0.")
    if R < 2:
        return 0.0  # no row-pairs

    # If T is impossible, no qualifying queries, so mixing is 0.
    if T > R:
        return 0.0

    # Compute D^C, with overflow-aware fallback.
    DC = D ** C  # Python big-int OK

    # Decide method
    use_hyper = (method == "hypergeom")
    if method == "auto":
        try:
            # math.lgamma needs float-ish magnitudes; if conversion fails/overflows,
            # hypergeom will be numerically infeasible anyway -> use binomial approx.
            float(DC)
            use_hyper = True
        except OverflowError:
            use_hyper = False
    if method == "binom":
        use_hyper = False

    total = 0.0
    for k in range(0, C + 1):
        # Probability two distinct tuples match on a fixed set of k columns:
        # (D^(C-k)-1)/(D^C-1)
        d_ck = D ** (C - k)
        if DC <= 1:
            p_match = 0.0
        else:
            p_match = (d_ck - 1) / (DC - 1)

        if p_match == 0.0:
            continue

        num_subsets = math.comb(C, k)

        # Given the pair matches on those k columns, the query that includes both
        # is the unique assignment to those k columns. It qualifies iff total>=T,
        # i.e. Y_k >= T-2 additional matches among remaining rows.
        t_need = T - 2

        if use_hyper:
            N = DC - 2
            K = d_ck - 2
            n = R - 2
            # If K < 0, then no extra tuples can match; but p_match should already be 0 then.
            if K < 0 or N < 0:
                prob_ge = 0.0
            else:
                prob_ge = hypergeom_sf(N=N, K=K, n=n, t=t_need)
        else:
            # Binomial approximation: Y_k ~ Binomial(R-2, 1/D^k)
            n = R - 2
            p = 1.0 / (D ** k) if k > 0 else 1.0
            prob_ge = binom_sf(n=n, p=p, t=t_need)

        total += num_subsets * p_match * prob_ge

    return total


# Example usage:
if __name__ == "__main__":
    R, C, D, T = 8000, 3, 2, 2
    print(f"Expected mixing M for rows={R}, columns={C}, distinct={D}, suppress={T}:")
    print(expected_mixing_M(R, C, D, T))
