#!/usr/bin/env python3
"""
Estimate required number of *samples*, where:
  1 sample = 1 observed (non-suppressed) noisy count Y_{q,k} for a single class k in a query q.

Heuristic:
1) Pick bucket size m so suppression is rare:
     p_sup(m) = P(Binom(m,1/K) < T)
   Require p_sup(m) <= delta0 / K  (so P(any class suppressed) <= ~delta0 by union bound)

2) Pick appearances per row:
     L = ceil(cprime * e^2 * log(R/delta))

3) Coverage:
     Q ≈ ceil((R/m) * L)

4) Samples:
     expected samples/query ≈ K * (1 - p_sup(m))
     N_samples ≈ Q * K * (1 - p_sup(m))
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

from scipy.stats import binom


@dataclass(frozen=True)
class SampleEstimate:
    m_bucket: int
    p_suppress_per_class: float
    expected_samples_per_query: float
    L_appearances_per_row: int
    Q_queries: int
    N_samples: int


def p_suppress_per_class(m: int, K: int, T: int) -> float:
    """p_sup(m) = P(X < T) where X ~ Binomial(m, 1/K)."""
    return float(binom.cdf(T - 1, m, 1.0 / K))


def find_min_m_for_low_suppression(
    K: int,
    T: int,
    delta0: float,
    m_start: Optional[int] = None,
    m_cap: int = 10_000_000,
) -> tuple[int, float]:
    """
    Find smallest m such that p_sup(m) <= delta0/K.
    Returns (m, p_sup(m)).
    """
    if K <= 0:
        raise ValueError("K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (0.0 < delta0 < 1.0):
        raise ValueError("delta0 must be in (0,1).")

    target = delta0 / K

    if m_start is None:
        # Reasonable initial guess: expected per-class count m/K at least T
        m_start = max(K * T, 1)

    lo = max(0, m_start - 1)
    hi = m_start

    p_hi = p_suppress_per_class(hi, K, T)
    while p_hi > target:
        lo = hi
        hi *= 2
        if hi > m_cap:
            raise RuntimeError(
                f"Could not find m <= {m_cap} meeting p_sup(m) <= delta0/K. "
                f"Try increasing delta0 or check (K,T)."
            )
        p_hi = p_suppress_per_class(hi, K, T)

    # Binary search for minimal m in (lo, hi]
    best = hi
    best_p = p_hi
    left = lo + 1
    right = hi
    while left <= right:
        mid = (left + right) // 2
        p_mid = p_suppress_per_class(mid, K, T)
        if p_mid <= target:
            best = mid
            best_p = p_mid
            right = mid - 1
        else:
            left = mid + 1

    return best, best_p


def estimate_samples_needed(
    R: int,
    K: int,
    T: int,
    e: int,
    delta: float = 0.01,
    delta0: float = 0.01,
    cprime: float = 8.0,
    m_override: Optional[int] = None,
) -> SampleEstimate:
    """
    If m_override is provided, uses that bucket size m directly.
    Otherwise chooses m via the suppression heuristic.
    """
    if R <= 0:
        raise ValueError("R must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if e < 0:
        raise ValueError("e must be >= 0.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    if not (0.0 < delta0 < 1.0):
        raise ValueError("delta0 must be in (0,1).")
    if cprime <= 0:
        raise ValueError("cprime must be positive.")

    if m_override is None:
        m, p_sup = find_min_m_for_low_suppression(K=K, T=T, delta0=delta0)
    else:
        m = int(m_override)
        if m <= 0:
            raise ValueError("m_override must be positive.")
        p_sup = p_suppress_per_class(m, K, T)

    # Per-row redundancy to average down bounded noise
    L = int(math.ceil(cprime * (e ** 2) * math.log(R / delta)))

    # Coverage approximation: each query touches ~m rows
    Q = int(math.ceil((R / m) * L))

    # Expected observed class-entries (samples) per query
    exp_samples_per_query = K * (1.0 - p_sup)

    # Total samples (rounded up from expectation)
    N_samples = int(math.ceil(Q * exp_samples_per_query))

    return SampleEstimate(
        m_bucket=m,
        p_suppress_per_class=p_sup,
        expected_samples_per_query=exp_samples_per_query,
        L_appearances_per_row=L,
        Q_queries=Q,
        N_samples=N_samples,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate number of samples needed.")
    ap.add_argument("--R", type=int, required=True, help="Number of rows R")
    ap.add_argument("--K", type=int, required=True, help="Number of target classes K")
    ap.add_argument("--T", type=int, required=True, help="Suppression threshold T")
    ap.add_argument("--e", type=int, required=True, help="Noise bound e (integer)")
    ap.add_argument("--delta", type=float, default=0.01, help="Overall failure prob (default 0.01)")
    ap.add_argument("--delta0", type=float, default=0.01, help="Per-query suppression budget (default 0.01)")
    ap.add_argument("--cprime", type=float, default=8.0, help="Conservativeness const (default 8.0)")
    ap.add_argument("--m", type=int, default=None, help="Override bucket size m (optional)")
    args = ap.parse_args()

    est = estimate_samples_needed(
        R=args.R, K=args.K, T=args.T, e=args.e,
        delta=args.delta, delta0=args.delta0, cprime=args.cprime,
        m_override=args.m,
    )

    print("=== Sample budget estimate ===")
    print(f"bucket size m:                         {est.m_bucket}")
    print(f"p_suppress_per_class (Binom tail):     {est.p_suppress_per_class:.3e}")
    print(f"expected samples per query:            {est.expected_samples_per_query:.2f} (out of K={args.K})")
    print(f"L appearances per row:                 {est.L_appearances_per_row}")
    print(f"Q queries (coverage approx):           {est.Q_queries}")
    print(f"TOTAL samples N_samples (approx):      {est.N_samples}")


if __name__ == "__main__":
    main()
