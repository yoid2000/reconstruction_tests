from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------
# Binomial helpers (SciPy optional)
# ---------------------------

try:
    from scipy.stats import binom as _scipy_binom  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    _scipy_binom = None

def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def binom_logpmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return float("-inf")
    if p <= 0.0:
        return 0.0 if k == 0 else float("-inf")
    if p >= 1.0:
        return 0.0 if k == n else float("-inf")
    return _log_choose(n, k) + k * math.log(p) + (n - k) * math.log(1.0 - p)

def binom_pmf(k: int, n: int, p: float) -> float:
    lp = binom_logpmf(k, n, p)
    return 0.0 if lp == float("-inf") else math.exp(lp)

def binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if _HAVE_SCIPY:
        return float(_scipy_binom.cdf(k, n, p))
    # fallback: sum pmf up to k (OK for moderate n; if huge, install scipy)
    return sum(binom_pmf(i, n, p) for i in range(0, k + 1))

# ---------------------------
# Model components
# ---------------------------

def pS_uniform_D(D: int, s: int) -> float:
    return (1.0 / D) ** s if s > 0 else 1.0

def noise_support(e: int) -> Tuple[int, int]:
    return (-e, e)

def noise_prob(u: int, e: int) -> float:
    if e <= 0:
        return 1.0 if u == 0 else 0.0
    return 1.0 / (2 * e + 1) if (-e <= u <= e) else 0.0

def safe_log(x: float) -> float:
    return math.log(x) if x > 0.0 else float("-inf")

@dataclass
class LikelihoodResult:
    loglik_total: float
    n_terms: int
    avg_loglik: float
    details: Dict[str, Any]

# ---------------------------
# Likelihood terms for one query
# suppress-then-noise
# ---------------------------

def logprob_group_size(R: int, pS: float, g_total: int) -> float:
    """
    g_total = 1 + Binom(R-1, pS)
    """
    n = R - 1
    x = g_total - 1
    return binom_logpmf(x, n, pS)

def logprob_class_observation(
    *,
    R: int,
    pS: float,
    qk: float,
    T: int,
    e: int,
    offset: int,     # 1 for own-class in row-anchored query, else 0
    obs: Optional[int],  # None means suppressed; int means released noisy count
) -> float:
    """
    Under model:
      X ~ Binom(R-1, pS*qk)  (count from other rows of class k)
      true_count = offset + X
      suppression: true_count < T
      release: true_count >= T, then observed = true_count + U, U~Unif{-e..e}
    """
    n = R - 1
    p = pS * qk

    # suppression event
    if obs is None:
        # P(offset + X < T) = P(X <= T - offset - 1)
        thr = T - offset - 1
        return safe_log(binom_cdf(thr, n, p))

    # released with observed noisy count
    y = int(obs)

    # Need P(offset+X >= T AND y = offset+X+U)
    # For a given X=x, U = y - (offset+x) must be in [-e,e].
    # So x must be in [y-offset-e, y-offset+e], and also x >= T-offset.
    x_min = max(0, T - offset, y - offset - e)
    x_max = min(n, y - offset + e)
    if x_min > x_max:
        return float("-inf")

    unif = 1.0 if e == 0 else 1.0 / (2 * e + 1)
    tot = 0.0
    for x in range(x_min, x_max + 1):
        u = y - (offset + x)
        if -e <= u <= e:
            tot += binom_pmf(x, n, p) * unif

    return safe_log(tot)

# ---------------------------
# Compute likelihood over a query log
# ---------------------------

def layerA_loglikelihood(
    *,
    R: int,
    C: int,
    D: int,
    K: int,
    T: int,
    e: int,
    queries: Sequence[Dict[str, Any]],
    class_probs: Optional[Sequence[float]] = None,
    require_anchor_for_offset: bool = False,
) -> LikelihoodResult:
    """
    Computes exact log-likelihood of observed (g_total, y vector with suppression) under Layer-A model.

    If require_anchor_for_offset=True and a query lacks anchor_class, offset is treated as 0 for all classes
    AND we record it as a warning in details.
    """
    if class_probs is None:
        q = np.full(K, 1.0 / K, dtype=float)
    else:
        q = np.asarray(class_probs, dtype=float)
        q = q / q.sum()

    ll = 0.0
    n_terms = 0
    missing_anchor = 0

    per_query_ll: List[float] = []

    for rec in queries:
        s = int(rec["s"])
        if not (0 <= s <= C):
            raise ValueError(f"s={s} outside [0,C]={C}")
        pS = pS_uniform_D(D, s)

        q_ll = 0.0

        # group size term (recommended; you said you can supply it)
        g_total = rec.get("g_total", None)
        if g_total is not None:
            q_ll += logprob_group_size(R, pS, int(g_total))
            n_terms += 1

        y = rec["y"]
        if len(y) != K:
            raise ValueError("Each record['y'] must have length K")

        anchor = rec.get("anchor_class", None)
        if anchor is None:
            if require_anchor_for_offset:
                missing_anchor += 1
            anchor = None
        else:
            anchor = int(anchor)
            if not (0 <= anchor < K):
                raise ValueError("anchor_class out of range")

        for k in range(K):
            offset = 1 if (anchor is not None and k == anchor) else 0
            obs = None if (y[k] is None) else int(y[k])
            q_ll += logprob_class_observation(
                R=R, pS=pS, qk=float(q[k]), T=T, e=e, offset=offset, obs=obs
            )
            n_terms += 1

        ll += q_ll
        per_query_ll.append(q_ll)

    avg = ll / n_terms if n_terms else float("nan")
    details = {
        "n_queries": len(queries),
        "missing_anchor_count": missing_anchor,
        "per_query_loglik": per_query_ll[:2000],  # cap to avoid huge objects
        "note_per_query_loglik_capped": len(per_query_ll) > 2000,
    }
    return LikelihoodResult(loglik_total=ll, n_terms=n_terms, avg_loglik=avg, details=details)

# ---------------------------
# Parametric bootstrap "plausibility" score
# ---------------------------

def simulate_query_under_model(
    *,
    rng: np.random.Generator,
    R: int,
    C: int,
    D: int,
    K: int,
    T: int,
    e: int,
    s: int,
    anchor_class: Optional[int],
    class_probs: np.ndarray,
    include_g_total: bool = True,
) -> Dict[str, Any]:
    """
    Simulates (g_total, y) under the Layer-A model, suppress-then-noise.
    """
    n = R - 1
    pS = pS_uniform_D(D, s)

    # group size
    g_total = None
    if include_g_total:
        xg = rng.binomial(n, pS)
        g_total = 1 + int(xg)

    y: List[Optional[int]] = []
    for k in range(K):
        offset = 1 if (anchor_class is not None and k == anchor_class) else 0
        p = pS * float(class_probs[k])
        x = int(rng.binomial(n, p))
        true_ct = offset + x
        if true_ct < T:
            y.append(None)
        else:
            u = int(rng.integers(-e, e + 1)) if e > 0 else 0
            y.append(true_ct + u)

    out = {"s": s, "y": y}
    if include_g_total:
        out["g_total"] = g_total
    if anchor_class is not None:
        out["anchor_class"] = int(anchor_class)
    return out

def layerA_model_plausibility(
    *,
    R: int,
    C: int,
    D: int,
    K: int,
    T: int,
    e: int,
    queries: Sequence[Dict[str, Any]],
    class_probs: Optional[Sequence[float]] = None,
    n_boot: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Returns:
      - observed avg log-likelihood
      - bootstrap distribution of avg log-likelihood under the model
      - p_value = P_boot( avg_loglik_sim <= avg_loglik_obs )
        (small p => observed data is unusually unlikely under the model)
    Each query in `queries` should be a dict like:
    {
      "s": 4,                    # subset size |S|
      "g_total": 23,             # true total matching rows for this query (includes the anchor row)
      "y": [12, None, 3, None],  # length K: noisy released count (int) or None if suppressed
    }

    """
    if class_probs is None:
        q = np.full(K, 1.0 / K, dtype=float)
    else:
        q = np.asarray(class_probs, dtype=float)
        q = q / q.sum()

    obs = layerA_loglikelihood(
        R=R, C=C, D=D, K=K, T=T, e=e, queries=queries, class_probs=q
    )

    rng = np.random.default_rng(seed)
    sim_avgs = []

    # We simulate using the same (s, anchor_class present?) pattern as the provided queries.
    for b in range(n_boot):
        sim_queries = []
        for rec in queries:
            s = int(rec["s"])
            anchor = rec.get("anchor_class", None)
            anchor = int(anchor) if anchor is not None else None
            include_g_total = ("g_total" in rec and rec["g_total"] is not None)
            sim_queries.append(
                simulate_query_under_model(
                    rng=rng, R=R, C=C, D=D, K=K, T=T, e=e,
                    s=s, anchor_class=anchor, class_probs=q,
                    include_g_total=include_g_total
                )
            )

        sim_ll = layerA_loglikelihood(
            R=R, C=C, D=D, K=K, T=T, e=e, queries=sim_queries, class_probs=q
        )
        sim_avgs.append(sim_ll.avg_loglik)

    sim_avgs_arr = np.array(sim_avgs, dtype=float)

    # one-sided: how often model produces avg_loglik <= observed (more negative => worse fit)
    p_value = float(np.mean(sim_avgs_arr <= obs.avg_loglik))

    return {
        "observed": {
            "loglik_total": obs.loglik_total,
            "n_terms": obs.n_terms,
            "avg_loglik": obs.avg_loglik,
        },
        "bootstrap": {
            "n_boot": int(n_boot),
            "avg_loglik_samples": sim_avgs[:2000],  # cap
            "note_samples_capped": len(sim_avgs) > 2000,
            "avg_loglik_mean": float(sim_avgs_arr.mean()),
            "avg_loglik_std": float(sim_avgs_arr.std(ddof=1)) if len(sim_avgs_arr) > 1 else float("nan"),
            "p_value": p_value,
            "quantiles": {
                "1%": float(np.quantile(sim_avgs_arr, 0.01)),
                "5%": float(np.quantile(sim_avgs_arr, 0.05)),
                "50%": float(np.quantile(sim_avgs_arr, 0.50)),
                "95%": float(np.quantile(sim_avgs_arr, 0.95)),
                "99%": float(np.quantile(sim_avgs_arr, 0.99)),
            },
        },
        "interpretation": (
            "p_value small (e.g., <0.01) suggests the observed query outcomes are "
            "unusually unlikely under the Layer-A model assumptions (uniform features/cardinality, "
            "independence, class_probs, suppress-then-noise with radius e)."
        ),
    }
