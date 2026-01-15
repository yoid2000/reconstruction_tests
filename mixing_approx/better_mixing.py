from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import math
import numpy as np
from scipy.stats import binom, norm


# -----------------------------
# Helpers
# -----------------------------

def _kl_bern(a: float, p: float) -> float:
    """KL(a || p) for Bernoulli(a) vs Bernoulli(p)."""
    eps = 1e-300
    a = min(1.0 - eps, max(eps, a))
    p = min(1.0 - eps, max(eps, p))
    return float(a * np.log(a / p) + (1.0 - a) * np.log((1.0 - a) / (1.0 - p)))


def _normalize_probs(class_probs: Optional[List[float]], K: int) -> np.ndarray:
    if class_probs is None:
        probs = np.full(K, 1.0 / K, dtype=float)
    else:
        probs = np.asarray(class_probs, dtype=float)
        if probs.ndim != 1 or len(probs) != K:
            raise ValueError("class_probs must be length K.")
        if np.any(probs < 0):
            raise ValueError("class_probs must be nonnegative.")
        s = probs.sum()
        if s <= 0:
            raise ValueError("class_probs must sum to > 0.")
        probs = probs / s
    return probs


# -----------------------------
# Release probabilities for multiclass
# -----------------------------

def _mean_var_released_counts_given_s(s: int, T: int, class_probs: np.ndarray) -> Tuple[float, float]:
    """
    Z(s) = number of released class-counts among K classes for a group of size s.
    Each class r is released iff U_r >= T.
    Under Multinomial(s, pi), marginally U_r ~ Binomial(s, pi_r).

    We compute:
      E[Z|s] = sum_r P(Binomial(s, pi_r) >= T)
      Var(Z|s) approximated as sum_r p_r(1-p_r) (ignores covariance).
    """
    if s <= 0:
        return 0.0, 0.0
    if T <= 0:
        # all released
        K = len(class_probs)
        return float(K), 0.0

    # p_r = P(U_r >= T) = binom.sf(T-1, s, pi_r)
    p = binom.sf(T - 1, s, class_probs)
    mean = float(np.sum(p))
    var = float(np.sum(p * (1.0 - p)))  # independence approximation
    return mean, var


def _mean_var_ownclass_release_given_s(s: int, T: int, class_probs: np.ndarray) -> Tuple[float, float]:
    """
    Indicator I_own(s): whether the count for the *row's own class* is released.

    Condition on row i having class r ~ class_probs.
    In a group of size s that includes row i, the count for class r is:
        U_r = 1 + Binomial(s-1, pi_r)   (approx; exact is multinomial but this marginal is correct)
    So I_own = 1{U_r >= T} = 1{Binomial(s-1, pi_r) >= T-1}.

    We compute:
      E[I_own | s] = sum_r pi_r * P(Binomial(s-1, pi_r) >= T-1)
      Var(I_own | s) = E[I_own|s] (1 - E[I_own|s])  (Bernoulli)
    """
    if s <= 0:
        return 0.0, 0.0
    if T <= 0:
        return 1.0, 0.0

    n = s - 1
    if n < 0:
        return 0.0, 0.0

    # P(Binomial(n, pi_r) >= T-1) = binom.sf((T-1)-1, n, pi_r) = binom.sf(T-2, n, pi_r)
    if T == 1:
        # need >=0 always true
        pr = np.ones_like(class_probs)
    else:
        pr = binom.sf(T - 2, n, class_probs)

    mean = float(np.sum(class_probs * pr))
    var = float(mean * (1.0 - mean))
    return mean, var


# -----------------------------
# Mixing measures
# -----------------------------

@dataclass(frozen=True)
class MultiMixingResult:
    d_eff: float
    d_min: float
    d_true_eff: float
    d_true_min: float
    I_eff: float
    I_min: float
    I_true_eff: float
    I_true_min: float
    var_eff: float
    var_true: float


def mixing_measures_multiclass(
    R: int,
    C: int,
    D: int,
    K: int,
    T: int,
    e: int,
    class_probs: Optional[List[float]] = None,
    pmf_mass: float = 1.0 - 1e-12,
) -> MultiMixingResult:
    """
    Computes multiclass extensions of:
      - d_eff / d_min: released counts (all classes) coverage per row
      - d_true_eff / d_true_min: released own-class count coverage per row

    Uses:
      S_k = 1 + Binomial(R-1, D^{-k}) for the group size distribution
      and binomial marginals for class counts.

    Variances are approximations (ignore covariance across classes and across queries).
    """
    if R < 2:
        return MultiMixingResult(
            d_eff=0.0, d_min=0.0, d_true_eff=0.0, d_true_min=0.0,
            I_eff=0.0, I_min=0.0, I_true_eff=0.0, I_true_min=0.0,
            var_eff=0.0, var_true=0.0
        )
    if C < 0 or D <= 0 or K <= 1 or T < 0 or e < 0:
        raise ValueError("Require C>=0, D>0, K>1, T>=0, e>=0.")
    if not (0.0 < pmf_mass < 1.0):
        raise ValueError("pmf_mass must be in (0,1).")

    pi = _normalize_probs(class_probs, K)

    n = R - 1
    d_eff = 0.0
    var_eff_total = 0.0

    d_true = 0.0
    var_true_total = 0.0

    # cache by s
    cache_all: Dict[int, Tuple[float, float]] = {}
    cache_own: Dict[int, Tuple[float, float]] = {}

    def moments_all(s: int) -> Tuple[float, float]:
        if s not in cache_all:
            cache_all[s] = _mean_var_released_counts_given_s(s, T, pi)
        return cache_all[s]

    def moments_own(s: int) -> Tuple[float, float]:
        if s not in cache_own:
            cache_own[s] = _mean_var_ownclass_release_given_s(s, T, pi)
        return cache_own[s]

    for k in range(0, C + 1):
        n_k = math.comb(C, k)  # number of row-specific queries at this k

        if k == 0:
            # S = R deterministically
            mean_all, var_all = moments_all(R)
            mean_own, var_own = moments_own(R)

            mean_k_all = mean_all
            var_k_all = var_all

            mean_k_own = mean_own
            var_k_own = var_own

        else:
            p_k = D ** (-k)

            lo, hi = binom.interval(pmf_mass, n=n, p=p_k)
            lo = int(max(0, lo))
            hi = int(min(n, hi))

            xs = np.arange(lo, hi + 1, dtype=int)
            pmf = binom.pmf(xs, n=n, p=p_k)
            s_pmf = float(pmf.sum())
            if not np.isfinite(s_pmf) or s_pmf <= 0.0:
                # fallback for extreme regimes
                xs = np.arange(0, min(n, 5) + 1, dtype=int)
                pmf = binom.pmf(xs, n=n, p=p_k)
                s_pmf = float(pmf.sum())
                if s_pmf <= 0.0:
                    mu = n * p_k
                    x0 = int(round(mu))
                    xs = np.array([x0], dtype=int)
                    pmf = np.array([1.0], dtype=float)
                    s_pmf = 1.0

            pmf = pmf / s_pmf

            # Integrate over s = x + 1
            mean_k_all = 0.0
            second_k_all = 0.0

            mean_k_own = 0.0
            second_k_own = 0.0

            for x, px in zip(xs, pmf):
                s = int(x) + 1

                m_all, v_all = moments_all(s)
                m_own, v_own = moments_own(s)

                # for "all classes": conditional second moment = var + mean^2
                mean_k_all += float(px) * m_all
                second_k_all += float(px) * (v_all + m_all * m_all)

                # for "own class": Bernoulli, same structure
                mean_k_own += float(px) * m_own
                second_k_own += float(px) * (v_own + m_own * m_own)

            var_k_all = max(0.0, second_k_all - mean_k_all * mean_k_all)
            var_k_own = max(0.0, second_k_own - mean_k_own * mean_k_own)

        d_eff += n_k * mean_k_all
        var_eff_total += n_k * var_k_all  # independence approximation across queries

        d_true += n_k * mean_k_own
        var_true_total += n_k * var_k_own

    # Lower-quantile proxies using Normal approx
    def lower_quantile(mean: float, var: float) -> float:
        sigma = float(np.sqrt(max(0.0, var)))
        if sigma == 0.0:
            return float(mean)
        z = float(norm.ppf(1.0 / R))
        return float(max(0.0, mean + sigma * z))

    d_min = lower_quantile(d_eff, var_eff_total)
    d_true_min = lower_quantile(d_true, var_true_total)

    denom = 2 * e + 1
    return MultiMixingResult(
        d_eff=float(d_eff),
        d_min=float(d_min),
        d_true_eff=float(d_true),
        d_true_min=float(d_true_min),
        I_eff=float(d_eff / denom),
        I_min=float(d_min / denom),
        I_true_eff=float(d_true / denom),
        I_true_min=float(d_true_min / denom),
        var_eff=float(var_eff_total),
        var_true=float(var_true_total),
    )


# -----------------------------
# Accuracy helper (multiclass)
# -----------------------------

@dataclass(frozen=True)
class AccuracyStatement:
    alpha: float
    beta_target: float
    beta_bound: float
    meets_target: bool
    p_row: float
    I_true_min: float
    statement: str


def accuracy_confidence_statement_multiclass(
    R: int,
    K: int,
    d_true_min: float,
    e: int,
    alpha: float = 0.01,
    beta_target: float = 0.01,
) -> AccuracyStatement:
    """
    Cheap model-based screen for multiclass accuracy:

    - Use own-class coverage (d_true_min) because any wrong label changes the own-class count by -1
      in every query containing the row.

    - Per released own-class count, a wrong alternative survives with probability q = 1 - 1/(2e+1).
      After d tests, survival per alternative ~ exp(-d/(2e+1)).
      Union bound over (K-1) alternatives: p_row <= (K-1) * exp(-d/(2e+1)).

    - Treat row errors as ~Binomial(R, p_row) and apply Chernoff:
        P(E >= alpha R) <= exp( -R * KL(alpha || p_row) )   when p_row < alpha.
    """
    if R <= 0:
        raise ValueError("R must be positive.")
    if K <= 1:
        raise ValueError("K must be > 1.")
    if e < 0:
        raise ValueError("e must be nonnegative.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if not (0.0 < beta_target < 1.0):
        raise ValueError("beta_target must be in (0,1).")

    denom = 2 * e + 1
    I_true_min = float(d_true_min) / denom

    # Union bound over K-1 incorrect labels
    p_row = float(min(1.0, (K - 1) * np.exp(-I_true_min)))

    if p_row < alpha:
        beta_bound = float(np.exp(-R * _kl_bern(alpha, p_row)))
    else:
        beta_bound = 1.0

    meets = beta_bound <= beta_target

    statement = (
        f"Under the i.i.d. model + independence approximations: "
        f"P(accuracy < {1.0 - alpha:.2%}) ≤ {beta_bound:.3e}. "
        f"Equivalently, with probability ≥ {1.0 - beta_bound:.2%}, "
        f"accuracy is at least {1.0 - alpha:.2%}."
    )
    if meets:
        statement += f" This meets the requested (alpha={alpha}, beta={beta_target})."
    else:
        statement += f" This does NOT meet the requested (alpha={alpha}, beta={beta_target})."

    return AccuracyStatement(
        alpha=float(alpha),
        beta_target=float(beta_target),
        beta_bound=float(beta_bound),
        meets_target=bool(meets),
        p_row=float(p_row),
        I_true_min=float(I_true_min),
        statement=statement,
    )


def mixing_and_accuracy_report_multiclass(
    R: int,
    C: int,
    D: int,
    K: int,
    T: int,
    e: int,
    class_probs: Optional[List[float]] = None,
    alpha: float = 0.01,
    beta_target: float = 0.01,
    pmf_mass: float = 1.0 - 1e-12,
) -> Dict[str, float | bool | str]:
    mm = mixing_measures_multiclass(
        R=R, C=C, D=D, K=K, T=T, e=e, class_probs=class_probs, pmf_mass=pmf_mass
    )
    acc = accuracy_confidence_statement_multiclass(
        R=R, K=K, d_true_min=mm.d_true_min, e=e, alpha=alpha, beta_target=beta_target
    )
    return {
        # mixing measures
        "d_eff": mm.d_eff,
        "d_min": mm.d_min,
        "d_true_eff": mm.d_true_eff,
        "d_true_min": mm.d_true_min,
        "I_eff": mm.I_eff,
        "I_min": mm.I_min,
        "I_true_eff": mm.I_true_eff,
        "I_true_min": mm.I_true_min,
        "var_eff": mm.var_eff,
        "var_true": mm.var_true,
        # accuracy statement
        "alpha": acc.alpha,
        "beta_target": acc.beta_target,
        "beta_bound": acc.beta_bound,
        "meets_target": acc.meets_target,
        "p_row": acc.p_row,
        "statement": acc.statement,
    }


# Example
if __name__ == "__main__":
    print("\n=========================================================")
    print("Case 1: should not meet target accuracy")
    report = mixing_and_accuracy_report_multiclass(
        R=150, C=4, D=4, K=2, T=3, e=2,
        class_probs=None,     # uniform over 5 classes
        alpha=0.01, beta_target=0.01
    )
    for k in ["d_true_min", "I_true_min", "p_row", "beta_bound", "meets_target"]:
        print(f"{k}: {report[k]}")
    print(report["statement"])

    print("\n=========================================================")
    print("Case 2: should meet target accuracy")
    report = mixing_and_accuracy_report_multiclass(
        R=150, C=7, D=3, K=2, T=3, e=2,
        class_probs=None,     # uniform over 5 classes
        alpha=0.01, beta_target=0.01
    )
    for k in ["d_true_min", "I_true_min", "p_row", "beta_bound", "meets_target"]:
        print(f"{k}: {report[k]}")
    print(report["statement"])