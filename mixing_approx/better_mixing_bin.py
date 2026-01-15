"""
Cleaned-up SciPy version (requires: numpy, scipy)

Computes:
  - d_eff: expected effective released degree per row (counts TRUE/FALSE releases separately)
  - d_min: conservative ~1/R lower-quantile proxy for per-row effective degree
  - (alpha, beta) statement: model-based bound on P(accuracy < 1-alpha)

Model:
  - Features i.i.d. uniform over D values per column
  - For k constrained columns, group size for a row-specific query:
        S_k = 1 + Binomial(R-1, D^{-k})
  - Targets i.i.d. Bernoulli(1/2) (used for suppression probabilities)
  - Suppression based on TRUE counts; True/False suppressed separately:
        release TRUE iff U >= T
        release FALSE iff (S-U) >= T  <=> U <= S-T
  - Noise: integer uniform in [-e, e] (used only in noise-normalized scores and accuracy bound)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import binom, norm


def _release_moments_given_group_size(s: int, T: int) -> Tuple[float, float]:
    """
    For group size s, with U ~ Binomial(s, 1/2):
      TRUE released iff U >= T
      FALSE released iff U <= s - T
    Let Z = #released counts in {0,1,2}.
    Return (E[Z], E[Z^2]).
    """
    if s <= 0:
        return 0.0, 0.0
    if T <= 0:
        # both always released
        return 2.0, 4.0

    # P(TRUE released) = P(U >= T) = 1 - P(U <= T-1)
    a = 1.0 - binom.cdf(T - 1, s, 0.5)

    # P(FALSE released) = P(U <= s-T)
    if s - T < 0:
        b = 0.0
    else:
        b = binom.cdf(s - T, s, 0.5)

    # P(both released) = P(T <= U <= s-T)
    if s >= 2 * T and (s - T) >= 0:
        p2 = binom.cdf(s - T, s, 0.5) - binom.cdf(T - 1, s, 0.5)
        p2 = float(max(0.0, min(1.0, p2)))
    else:
        p2 = 0.0

    # Z = I_true + I_false
    mean = a + b
    # Z^2 = I_true + I_false + 2 I_true I_false
    second = a + b + 2.0 * p2
    return float(mean), float(second)


def _kl_bern(a: float, p: float) -> float:
    """
    KL(a || p) for Bernoulli(a) vs Bernoulli(p):
      a log(a/p) + (1-a) log((1-a)/(1-p))
    """
    eps = 1e-300
    a = min(1.0 - eps, max(eps, a))
    p = min(1.0 - eps, max(eps, p))
    return a * np.log(a / p) + (1.0 - a) * np.log((1.0 - a) / (1.0 - p))


@dataclass(frozen=True)
class MixingResult:
    d_eff: float
    d_min: float
    I_eff: float
    I_min: float
    var_eff: float


def mixing_measures_scipy(
    R: int,
    C: int,
    D: int,
    T: int,
    e: int,
    pmf_mass: float = 1.0 - 1e-12,
) -> MixingResult:
    """
    Compute d_eff and d_min under the i.i.d. model using SciPy.

    pmf_mass:
      central mass of Binomial(R-1, D^{-k}) to include when integrating over group sizes.
      (1-1e-12) is usually plenty and keeps the support window small.
    """
    if R < 2:
        return MixingResult(d_eff=0.0, d_min=0.0, I_eff=0.0, I_min=0.0, var_eff=0.0)
    if C < 0 or D <= 0 or T < 0 or e < 0:
        raise ValueError("Require C>=0, D>0, T>=0, e>=0.")
    if not (0.0 < pmf_mass < 1.0):
        raise ValueError("pmf_mass must be in (0,1).")

    n = R - 1
    d_eff = 0.0
    var_total = 0.0

    # Cache moments by group size s to avoid recomputing binom CDFs repeatedly
    moments_cache: Dict[int, Tuple[float, float]] = {}

    def moments(s: int) -> Tuple[float, float]:
        if s not in moments_cache:
            moments_cache[s] = _release_moments_given_group_size(s, T)
        return moments_cache[s]

    for k in range(0, C + 1):
        n_k = math.comb(C, k)

        if k == 0:
            # p = 1 => X = n deterministically, S = R
            mZ, sZ = moments(R)
            mean_k, second_k = mZ, sZ
        else:
            p_k = D ** (-k)

            # Central interval capturing pmf_mass
            lo, hi = binom.interval(pmf_mass, n=n, p=p_k)
            lo = int(max(0, lo))
            hi = int(min(n, hi))

            xs = np.arange(lo, hi + 1, dtype=int)
            pmf = binom.pmf(xs, n=n, p=p_k)

            # In rare cases, interval can be too narrow (numerical edge); fall back to a tiny window
            s_pmf = float(pmf.sum())
            if not np.isfinite(s_pmf) or s_pmf <= 0.0:
                # fallback: include 0..min(n, 5) (works fine when p is extremely small)
                xs = np.arange(0, min(n, 5) + 1, dtype=int)
                pmf = binom.pmf(xs, n=n, p=p_k)
                s_pmf = float(pmf.sum())
                if s_pmf <= 0.0:
                    # ultimate fallback: point mass at round(mu)
                    mu = n * p_k
                    x0 = int(round(mu))
                    xs = np.array([x0], dtype=int)
                    pmf = np.array([1.0], dtype=float)
                    s_pmf = 1.0

            pmf = pmf / s_pmf

            # S = X + 1
            mean_k = 0.0
            second_k = 0.0
            for x, px in zip(xs, pmf):
                s = int(x) + 1
                mZ, sZ = moments(s)
                mean_k += float(px) * mZ
                second_k += float(px) * sZ

        var_k = max(0.0, second_k - mean_k * mean_k)

        d_eff += n_k * mean_k
        var_total += n_k * var_k  # independence approximation across queries

    sigma = float(np.sqrt(max(0.0, var_total)))
    if sigma == 0.0:
        d_min = d_eff
    else:
        z = float(norm.ppf(1.0 / R))
        d_min = max(0.0, d_eff + sigma * z)

    denom = 2 * e + 1
    I_eff = d_eff / denom
    I_min = d_min / denom

    return MixingResult(
        d_eff=float(d_eff),
        d_min=float(d_min),
        I_eff=float(I_eff),
        I_min=float(I_min),
        var_eff=float(var_total),
    )


@dataclass(frozen=True)
class AccuracyStatement:
    alpha: float
    beta_target: float
    beta_bound: float
    meets_target: bool
    p_row: float
    I_min: float
    statement: str


def accuracy_confidence_statement(
    R: int,
    d_min: float,
    e: int,
    alpha: float = 0.01,
    beta_target: float = 0.01,
) -> AccuracyStatement:
    """
    Model-based (cheap) accuracy screen:

      - Convert d_min to a per-row ambiguity/error proxy:
            p_row = exp(- d_min / (2e+1))

      - Approximate total errors E ~ Binomial(R, p_row) and use a Chernoff bound:
            P(E >= alpha R) <= exp(-R * KL(alpha || p_row))   when p_row < alpha

    Returns an explicit (alpha,beta) statement using beta_bound.
    """
    if R <= 0:
        raise ValueError("R must be positive.")
    if e < 0:
        raise ValueError("e must be nonnegative.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if not (0.0 < beta_target < 1.0):
        raise ValueError("beta_target must be in (0,1).")

    denom = 2 * e + 1
    I_min = float(d_min) / denom
    p_row = float(np.exp(-I_min))

    if p_row < alpha:
        beta_bound = float(np.exp(-R * _kl_bern(alpha, p_row)))
    else:
        beta_bound = 1.0  # bound is not informative if the mean error rate exceeds alpha

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
        I_min=float(I_min),
        statement=statement,
    )


def mixing_and_accuracy_report(
    R: int,
    C: int,
    D: int,
    T: int,
    e: int,
    alpha: float = 0.01,
    beta_target: float = 0.01,
    pmf_mass: float = 1.0 - 1e-12,
) -> Dict[str, float | bool | str]:
    """
    Convenience wrapper:
      - computes d_eff/d_min/I_eff/I_min
      - returns explicit (alpha,beta) accuracy statement
    """
    mm = mixing_measures_scipy(R=R, C=C, D=D, T=T, e=e, pmf_mass=pmf_mass)
    acc = accuracy_confidence_statement(
        R=R, d_min=mm.d_min, e=e, alpha=alpha, beta_target=beta_target
    )
    return {
        "d_eff": mm.d_eff,
        "d_min": mm.d_min,
        "I_eff": mm.I_eff,
        "I_min": mm.I_min,
        "var_eff": mm.var_eff,
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
    report = mixing_and_accuracy_report(
        R=150, C=4, D=4, T=3, e=2,
        alpha=0.01, beta_target=0.01
    )
    for k in ["d_eff", "d_min", "I_min", "beta_bound", "meets_target"]:
        print(f"{k}: {report[k]}")
    print(report["statement"])

    print("\n=========================================================")
    print("Case 2: should meet target accuracy")
    report = mixing_and_accuracy_report(
        R=150, C=7, D=3, T=3, e=2,
        alpha=0.01, beta_target=0.01
    )
    for k in ["d_eff", "d_min", "I_min", "beta_bound", "meets_target"]:
        print(f"{k}: {report[k]}")
    print(report["statement"])
