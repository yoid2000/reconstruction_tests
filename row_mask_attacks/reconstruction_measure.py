"""
Heterogeneous-cardinality (per-column D_j) mixing analysis with reusable results.

You run the expensive analysis once:
    analysis = run_heteroD_analysis(...)  # returns a dict

Then you can cheaply:
  - evaluate different (alpha, beta) via:
        HeteroDAnalysis.accuracy_statement(analysis, alpha, beta_target)
  - compute the minimum required I_true_min to meet (alpha, beta_target) via:
        HeteroDAnalysis.required_I_true_min(analysis, alpha, beta_target)

Model recap (as used in the screening bounds):
  - Targets are K-class with priors pi (default uniform).
  - For each query, each class-count is released iff the TRUE count >= T.
  - Released counts are perturbed with integer noise Unif{-e,...,e}.
  - For reconstruction screening, we use own-class released coverage d_true_min and normalize:
        I_true_min = d_true_min / (2e+1)
  - Per-row error proxy:
        p_row <= (K-1) * exp(-I_true_min)
  - Accuracy tail bound (Chernoff/KL under Binomial approximation):
        P(accuracy < 1-alpha) = P(E >= alpha R) <= exp( -R * KL(alpha || p_row) )
    when p_row < alpha; otherwise the bound is not informative (returns 1).

This code uses SciPy (scipy.stats.binom, scipy.stats.norm).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Literal, Tuple

import numpy as np
from scipy.stats import binom, norm


# -----------------------------
# Utilities
# -----------------------------

def _normalize_probs(class_probs: Optional[List[float]], K: int) -> np.ndarray:
    if class_probs is None:
        return np.full(K, 1.0 / K, dtype=float)
    p = np.asarray(class_probs, dtype=float)
    if p.ndim != 1 or len(p) != K:
        raise ValueError("class_probs must be a length-K 1D list/array.")
    if np.any(p < 0):
        raise ValueError("class_probs must be nonnegative.")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("class_probs must sum to > 0.")
    return p / s


def _kl_bern(a: float, p: float) -> float:
    """KL(a || p) for Bernoulli(a) vs Bernoulli(p)."""
    eps = 1e-300
    a = min(1.0 - eps, max(eps, a))
    p = min(1.0 - eps, max(eps, p))
    return float(a * np.log(a / p) + (1.0 - a) * np.log((1.0 - a) / (1.0 - p)))


def _beta_bound_from_p_row(R: int, alpha: float, p_row: float) -> float:
    """
    Chernoff/KL upper tail bound for Binomial(R, p_row):
      P(E >= alpha R) <= exp( -R * KL(alpha || p_row) )  when p_row < alpha.
    Otherwise return 1.0 (not informative).
    """
    if p_row >= alpha:
        return 1.0
    return float(np.exp(-R * _kl_bern(alpha, p_row)))


# -----------------------------
# Conditional release moments (given group size s)
# -----------------------------

def _mean_var_released_counts_given_s(s: int, T: int, pi: np.ndarray) -> Tuple[float, float]:
    """
    Z(s) = number of released class-counts among K classes for a group of size s.
    Each class r is released iff U_r >= T.

    Under Multinomial(s, pi), marginally U_r ~ Binomial(s, pi_r).

    Returns:
      mean = E[Z | s] = sum_r P(Binomial(s, pi_r) >= T)
      var  = Var(Z | s) approx sum_r p_r(1-p_r) (ignores covariance across classes)
    """
    if s <= 0:
        return 0.0, 0.0
    if T <= 0:
        K = len(pi)
        return float(K), 0.0
    p = binom.sf(T - 1, s, pi)
    mean = float(np.sum(p))
    var = float(np.sum(p * (1.0 - p)))
    return mean, var


def _mean_var_ownclass_release_given_s(s: int, T: int, pi: np.ndarray) -> Tuple[float, float]:
    """
    I_own(s) indicates whether the count for the row's own class is released.

    If row has class r, then in a group of size s containing it:
      U_r = 1 + Binomial(s-1, pi_r)
    Release own-class iff U_r >= T <=> Binomial(s-1, pi_r) >= T-1.

    Returns:
      mean = E[I_own | s] = sum_r pi_r * P(Binomial(s-1, pi_r) >= T-1)
      var  = mean*(1-mean)
    """
    if s <= 0:
        return 0.0, 0.0
    if T <= 0:
        return 1.0, 0.0

    n = s - 1
    if n < 0:
        return 0.0, 0.0

    if T <= 1:
        pr = np.ones_like(pi)
    else:
        pr = binom.sf(T - 2, n, pi)  # P(Binomial(n,pi_r) >= T-1)

    mean = float(np.sum(pi * pr))
    var = float(mean * (1.0 - mean))
    return mean, var


# -----------------------------
# Integrate over group size distribution:
#   G = 1 + Binomial(R-1, p)
# -----------------------------

def _query_moments_for_match_prob(
    R: int,
    p: float,
    T: int,
    pi: np.ndarray,
    pmf_mass: float,
    cache_by_s: Dict[int, tuple],
) -> Tuple[float, float, float, float]:
    """
    For match prob p, group size G = 1 + Binomial(R-1, p).
    Returns per-query moments:
      mean_all, var_all, mean_own, var_own
    using a truncated pmf window capturing pmf_mass.
    """
    n = R - 1

    # Degenerate p
    if p <= 0.0:
        s = 1
        if s not in cache_by_s:
            cache_by_s[s] = (
                _mean_var_released_counts_given_s(s, T, pi),
                _mean_var_ownclass_release_given_s(s, T, pi),
            )
        (m_all_s, v_all_s), (m_own_s, v_own_s) = cache_by_s[s]
        return m_all_s, v_all_s, m_own_s, v_own_s

    if p >= 1.0:
        s = R
        if s not in cache_by_s:
            cache_by_s[s] = (
                _mean_var_released_counts_given_s(s, T, pi),
                _mean_var_ownclass_release_given_s(s, T, pi),
            )
        (m_all_s, v_all_s), (m_own_s, v_own_s) = cache_by_s[s]
        return m_all_s, v_all_s, m_own_s, v_own_s

    lo, hi = binom.interval(pmf_mass, n=n, p=p)
    lo = int(max(0, lo))
    hi = int(min(n, hi))

    xs = np.arange(lo, hi + 1, dtype=int)
    pmf = binom.pmf(xs, n=n, p=p)
    s_pmf = float(pmf.sum())
    if not np.isfinite(s_pmf) or s_pmf <= 0.0:
        xs = np.arange(0, min(n, 5) + 1, dtype=int)
        pmf = binom.pmf(xs, n=n, p=p)
        s_pmf = float(pmf.sum())
        if s_pmf <= 0.0:
            mu = n * p
            x0 = int(round(mu))
            xs = np.array([x0], dtype=int)
            pmf = np.array([1.0], dtype=float)
            s_pmf = 1.0
    pmf = pmf / s_pmf

    mean_all = 0.0
    second_all = 0.0
    mean_own = 0.0
    second_own = 0.0

    for x, px in zip(xs, pmf):
        s = int(x) + 1
        if s not in cache_by_s:
            cache_by_s[s] = (
                _mean_var_released_counts_given_s(s, T, pi),
                _mean_var_ownclass_release_given_s(s, T, pi),
            )
        (m_all_s, v_all_s), (m_own_s, v_own_s) = cache_by_s[s]

        mean_all += float(px) * m_all_s
        second_all += float(px) * (v_all_s + m_all_s * m_all_s)

        mean_own += float(px) * m_own_s
        second_own += float(px) * (v_own_s + m_own_s * m_own_s)

    var_all = max(0.0, second_all - mean_all * mean_all)
    var_own = max(0.0, second_own - mean_own * mean_own)
    return float(mean_all), float(var_all), float(mean_own), float(var_own)


# -----------------------------
# Analysis dict + post-hoc methods
# -----------------------------

AnalysisDict = Dict[str, object]

@dataclass(frozen=True)
class HeteroDAnalysis:
    # Inputs / configuration
    R: int
    C: int
    D_cols: List[int]
    K: int
    T: int
    e: int
    class_probs: List[float]
    method: str
    pmf_mass: float

    # Mixing outputs
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

    def expected_accuracy_and_ci(self, ci_level: float = 0.95, use: str = "eff"):
        """
        Returns an expected accuracy proxy and a (ci_level) interval under a Binomial(R,p) model.

        use:
        - "eff": uses I_true_eff (point estimate)
        - "min": uses I_true_min (more conservative)
        """
        if not (0.0 < ci_level < 1.0):
            raise ValueError("ci_level must be in (0,1).")
        I = self.I_true_eff if use == "eff" else self.I_true_min

        p = float(min(1.0, (self.K - 1) * np.exp(-I)))
        acc_hat = 1.0 - p

        gamma = 1.0 - ci_level
        # Quantiles for number of errors E ~ Binomial(R,p)
        E_lo = int(binom.ppf(gamma / 2.0, self.R, p))
        E_hi = int(binom.ppf(1.0 - gamma / 2.0, self.R, p))

        # Convert to accuracy interval (note the flip)
        acc_lo = 1.0 - (E_hi / self.R)
        acc_hi = 1.0 - (E_lo / self.R)

        return {
            "use": use,
            "p_used": p,
            "expected_accuracy": acc_hat,
            "ci_level": ci_level,
            "accuracy_ci": (acc_lo, acc_hi),
            "expected_errors": self.R * p,
            "errors_ci": (E_lo, E_hi),
        }

    @staticmethod
    def p_row(analysis: AnalysisDict) -> float:
        """
        Per-row error proxy (union bound over K-1 incorrect labels):
          p_row <= (K-1) * exp(-I_true_min)
        """
        K = int(analysis["K"])
        I_true_min = float(analysis["I_true_min"])
        return float(min(1.0, (K - 1) * np.exp(-I_true_min)))

    @staticmethod
    def mixing_outputs(analysis: AnalysisDict) -> Dict[str, float]:
        """
        Return mixing-related outputs as a dict for downstream reporting.
        """
        return {
            "d_eff": float(analysis["d_eff"]),
            "d_min": float(analysis["d_min"]),
            "d_true_eff": float(analysis["d_true_eff"]),
            "d_true_min": float(analysis["d_true_min"]),
            "I_eff": float(analysis["I_eff"]),
            "I_min": float(analysis["I_min"]),
            "I_true_eff": float(analysis["I_true_eff"]),
            "I_true_min": float(analysis["I_true_min"]),
            "var_eff": float(analysis["var_eff"]),
            "var_true": float(analysis["var_true"]),
        }

    @staticmethod
    def accuracy_statement(
        analysis: AnalysisDict,
        alpha: float,
        beta_target: float = 0.01,
    ) -> Dict[str, float | bool | str]:
        """
        Cheap post-hoc evaluation for any (alpha, beta_target) without recomputing mixing.

        Uses:
          p_row <= (K-1) * exp(-I_true_min)
          beta_bound = exp(-R * KL(alpha || p_row))  when p_row < alpha
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1).")
        if not (0.0 < beta_target < 1.0):
            raise ValueError("beta_target must be in (0,1).")

        R = int(analysis["R"])
        p_row = HeteroDAnalysis.p_row(analysis)
        beta_bound = _beta_bound_from_p_row(R, alpha, p_row)
        meets = beta_bound <= beta_target

        statement = (
            f"Under the i.i.d. model + independence approximations: "
            f"P(accuracy < {1.0 - alpha:.2%}) ≤ {beta_bound:.3e}. "
            f"Equivalently, with probability ≥ {1.0 - beta_bound:.2%}, "
            f"accuracy is at least {1.0 - alpha:.2%}."
        )
        if meets:
            statement += f" This meets (alpha={alpha}, beta={beta_target})."
        else:
            statement += f" This does NOT meet (alpha={alpha}, beta={beta_target})."

        return {
            "alpha": float(alpha),
            "beta_target": float(beta_target),
            "p_row": float(p_row),
            "beta_bound": float(beta_bound),
            "meets_target": bool(meets),
            "statement": statement,
        }

    @staticmethod
    def required_I_true_min(analysis: AnalysisDict, alpha: float, beta_target: float) -> float:
        """
        Minimum I_true_min needed to satisfy beta_bound <= beta_target.

        We solve for the smallest p in (0, alpha) such that:
            exp(-R * KL(alpha || p)) <= beta_target
        equivalently:
            KL(alpha || p) >= (1/R) * ln(1/beta_target)

        Then convert p -> I using p = (K-1) * exp(-I):
            I = ln((K-1)/p)

        This routine is numerically stable (log-space bisection).
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1).")
        if not (0.0 < beta_target < 1.0):
            raise ValueError("beta_target must be in (0,1).")

        R = int(analysis["R"])
        target_kl = (1.0 / R) * float(np.log(1.0 / beta_target))
        Kminus1 = float(int(analysis["K"]) - 1)
        if Kminus1 <= 0.0:
            raise ValueError("K must be > 1.")

        # We search p in (0, alpha). Use log-space to avoid underflow.
        # Upper end: just below alpha
        p_hi = min(alpha * (1.0 - 1e-12), 1.0 - 1e-12)
        if p_hi <= 0.0:
            # alpha is too tiny for float representation
            return float("inf")

        # If already sufficient at p just below alpha, return that I
        if _kl_bern(alpha, p_hi) >= target_kl:
            return float(np.log(Kminus1 / p_hi))

        # Lower end: extremely small p, but not zero.
        # exp(-745) ~ 5e-324 is around smallest positive float; go a bit above that.
        logp_lo = -745.0
        logp_hi = float(np.log(p_hi))

        # Ensure KL at p_lo is above target_kl (as p -> 0, KL -> +inf)
        # If numeric issues arise, push logp_lo down to -745 (already minimal safe).
        p_lo = float(np.exp(logp_lo))
        # In extremely rare numeric cases, if KL doesn't compute, we'd still proceed.

        # Bisection in logp
        for _ in range(200):
            logp_mid = 0.5 * (logp_lo + logp_hi)
            p_mid = float(np.exp(logp_mid))
            kl_mid = _kl_bern(alpha, p_mid)

            if kl_mid >= target_kl:
                # p can be larger (less strict) and still meet target, so move hi down to mid (toward larger p)
                logp_hi = logp_mid
            else:
                logp_lo = logp_mid

            if abs(logp_hi - logp_lo) < 1e-12:
                break

        p_req = float(np.exp(logp_hi))
        # Guard: never allow p_req to become 0 by underflow
        p_req = max(p_req, float(np.exp(-745.0)))

        return float(np.log(Kminus1 / p_req))

# -----------------------------
# Expensive analysis runner
# -----------------------------

def run_heteroD_analysis(
    R: int,
    D_cols: List[int],
    K: int,
    T: int,
    e: int,
    class_probs: Optional[List[float]] = None,
    pmf_mass: float = 1.0 - 1e-12,
    method: Literal["auto", "exact", "mc"] = "auto",
    exact_C_max: int = 20,
    subset_samples: int = 20000,
    seed: Optional[int] = 0,
) -> AnalysisDict:
    """
    Runs the expensive mixing analysis once and returns a reusable analysis dict.
    """
    start_time = time.time()
    if R < 2:
        D_cols = list(D_cols)
        C = len(D_cols)
        pi = _normalize_probs(class_probs, K).tolist()
        return {
            "R": R,
            "C": C,
            "D_cols": D_cols,
            "K": K,
            "T": T,
            "e": e,
            "class_probs": pi,
            "method": "degenerate",
            "pmf_mass": pmf_mass,
            "d_eff": 0.0,
            "d_min": 0.0,
            "d_true_eff": 0.0,
            "d_true_min": 0.0,
            "I_eff": 0.0,
            "I_min": 0.0,
            "I_true_eff": 0.0,
            "I_true_min": 0.0,
            "var_eff": 0.0,
            "var_true": 0.0,
            "measure_elapsed_time": float(time.time() - start_time),
        }

    D_arr = np.asarray(D_cols, dtype=float)
    if D_arr.ndim != 1 or len(D_arr) == 0:
        raise ValueError("D_cols must be a non-empty 1D list/array.")
    if np.any(D_arr <= 0):
        raise ValueError("All D_j must be > 0.")
    C = int(len(D_arr))

    if method == "auto":
        method_use = "exact" if C <= exact_C_max else "mc"
    else:
        method_use = method
    if method_use == "exact" and C > exact_C_max:
        raise ValueError(f"C={C} too large for exact; use method='mc' or increase exact_C_max.")

    pi = _normalize_probs(class_probs, K)
    logD = np.log(D_arr)

    cache_by_s: Dict[int, tuple] = {}

    def subset_logp(mask: np.ndarray) -> float:
        # match probability p_S = exp( - sum_{j in S} log D_j )
        if not mask.any():
            return 0.0
        return float(-np.sum(logD[mask]))

    total_subsets = 2 ** C

    if method_use == "exact":
        mean_all_sum = 0.0
        var_all_sum = 0.0
        mean_own_sum = 0.0
        var_own_sum = 0.0

        for m in range(1 << C):
            bits = [(m >> j) & 1 for j in range(C)]
            mask = np.array(bits, dtype=bool)
            p = float(np.exp(subset_logp(mask)))

            m_all, v_all, m_own, v_own = _query_moments_for_match_prob(
                R=R, p=p, T=T, pi=pi, pmf_mass=pmf_mass, cache_by_s=cache_by_s
            )
            mean_all_sum += m_all
            var_all_sum += v_all
            mean_own_sum += m_own
            var_own_sum += v_own

        d_eff = float(mean_all_sum)
        var_eff_total = float(var_all_sum)
        d_true = float(mean_own_sum)
        var_true_total = float(var_own_sum)

    else:
        rng = np.random.default_rng(seed)

        mean_all = 0.0
        var_all = 0.0
        mean_own = 0.0
        var_own = 0.0

        for _ in range(subset_samples):
            mask = rng.random(C) < 0.5  # uniform over all subsets
            p = float(np.exp(subset_logp(mask)))

            m_all, v_all, m_own, v_own = _query_moments_for_match_prob(
                R=R, p=p, T=T, pi=pi, pmf_mass=pmf_mass, cache_by_s=cache_by_s
            )
            mean_all += m_all
            var_all += v_all
            mean_own += m_own
            var_own += v_own

        mean_all /= subset_samples
        var_all /= subset_samples
        mean_own /= subset_samples
        var_own /= subset_samples

        d_eff = float(total_subsets * mean_all)
        var_eff_total = float(total_subsets * var_all)
        d_true = float(total_subsets * mean_own)
        var_true_total = float(total_subsets * var_own)

    def lower_quantile(mean: float, var: float) -> float:
        sigma = float(np.sqrt(max(0.0, var)))
        if sigma == 0.0:
            return float(mean)
        z = float(norm.ppf(1.0 / R))
        return float(max(0.0, mean + sigma * z))

    d_min = lower_quantile(d_eff, var_eff_total)
    d_true_min = lower_quantile(d_true, var_true_total)

    denom = 2 * e + 1
    I_eff = float(d_eff / denom)
    I_min = float(d_min / denom)
    I_true_eff = float(d_true / denom)
    I_true_min = float(d_true_min / denom)

    return {
        "R": R,
        "C": C,
        "D_cols": [int(x) for x in D_cols],
        "K": K,
        "T": T,
        "e": e,
        "class_probs": pi.tolist(),
        "method": method_use,
        "pmf_mass": pmf_mass,
        "d_eff": float(d_eff),
        "d_min": float(d_min),
        "d_true_eff": float(d_true),
        "d_true_min": float(d_true_min),
        "I_eff": I_eff,
        "I_min": I_min,
        "I_true_eff": I_true_eff,
        "I_true_min": I_true_min,
        "var_eff": float(var_eff_total),
        "var_true": float(var_true_total),
        "measure_elapsed_time": float(time.time() - start_time),
    }


# -----------------------------
# Examples
# -----------------------------
if __name__ == "__main__":
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    def example_usage(analysis: AnalysisDict):
        pp.pprint(analysis)
        print("Achieved I_true_min:", analysis["I_true_min"])
        print("Achieved p_row proxy:", HeteroDAnalysis.p_row(analysis))

        # 2) Evaluate different (alpha, beta) pairs WITHOUT recomputing analysis
        for alpha, beta in [(0.01, 0.01), (0.10, 0.01), (0.05, 1e-6)]:
            out = HeteroDAnalysis.accuracy_statement(analysis, alpha=alpha, beta_target=beta)
            print(out["statement"])

        # 3) Compute required I_true_min for various (alpha, beta)
        for alpha, beta in [(0.01, 0.01), (0.10, 0.01), (0.05, 1e-6)]:
            I_req = HeteroDAnalysis.required_I_true_min(analysis, alpha=alpha, beta_target=beta)
            print(f"Required I_true_min for (alpha={alpha}, beta={beta}): {I_req:.4f}")
            print(f"Achieved/Required ratio: {analysis['I_true_min'] / I_req:.3f}")

        # 4) Convert required I_true_min into required d_true_min:
        #    d_true_min_required = (2e+1) * I_true_min_required
        alpha, beta = 0.01, 0.01
        I_req = HeteroDAnalysis.required_I_true_min(analysis, alpha=alpha, beta_target=beta)
        d_req = (2 * analysis["e"] + 1) * I_req
        print(f"For (alpha={alpha}, beta={beta}), required d_true_min ≈ {d_req:.2f}")


    print("\n=========================================================")
    print("Case 1: should not meet target accuracy")
    # Each of 4 columns has D=4 distinct values
    D_cols = [4, 4, 4, 4]

    analysis = run_heteroD_analysis(
        R=150,
        D_cols=D_cols,
        K=2,
        T=3,
        e=2,
        class_probs=None,      # uniform over K classes
        method="auto",         # exact if C<=20 else Monte Carlo
        subset_samples=30000,  # used only if MC mode
        seed=0
    )
    example_usage(analysis)

    print("\n=========================================================")
    print("Case 2: should meet target accuracy")
    # Each of 4 columns has D=4 distinct values
    D_cols = [3, 3, 3, 3, 3, 3, 3]

    analysis = run_heteroD_analysis(
        R=150,
        D_cols=D_cols,
        K=2,
        T=3,
        e=2,
        class_probs=None,      # uniform over K classes
        method="auto",         # exact if C<=20 else Monte Carlo
        subset_samples=30000,  # used only if MC mode
        seed=0
    )
    example_usage(analysis)
