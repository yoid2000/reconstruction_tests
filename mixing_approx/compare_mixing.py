from __future__ import annotations

import math
import statistics

from approx_mixing import expected_mixing_M
from simulated_mixing import simulated_mixing


def min_distinct_D(R: int, C: int) -> int:
    if R <= 1:
        return 1
    D = 1
    while D ** C < R:
        D += 1
    return D


def main() -> None:
    # Small ranges to keep the full query enumeration manageable.
    R_values = [25, 50, 100, 200]
    C_values = [3, 5, 7]
    T_values = [2, 3, 4]

    min_sim_trials = 10
    max_sim_attempts = 200
    sim_rel_error = 0.05
    sim_conf_z = 1.96
    analytic_rel_error = 0.05

    results = []
    invalid_combos = []

    for R in R_values:
        for C in C_values:
            D_min = min_distinct_D(R, C)
            for D in (D_min, D_min + 1):
                for T in T_values:
                    if T > R:
                        continue
                    base_seed = (R * 10_000) + (C * 1_000) + (D * 100) + T
                    sim_values = []
                    sim_mean = 0.0
                    sim_std = 0.0
                    sim_err = float("inf")
                    sim_rel = float("inf")
                    invalid_runs = 0
                    for attempt in range(max_sim_attempts):
                        sim = simulated_mixing(R, C, D, T, seed=base_seed + attempt)
                        if sim is None:
                            invalid_runs += 1
                            continue
                        sim_values.append(sim)
                        if len(sim_values) < min_sim_trials:
                            continue
                        sim_mean = statistics.fmean(sim_values)
                        sim_std = statistics.pstdev(sim_values) if len(sim_values) > 1 else 0.0
                        sim_err = sim_conf_z * sim_std / math.sqrt(len(sim_values))
                        if sim_mean == 0.0:
                            sim_rel = 0.0 if sim_err == 0.0 else float("inf")
                        else:
                            sim_rel = sim_err / abs(sim_mean)
                        if sim_rel <= sim_rel_error:
                            break
                    if invalid_runs > 0:
                        invalid_combos.append((R, C, D, T, invalid_runs))
                        continue
                    if not sim_values:
                        continue
                    if len(sim_values) < min_sim_trials or not math.isfinite(sim_rel):
                        sim_mean = statistics.fmean(sim_values)
                        sim_std = statistics.pstdev(sim_values) if len(sim_values) > 1 else 0.0
                        sim_err = sim_conf_z * sim_std / math.sqrt(len(sim_values))
                        if sim_mean == 0.0:
                            sim_rel = 0.0 if sim_err == 0.0 else float("inf")
                        else:
                            sim_rel = sim_err / abs(sim_mean)

                    expected_binom = expected_mixing_M(R, C, D, T, method="binom")
                    expected_hyper = None
                    try:
                        expected_hyper = expected_mixing_M(R, C, D, T, method="hypergeom")
                    except (OverflowError, ValueError):
                        expected_hyper = None

                    if expected_hyper is not None:
                        expected = expected_hyper
                        analytic_err = abs(expected_hyper - expected_binom)
                        analytic_method = "hypergeom"
                    else:
                        expected = expected_binom
                        analytic_err = abs(expected) * analytic_rel_error
                        analytic_method = "binom"

                    diff = sim_mean - expected
                    rel = diff / expected if expected != 0 else float("inf")
                    total_err = sim_err + analytic_err
                    sim_met = (len(sim_values) >= min_sim_trials) and (sim_rel <= sim_rel_error)

                    if total_err == 0:
                        confidence = "match" if diff == 0 else "mismatch"
                    elif abs(diff) <= total_err:
                        confidence = "match"
                    elif abs(diff) <= 2 * total_err:
                        confidence = "borderline"
                    else:
                        confidence = "mismatch"

                    results.append(
                        (
                            R,
                            C,
                            D,
                            T,
                            expected,
                            sim_mean,
                            sim_std,
                            sim_err,
                            analytic_err,
                            total_err,
                            diff,
                            rel,
                            confidence,
                            analytic_method,
                            len(sim_values),
                            sim_met,
                            sim_rel,
                        )
                    )

    if not results:
        print("No comparable results (simulated_mixing returned None for all cases).")
        return

    header = (
        "R  C  D  T  expected     sim_mean    sim_std     sim_err    "
        "anal_err   total_err   diff        rel_diff  conf       method   n  sim_ok  sim_rel"
    )
    print(header)
    print("-" * len(header))
    for (
        R,
        C,
        D,
        T,
        expected,
        sim_mean,
        sim_std,
        sim_err,
        analytic_err,
        total_err,
        diff,
        rel,
        confidence,
        analytic_method,
        n_sim,
        sim_met,
        sim_rel,
    ) in results:
        print(
            f"{R:2d} {C:2d} {D:2d} {T:2d} "
            f"{expected:10.4f} {sim_mean:10.4f} {sim_std:10.4f} {sim_err:10.4f} "
            f"{analytic_err:10.4f} {total_err:10.4f} {diff:10.4f} {rel:9.4f} "
            f"{confidence:10s} {analytic_method:7s} {n_sim:2d} "
            f"{str(sim_met):6s} {sim_rel:7.4f}"
        )

    total = len(results)
    matches = sum(1 for r in results if r[12] == "match")
    borderline = sum(1 for r in results if r[12] == "borderline")
    mismatches = sum(1 for r in results if r[12] == "mismatch")
    sim_met_count = sum(1 for r in results if r[15] is True)
    rel_diffs = [abs(r[11]) for r in results if math.isfinite(r[11])]
    avg_rel_diff = statistics.fmean(rel_diffs) if rel_diffs else float("nan")
    max_rel_diff = max(rel_diffs) if rel_diffs else float("nan")

    print("\nSummary:")
    print(f"Total parameter sets compared: {total}")
    print(f"Sim error target met: {sim_met_count} / {total}")
    print(f"Confidence counts -> match: {matches}, borderline: {borderline}, mismatch: {mismatches}")
    print(f"Avg |rel_diff|: {avg_rel_diff:.4f}, max |rel_diff|: {max_rel_diff:.4f}")
    if invalid_combos:
        print(f"Skipped parameter sets due to invalid simulations: {len(invalid_combos)}")
        for R, C, D, T, invalid_runs in invalid_combos[:10]:
            print(f"  skipped R={R} C={C} D={D} T={T} (invalid runs: {invalid_runs})")
        if len(invalid_combos) > 10:
            print("  ...")


if __name__ == "__main__":
    main()
