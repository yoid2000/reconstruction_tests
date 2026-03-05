import numpy as np

from build_row_masks import build_row_masks_qi


def _pair_correlations(df, pairs):
    return {f"{a}-{b}": float(df[a].corr(df[b])) for a, b in pairs}


def test_correlation_strength_increases_pair_matches():
    strengths = [0.0, 0.1, 0.2, 0.3]
    avg_corrs = []
    pairs = [(f"qi{i}", f"qi{i+1}") for i in range(0, 10, 2)]
    seeds = [12345, 54321, 999]

    for strength in strengths:
        pair_corrs = {f"{a}-{b}": [] for a, b in pairs}
        seed_avgs = []
        for seed in seeds:
            np.random.seed(seed)
            df = build_row_masks_qi(
                nrows=150,
                nunique=2,
                nqi=10,
                vals_per_qi=4,
                corr_strength=strength,
            )
            corrs = _pair_correlations(df, pairs)
            seed_avgs.append(float(np.mean(list(corrs.values()))))
            for name, val in corrs.items():
                pair_corrs[name].append(val)

        avg_corr = float(np.mean(seed_avgs))
        avg_corrs.append(avg_corr)
        print(f"corr_strength={strength:.1f} avg_pair_corr={avg_corr:.4f} seeds={seeds}")
        for name, vals in pair_corrs.items():
            print(f"  {name}: {float(np.mean(vals)):.4f}")

    min_step = 0.005
    for i in range(len(avg_corrs) - 1):
        delta = avg_corrs[i + 1] - avg_corrs[i]
        print(
            f"delta {strengths[i]:.1f}->{strengths[i + 1]:.1f}: {delta:.4f} "
            f"(min {min_step:.2f})"
        )
        assert delta >= min_step, (
            f"Expected average correlation to increase by at least {min_step} "
            f"from {strengths[i]} to {strengths[i + 1]}, got {avg_corrs}"
        )


if __name__ == "__main__":
    test_correlation_strength_increases_pair_matches()
    print("OK")
