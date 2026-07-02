import argparse
from pathlib import Path

from build_row_masks import build_row_masks_qi


def _format_float_for_filename(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def get_default_output_path(args: argparse.Namespace) -> Path:
    return Path(
        f"nr_{args.nrows}_"
        f"nu_{args.nunique}_"
        f"nq_{args.nqi}_"
        f"vq_{args.vals_per_qi}_"
        f"cs_{_format_float_for_filename(args.corr_strength)}.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a row-mask DataFrame and write it as parquet."
    )
    parser.add_argument("--nrows", type=int, default=150)
    parser.add_argument("--nunique", type=int, default=2)
    parser.add_argument("--nqi", type=int, default=11)
    parser.add_argument("--vals_per_qi", type=int, default=2)
    parser.add_argument("--corr_strength", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_row_masks_qi(
        nrows=args.nrows,
        nunique=args.nunique,
        nqi=args.nqi,
        vals_per_qi=args.vals_per_qi,
        corr_strength=args.corr_strength,
    )
    output = args.output or get_default_output_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)


if __name__ == "__main__":
    main()
