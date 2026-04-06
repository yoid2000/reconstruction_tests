from pathlib import Path

SUBSTRING_REWRITES = {
    "_0.600_": "_0.800_",
    "_0.800_": "_0.900_",
    "_0.900_": "_0.950_",
}


def list_png_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.png") if path.is_file()}


def print_section(title: str, file_names: list[str]) -> None:
    print(f"\n{title} ({len(file_names)}):")
    if not file_names:
        print("  (none)")
        return
    for name in file_names:
        print(f"  {name}")


def rewrite_filename(name: str) -> str:
    """Apply all substring rewrites without cascading replacements."""
    tmp_map = {
        "_0.600_": "__TMP_0600__",
        "_0.800_": "__TMP_0800__",
        "_0.900_": "__TMP_0900__",
    }
    updated = name
    for src, tmp in tmp_map.items():
        updated = updated.replace(src, tmp)
    for src, dst in SUBSTRING_REWRITES.items():
        updated = updated.replace(tmp_map[src], dst)
    return updated


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "results" / "plots"
    plots_alc_dir = base_dir / "results" / "plots_alc"

    if not plots_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {plots_dir}")
    if not plots_alc_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {plots_alc_dir}")

    plots_pngs = list_png_names(plots_dir)
    plots_alc_pngs = list_png_names(plots_alc_dir)
    plots_alc_repr = {rewrite_filename(name) for name in plots_alc_pngs}

    only_in_plots = sorted(plots_pngs - plots_alc_repr)
    only_in_plots_alc_repr = sorted(plots_alc_repr - plots_pngs)

    print(f"Comparing .png files in:\n  {plots_dir}\n  {plots_alc_dir}")
    print("Note: plots_alc names are rewritten in-memory for comparison only; files are not renamed.")
    print_section("Only in results/plots", only_in_plots)
    print_section("Only in results/plots_alc (after representation rewrite)", only_in_plots_alc_repr)


if __name__ == "__main__":
    main()
