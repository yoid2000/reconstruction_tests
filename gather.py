import json
import pandas as pd
import os
from glob import glob


# Expand a single result JSON into a list of rows, one per attack_results entry.
# Each row contains top-level metadata plus the attack entry fields.
# Adds:
#   - final_attack: True if this is the last entry in attack_results
#   - attack_index: index of the entry in attack_results (0-based)
# Flattens 'mixing' and 'solver_metrics' nested dicts by prefixing keys.
def _rows_from_result(result_json, source_file=None):
	# base metadata: everything except attack_results
	base = {k: v for k, v in result_json.items() if k != "attack_results"}
	if source_file is not None:
		base["source_file"] = source_file

	attack_results = result_json.get("attack_results", [])
	rows = []
	if not attack_results:
		# keep one row using only base metadata
		row = base.copy()
		row["final_attack"] = True
		row["attack_index"] = 0
		rows.append(row)
		return rows

	for idx, entry in enumerate(attack_results):
		row = base.copy()
		# copy simple fields from the attack entry, flatten certain nested dicts
		for k, v in entry.items():
			if k in ("mixing", "solver_metrics") and isinstance(v, dict):
				for subk, subv in v.items():
					row[f"{k}_{subk}"] = subv
			else:
				row[k] = v
		row["final_attack"] = (idx == len(attack_results) - 1)
		row["attack_index"] = idx
		rows.append(row)
	return rows

def gather_results(results_root=None, out_csv=None, verbose=False):
	# default: search under the package's row_mask_attacks/results directory
	if results_root is None:
		results_root = os.path.join(os.path.dirname(__file__), "row_mask_attacks", "results")

	pattern = os.path.join(results_root, "**", "*.json")
	json_files = glob(pattern, recursive=True)

	all_rows = []
	for fp in sorted(json_files):
		try:
			with open(fp, "r", encoding="utf-8") as fh:
				data = json.load(fh)
		except Exception as e:
			if verbose:
				print(f"skipping {fp}: {e}")
			continue

		# expand into one row per attack_results entry
		for r in _rows_from_result(data, source_file=os.path.basename(fp)):
			all_rows.append(r)

	if not all_rows:
		if verbose:
			print("no rows gathered")
		return pd.DataFrame()

	df = pd.DataFrame(all_rows)

	if out_csv:
		df.to_csv(out_csv, index=False)
	elif verbose:
		print(f"gathered {len(df)} rows from {len(json_files)} files")

	return df

if __name__ == "__main__":
	# when run directly, write to gathered_results.csv next to this file
	out_path = os.path.join(os.path.dirname(__file__), "gathered_results.csv")
	df = gather_results(out_csv=out_path, verbose=True)
	print("output written to:", out_path)