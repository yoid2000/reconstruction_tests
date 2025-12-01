import json
import pandas as pd
import os
from glob import glob

# ...existing code...

def _rows_from_result(result_json):
	"""
	Expand a single result JSON into a list of rows, one per attack_results entry.
	Each row contains top-level metadata plus the attack entry fields.
	Adds:
	  - final_attack: True if this is the last entry in attack_results
	  - attack_index: index of the entry in attack_results (0-based)
	Flattens 'mixing' and 'solver_metrics' nested dicts by prefixing keys.
	"""
	base = {k: v for k, v in result_json.items() if k != "attack_results"}
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
		# copy simple fields from the attack entry
		for k, v in entry.items():
			if k in ("mixing", "solver_metrics") and isinstance(v, dict):
				# flatten nested dicts with a prefix
				for subk, subv in v.items():
					row[f"{k}_{subk}"] = subv
			else:
				# keep other fields (including nested non-dict if any)
				row[k] = v
		row["final_attack"] = (idx == len(attack_results) - 1)
		row["attack_index"] = idx
		rows.append(row)
	return rows

# ...existing code...

# Replace the existing single-row append where results are processed with usage of the helper:
# Example location in the file where JSON files are iterated:
# for fp in json_files:
#     data = json.load(open(fp))
#     rows.append(process_single_result(data))
# ...existing code...

# New/changed region:
# ...existing code...
for fp in json_files:
	# ...existing code...
	data = json.load(open(fp))
	# replace single-row logic with expanding into multiple rows
	for r in _rows_from_result(data):
		all_rows.append(r)
# ...existing code...