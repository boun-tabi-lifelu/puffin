#!/usr/bin/env python3
import csv
import json
import os
import re
import sys
import glob
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------- config: what to extract (edit here) --------
KEY_PATHS: List[Tuple[str, str]] = [
    # (output_column_name, dotted_key_path_in_json)
    ("n_proteins", "n_proteins"),
    ("n_segments", "n_segments"),
    ("has_structure_metrics", "has_structure_metrics"),

    # protein-level (means)
    ("protein_n_res_mean", "protein_level.n_res.mean"),
    ("protein_unique_labels_mean", "protein_level.unique_labels.mean"),
    ("protein_boundaries_mean", "protein_level.boundaries.mean"),
    ("protein_mean_run_length_mean", "protein_level.mean_run_length.mean"),
    ("protein_usage_gini_mean", "protein_level.usage_gini.mean"),
    ("protein_usage_entropy_norm_mean", "protein_level.usage_entropy_norm.mean"),

    # segment-level (means)
    ("segment_n_res_mean", "segment_level.n_res.mean"),
    ("segment_seq_span_mean", "segment_level.seq_span.mean"),
    ("segment_seq_compactness_mean", "segment_level.seq_compactness.mean"),
    ("segment_run_count_mean", "segment_level.run_count.mean"),
    ("segment_max_run_length_mean", "segment_level.max_run_length.mean"),
    ("segment_fragmentation_ratio_mean", "segment_level.fragmentation_ratio.mean"),

    # structure segment-level (means) - may be missing
    ("segment_mean_intra_ca_dist_mean", "segment_level.mean_intra_ca_dist.mean"),
    ("segment_contact_density_mean", "segment_level.contact_density.mean"),
    ("segment_cut_ratio_mean", "segment_level.cut_ratio.mean"),
    ("segment_rg_mean", "segment_level.rg.mean"),
    ("segment_packing_density_mean", "segment_level.packing_density.mean"),
    ("segment_log_packing_density_mean", "segment_level.log_packing_density.mean"),

    # random baseline deltas (means) - may be missing
    ("delta_seq_span_mean", "random_baseline.model_minus_random.seq_span.delta_mean"),
    ("delta_seq_compactness_mean", "random_baseline.model_minus_random.seq_compactness.delta_mean"),
    ("delta_run_count_mean", "random_baseline.model_minus_random.run_count.delta_mean"),
    ("delta_max_run_length_mean", "random_baseline.model_minus_random.max_run_length.delta_mean"),
    ("delta_fragmentation_ratio_mean", "random_baseline.model_minus_random.fragmentation_ratio.delta_mean"),
    ("delta_mean_intra_ca_dist_mean", "random_baseline.model_minus_random.mean_intra_ca_dist.delta_mean"),
    ("delta_contact_density_mean", "random_baseline.model_minus_random.contact_density.delta_mean"),
    ("delta_cut_ratio_mean", "random_baseline.model_minus_random.cut_ratio.delta_mean"),
    ("delta_rg_mean", "random_baseline.model_minus_random.rg.delta_mean"),
    ("delta_packing_density_mean", "random_baseline.model_minus_random.packing_density.delta_mean"),
    ("delta_log_packing_density_mean", "random_baseline.model_minus_random.log_packing_density.delta_mean"),
]

DEFAULT_GLOB = "ismb26/results/**/**_summary.json"


# -------- helpers --------
def get_by_dotted_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def parse_metadata(p: Path) -> Dict[str, Any]:
    """
    Extract method, K, split from paths like:
      ismb26/results/segment_reports/mincut_K16/test/test_summary.json
    """
    s = str(p)

    # split: parent folder of the json (train/valid/test)
    split = p.parent.name

    # try: ".../segment_reports/<method>_K<k>/<split>/..."
    method = None
    K = None

    m = re.search(r"segment_reports/([^/]+?)(?:/|$)", s)
    # m.group(1) might be "mincut_K16" (folder at segment_reports level)
    if m:
        top = m.group(1)
        m2 = re.match(r"(?P<method>.+)_K(?P<K>\d+)$", top)
        if m2:
            method = m2.group("method")
            K = int(m2.group("K"))
        else:
            method = top

    return {
        "file": s,
        "method": method,
        "K": K,
        "split": split,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect key metrics from *_summary.json into a CSV.")
    ap.add_argument("--glob", default=DEFAULT_GLOB, help=f"Glob for json files (default: {DEFAULT_GLOB})")
    ap.add_argument("--out", default="segment_summary_key_metrics.csv", help="Output CSV path")
    args = ap.parse_args()


    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        print(f"ERROR: no files matched glob: {args.glob}", file=sys.stderr)
        return 2
    if not files:
        print(f"ERROR: no files matched glob: {args.glob}", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for f in files:
        fp = Path(f) 
        try:
            with fp.open("r") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"WARNING: failed to read {f}: {e}", file=sys.stderr)
            continue

        row: Dict[str, Any] = {}
        row.update(parse_metadata(fp))

        for col, jpath in KEY_PATHS:
            row[col] = get_by_dotted_path(data, jpath)

        rows.append(row)

    # stable column order
    cols = ["file", "method", "K", "split"] + [c for c, _ in KEY_PATHS]

    with open(args.out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    main()
# Example usage:
# Collect segment metrics from JSON files
# python3 collect_segment_json_metrics.py \
#   --glob 'ismb26/results/**/**_summary.json' \
#   --out segment_summary_key_metrics.csv

