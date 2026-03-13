# src/cluster_compare.py
# -----------------------------------------------------------------------------
# Purpose
#   Direct, paired comparisons between TWO clusterings (e.g., mincut vs mincut+fp)
#
# Additions
#   - boundary-matched evaluation over MULTIPLE seeds + aggregation
#   - paired Wilcoxon + sign test on delta_f1 (soft deps)
#   - per-protein delta csv for easy plotting

# Usage example:


# python src/cluster_compare.py \
#   --runA_dir results/mincut_function \
#   --runA_prefix mincut_function \
#   --runB_dir results/mincut \
#   --runB_prefix mincut \
#   --annotation_dir data/GeneOntology \
#   --output_dir eval/compare_mincut_function_vs_mincut \
#   --unit_eval \
#   --boundary_window 3 \
#   --boundary_seeds 0,1,2,3,4,5,6,7,8,9 \
#   --go_probe \
#   --go_aspect MF



import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger as log

from src.data.interpro import InterProManager
from src.cluster_eval import (  
    ClusterOutputs,
    GOAnnotations,
    build_protein_df_from_residue_assignments,
    require_columns,
    ensure_dir,
    clusters_to_boundaries,
    fragments_to_boundaries_from_pairs,
    BoundaryMatchedConfig,
    boundary_matched_f1_pair,
    segment_go_probe_aupr,
    legacy_id_from_new,
)


from scipy.stats import wilcoxon, binomtest  # type: ignore


# =============================================================================
# Utilities
# =============================================================================

def _parse_int_list(s: str) -> List[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out


def _load_df_prot(cluster_dir: Path, prefix: str, go_ann: GOAnnotations, go_aspect: str) -> pd.DataFrame:
    outputs = ClusterOutputs(cluster_dir, prefix)
    res_df = outputs.load_residue_assignments()
    df_prot = build_protein_df_from_residue_assignments(res_df, go_ann, go_aspect=go_aspect)
    return df_prot


def _paired_join(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    require_columns(dfA, ["id", "pdb", "chain", "clusters_int"], "dfA")
    require_columns(dfB, ["id", "pdb", "chain", "clusters_int"], "dfB")

    j = dfA[["id", "pdb", "chain", "clusters_int"]].merge(
        dfB[["id", "clusters_int"]],
        on="id",
        how="inner",
        suffixes=("_A", "_B"),
    )
    if j.empty:
        raise ValueError("No overlapping proteins between A and B (by id).")
    return j


def _summarize_delta(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return {
        "n": float(x.size),
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "median": float(np.median(x)) if x.size else float("nan"),
        "std": float(np.std(x)) if x.size else float("nan"),
        "p10": float(np.quantile(x, 0.10)) if x.size else float("nan"),
        "p90": float(np.quantile(x, 0.90)) if x.size else float("nan"),
    }


def _sign_test(delta: np.ndarray) -> Dict[str, Any]:
    """
    H0: median(delta)=0. Counts positive vs negative, ignores zeros.
    Returns p-value (two-sided).
    """
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    n = pos + neg
    if n == 0:
        return {"n": 0.0, "pos": 0.0, "neg": 0.0, "pvalue_two_sided": float("nan"), "note": "all deltas are zero/NaN"}


    p = float(binomtest(k=min(pos, neg), n=n, p=0.5, alternative="two-sided").pvalue)
   
    return {"n": float(n), "pos": float(pos), "neg": float(neg), "pvalue_two_sided": p}


def _wilcoxon_test(delta: np.ndarray) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank on delta, two-sided. Soft-fails if scipy missing.
    """
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return {"n": 0.0, "pvalue_two_sided": float("nan"), "note": "empty"}
   
    stat = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    return {"n": float(d.size), "pvalue_two_sided": float(stat.pvalue)}


# =============================================================================
# Boundary-matched evaluation (single seed) + multi-seed wrapper
# =============================================================================

def interpro_boundary_matched_eval_one_seed(
    joined: pd.DataFrame,
    interpro_json: Path,
    annotation_types: List[str],
    *,
    cfg: BoundaryMatchedConfig,
    pad_label: int = -1,
    one_based_inclusive: bool = True,
) -> pd.DataFrame:

    test_pdb_ids = joined["pdb"].tolist()
    test_chains = joined["chain"].tolist()

    interpro = InterProManager(pdb_list=test_pdb_ids, filepath=str(interpro_json))
    ann = interpro.get_annotations(test_pdb_ids, test_chains)

    require_columns(ann, ["id", "annotation_type", "fragment_range"], "interpro_annotations")
    ann = ann[ann["annotation_type"].isin(annotation_types)].copy()
    if ann.empty:
        raise ValueError("No InterPro annotations after filtering annotation_types.")

    ann_group = (
        ann.groupby(["id", "annotation_type"])["fragment_range"]
        .apply(list)  # list of tuples (start,end)
        .reset_index()
    )

    df = joined.merge(ann_group, on="id", how="inner")
    if df.empty:
        raise ValueError("No overlap between joined proteins and InterPro annotations.")

    rows = []
    for _, r in df.iterrows():
        cA = r["clusters_int_A"]
        cB = r["clusters_int_B"]
        if not isinstance(cA, list) or not isinstance(cB, list):
            continue
        if len(cA) < 2 or len(cB) < 2:
            continue

        L = min(len(cA), len(cB))
        cA = cA[:L]
        cB = cB[:L]

        predA = clusters_to_boundaries(cA, pad_label=pad_label)
        predB = clusters_to_boundaries(cB, pad_label=pad_label)

        fragments = r["fragment_range"]
        trueB = fragments_to_boundaries_from_pairs(
            fragments, L=L, one_based_inclusive=one_based_inclusive
        )

        m = boundary_matched_f1_pair(predA, predB, trueB, cfg=cfg)

        rows.append(
            {
                "id": r["id"],
                "pdb": r["pdb"],
                "chain": r["chain"],
                "annotation_type": r["annotation_type"],
                "L": int(L),
                "window": int(cfg.window),
                "seed": int(cfg.seed),
                **m,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError("Boundary-matched evaluation produced no rows.")

    out_df["delta_f1"] = out_df["A_f1"].astype(float) - out_df["B_f1"].astype(float)
    out_df["delta_precision"] = out_df["A_precision"].astype(float) - out_df["B_precision"].astype(float)
    out_df["delta_recall"] = out_df["A_recall"].astype(float) - out_df["B_recall"].astype(float)
    return out_df


def aggregate_matched_across_seeds(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Aggregates per (id, annotation_type) across seeds.
    Returns:
      - agg_df: one row per (id, annotation_type) with mean/std over seeds
      - summary: overall + per-type summaries over aggregated deltas
      - significance: paired tests over aggregated protein-level deltas
    """
    if df_all.empty:
        raise ValueError("df_all empty in aggregation")

    group_keys = ["id", "pdb", "chain", "annotation_type"]

    # mean over seeds per row-key
    agg = (
        df_all.groupby(group_keys, as_index=False)
        .agg(
            L=("L", "first"),
            n_target_mean=("n_target", "mean"),
            A_f1_mean=("A_f1", "mean"),
            B_f1_mean=("B_f1", "mean"),
            delta_f1_mean=("delta_f1", "mean"),
            delta_f1_std=("delta_f1", "std"),
            A_precision_mean=("A_precision", "mean"),
            B_precision_mean=("B_precision", "mean"),
            A_recall_mean=("A_recall", "mean"),
            B_recall_mean=("B_recall", "mean"),
            delta_precision_mean=("delta_precision", "mean"),
            delta_recall_mean=("delta_recall", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )

    # summaries over (id,annotation_type) items
    def _summary_block(g: pd.DataFrame) -> Dict[str, Any]:
        d = g["delta_f1_mean"].to_numpy(dtype=float)
        return {
            "n": float(len(g)),
            "A_f1_mean": float(g["A_f1_mean"].mean()),
            "B_f1_mean": float(g["B_f1_mean"].mean()),
            "delta_f1_summary": _summarize_delta(d),
            "n_target_mean": float(g["n_target_mean"].mean()),
        }

    summary: Dict[str, Any] = {"overall": _summary_block(agg), "by_annotation_type": {}}
    for a_type, g in agg.groupby("annotation_type"):
        summary["by_annotation_type"][str(a_type)] = _summary_block(g)

    # protein-level aggregation (mean over annotation types per protein id)
    by_prot = (
        agg.groupby(["id", "pdb", "chain"], as_index=False)
        .agg(
            delta_f1_mean=("delta_f1_mean", "mean"),
            A_f1_mean=("A_f1_mean", "mean"),
            B_f1_mean=("B_f1_mean", "mean"),
            n_types=("annotation_type", "nunique"),
        )
    )
    prot_delta = by_prot["delta_f1_mean"].to_numpy(dtype=float)

    significance: Dict[str, Any] = {
        "protein_level": {
            "sign_test": _sign_test(prot_delta),
            "wilcoxon": _wilcoxon_test(prot_delta),
            "note": "Tests run on per-protein mean(delta_f1) aggregated across annotation types and seeds.",
        },
        "row_level": {
            "sign_test": _sign_test(agg["delta_f1_mean"].to_numpy(dtype=float)),
            "wilcoxon": _wilcoxon_test(agg["delta_f1_mean"].to_numpy(dtype=float)),
            "note": "Tests run on per-(id,annotation_type) mean(delta_f1) aggregated across seeds.",
        },
    }

    return agg, summary, {"by_protein": by_prot, "tests": significance}


# =============================================================================
# Optional: GO probe AUPR comparison (run-level)
# =============================================================================

def compute_go_probe_for_run(
    cluster_dir: Path,
    prefix: str,
    annotation_dir: Path,
    go_aspect: str,
    *,
    min_assigned: int = 5,
    max_assigned: Optional[int] = None,
    test_frac: float = 0.2,
    seed: int = 0,
    min_pos_train: int = 50,
    min_pos_test: int = 10,
) -> Dict[str, Any]:
    outputs = ClusterOutputs(cluster_dir, prefix)
    E, seg_meta = outputs.load_segment_outputs_filtered(min_assigned=min_assigned, max_assigned=max_assigned)

    go_ann = GOAnnotations(annotation_dir)
    go_map_legacy = go_ann.build_map(go_aspect)  # legacy keys
    go_map: Dict[str, List[str]] = {k: v for k, v in go_map_legacy.items()}

    res_df = outputs.load_residue_assignments()
    for pid in res_df["pdb_id"].astype(str).tolist():
        go_map[pid] = go_map_legacy.get(legacy_id_from_new(pid), [])

    return segment_go_probe_aupr(
        E=E,
        seg_meta=seg_meta,
        go_map=go_map,
        test_frac=test_frac,
        seed=seed,
        min_pos_train=min_pos_train,
        min_pos_test=min_pos_test,
    )


# =============================================================================
# CLI + main
# =============================================================================

@dataclass(frozen=True)
class Args:
    runA_dir: Path
    runA_prefix: str
    runB_dir: Path
    runB_prefix: str

    annotation_dir: Path
    go_aspect: str

    output_dir: Path

    unit_eval: bool
    interpro_json: Path
    annotation_types: List[str]
    boundary_window: int
    boundary_seeds: List[int]

    go_probe: bool
    go_probe_test_frac: float
    go_probe_min_pos_train: int
    go_probe_min_pos_test: int
    min_assigned: int
    max_assigned: Optional[int]


def parse_args() -> Args:
    ap = argparse.ArgumentParser()

    ap.add_argument("--runA_dir", type=str, required=True)
    ap.add_argument("--runA_prefix", type=str, required=True)
    ap.add_argument("--runB_dir", type=str, required=True)
    ap.add_argument("--runB_prefix", type=str, required=True)

    ap.add_argument("--annotation_dir", type=str, required=True)
    ap.add_argument("--go_aspect", type=str, default="MF", choices=["MF", "BP", "CC"])

    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--unit_eval", action="store_true", default=False)
    ap.add_argument("--interpro_json", type=str, default="data/GeneOntology/test_interpro.json")
    ap.add_argument("--annotation_types", type=str, default="active_site,binding_site,conserved_site,domain,repeats")
    ap.add_argument("--boundary_window", type=int, default=3)
    ap.add_argument("--boundary_seeds", type=str, default="0,1,2,3,4")

    ap.add_argument("--go_probe", action="store_true", default=False)
    ap.add_argument("--go_probe_test_frac", type=float, default=0.2)
    ap.add_argument("--go_probe_min_pos_train", type=int, default=50)
    ap.add_argument("--go_probe_min_pos_test", type=int, default=10)
    ap.add_argument("--min_assigned", type=int, default=5)
    ap.add_argument("--max_assigned", type=int, default=None)

    a = ap.parse_args()

    return Args(
        runA_dir=Path(a.runA_dir),
        runA_prefix=str(a.runA_prefix),
        runB_dir=Path(a.runB_dir),
        runB_prefix=str(a.runB_prefix),
        annotation_dir=Path(a.annotation_dir),
        go_aspect=str(a.go_aspect),
        output_dir=ensure_dir(Path(a.output_dir)),
        unit_eval=bool(a.unit_eval),
        interpro_json=Path(a.interpro_json),
        annotation_types=[s.strip() for s in str(a.annotation_types).split(",") if s.strip()],
        boundary_window=int(a.boundary_window),
        boundary_seeds=_parse_int_list(a.boundary_seeds),
        go_probe=bool(a.go_probe),
        go_probe_test_frac=float(a.go_probe_test_frac),
        go_probe_min_pos_train=int(a.go_probe_min_pos_train),
        go_probe_min_pos_test=int(a.go_probe_min_pos_test),
        min_assigned=int(a.min_assigned),
        max_assigned=None if a.max_assigned is None else int(a.max_assigned),
    )


def main() -> None:
    args = parse_args()

    go_ann = GOAnnotations(args.annotation_dir)

    dfA = _load_df_prot(args.runA_dir, args.runA_prefix, go_ann, args.go_aspect)
    dfB = _load_df_prot(args.runB_dir, args.runB_prefix, go_ann, args.go_aspect)
    joined = _paired_join(dfA, dfB)
    log.info(f"Overlapping proteins: {len(joined)}")

    report: Dict[str, Any] = {
        "runA": {"dir": str(args.runA_dir), "prefix": args.runA_prefix},
        "runB": {"dir": str(args.runB_dir), "prefix": args.runB_prefix},
        "go_aspect": args.go_aspect,
        "n_overlap_proteins": float(len(joined)),
    }

    unit_dir = ensure_dir(args.output_dir / "unit_eval")

    # -------------------------
    # Boundary-matched (multi-seed)
    # -------------------------
    if args.unit_eval:
        all_seed_dfs: List[pd.DataFrame] = []

        for seed in args.boundary_seeds:
            cfg = BoundaryMatchedConfig(window=args.boundary_window, seed=int(seed))

            df_seed = interpro_boundary_matched_eval_one_seed(
                joined=joined,
                interpro_json=args.interpro_json,
                annotation_types=args.annotation_types,
                cfg=cfg,
                pad_label=-1,
                one_based_inclusive=True,
            )

            out_csv = unit_dir / f"interpro_boundary_f1_matched_seed{seed}.csv"
            df_seed.to_csv(out_csv, index=False)
            log.info(f"Saved seed CSV: {out_csv}")

            all_seed_dfs.append(df_seed)

        df_all = pd.concat(all_seed_dfs, ignore_index=True)

        # Aggregate across seeds
        agg_df, agg_summary, sig_bundle = aggregate_matched_across_seeds(df_all)

        agg_csv = unit_dir / "interpro_boundary_f1_matched_agg.csv"
        agg_df.to_csv(agg_csv, index=False)
        log.info(f"Saved aggregated CSV: {agg_csv}")

        # Per-protein delta file (nice for plots)
        by_prot_df: pd.DataFrame = sig_bundle["by_protein"]
        by_prot_csv = unit_dir / "interpro_boundary_f1_matched_delta_by_protein.csv"
        by_prot_df.to_csv(by_prot_csv, index=False)
        log.info(f"Saved per-protein delta CSV: {by_prot_csv}")

        agg_json = unit_dir / "interpro_boundary_f1_matched_agg_summary.json"
        agg_json.write_text(json.dumps(agg_summary, indent=2))
        log.info(f"Saved aggregated summary JSON: {agg_json}")

        sig_json = unit_dir / "interpro_boundary_f1_matched_significance.json"
        sig_json.write_text(json.dumps(sig_bundle["tests"], indent=2))
        log.info(f"Saved significance JSON: {sig_json}")

        report["interpro_boundary_f1_matched"] = {
            "boundary_window": float(args.boundary_window),
            "boundary_seeds": [int(s) for s in args.boundary_seeds],
            "agg_summary": agg_summary,
            "significance": sig_bundle["tests"],
            "files": {
                "agg_csv": str(agg_csv),
                "by_protein_csv": str(by_prot_csv),
                "agg_summary_json": str(agg_json),
                "significance_json": str(sig_json),
            },
        }

    # -------------------------
    # GO probe comparison (run-level)
    # -------------------------
    if args.go_probe:
        probeA = compute_go_probe_for_run(
            cluster_dir=args.runA_dir,
            prefix=args.runA_prefix,
            annotation_dir=args.annotation_dir,
            go_aspect=args.go_aspect,
            min_assigned=args.min_assigned,
            max_assigned=args.max_assigned,
            test_frac=args.go_probe_test_frac,
            seed=0,
            min_pos_train=args.go_probe_min_pos_train,
            min_pos_test=args.go_probe_min_pos_test,
        )
        probeB = compute_go_probe_for_run(
            cluster_dir=args.runB_dir,
            prefix=args.runB_prefix,
            annotation_dir=args.annotation_dir,
            go_aspect=args.go_aspect,
            min_assigned=args.min_assigned,
            max_assigned=args.max_assigned,
            test_frac=args.go_probe_test_frac,
            seed=0,
            min_pos_train=args.go_probe_min_pos_train,
            min_pos_test=args.go_probe_min_pos_test,
        )

        out_probe = {"A": probeA, "B": probeB}
        if "micro_aupr" in probeA and "micro_aupr" in probeB:
            out_probe["delta_micro"] = float(probeA["micro_aupr"] - probeB["micro_aupr"])
        if "macro_aupr" in probeA and "macro_aupr" in probeB:
            out_probe["delta_macro"] = float(probeA["macro_aupr"] - probeB["macro_aupr"])

        report["segment_go_probe_aupr"] = out_probe

    out = args.output_dir / "compare_report.json"
    out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote comparison report: {out}")
    log.info("Done.")


if __name__ == "__main__":
    main()
