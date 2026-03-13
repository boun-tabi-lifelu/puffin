# src/cluster_structural_compare.py
# -----------------------------------------------------------------------------
# Purpose
#   Compare TWO clusterings (A vs B) on structural coherence metrics for segments.
#
# Mandatory metrics (per segment)
#   - mean intra-segment Cα distance
#   - contact density (intra-segment contacts / possible pairs)
#   - cut ratio (inter-segment contacts / total contacts incident to segment)
#   - compactness proxies:
#       * radius of gyration (Rg)
#       * packing density: N / (Rg^3 + eps)
#       * (optional) SASA-based compactness if freesasa installed
#
# Baselines
#   - Random size-matched partitions (within each protein, shuffle labels preserving segment sizes)
#
# Outputs
#   - structural_eval/segments_A.csv, segments_B.csv
#   - structural_eval/segments_random_A.csv, segments_random_B.csv
#   - structural_eval/summary.json  (A vs B + baselines)
#
# Inputs expected
#   - runA_dir/<prefix>_residue_assignments.csv
#   - runB_dir/<prefix>_residue_assignments.csv
#   - structure_dir containing PDB or mmCIF files for the proteins
# -----------------------------------------------------------------------------

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger as log

# ----------------------------
# Optional deps (soft)
# ----------------------------

from Bio.PDB import PDBParser, MMCIFParser  # type: ignore
from scipy.spatial import cKDTree  # type: ignore


# =============================================================================
# Parsing helpers
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}. Found: {list(df.columns)}")


def parse_csv_int_list(s: Any) -> List[int]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return list(map(int, str(s).split(",")))


def parse_csv_str_list(s: Any) -> List[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return list(map(str, str(s).split(",")))


def split_id(pdb_id: str) -> Tuple[str, str]:
    """
    Accepts:
      - 3ONG-B_B
      - 1AD3-A_A
      - 3ONG-B
      - 3ONG
    Returns: (pdb, chain) with chain possibly 'ALL'
    """
    s = str(pdb_id)
    if "_" in s:
        left, _ = s.split("_", 1)
    else:
        left = s
    if "-" in left:
        pdb, chain = left.split("-", 1)
        return pdb.upper(), chain
    return left.upper(), "ALL"


def residue_number_key(res_str: str) -> int:
    # handles "chain:resname:resnum" OR anything containing integers
    parts = str(res_str).split(":")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except Exception:
            pass
    m = re.findall(r"-?\d+", str(res_str))
    return int(m[-1]) if m else 10**9


def residue_chain_key(res_str: str, default_chain: str) -> Tuple[str, int]:
    parts = str(res_str).split(":")
    if len(parts) >= 3:
        ch = str(parts[0]) if str(parts[0]) != "" else default_chain
        try:
            return ch, int(parts[2])
        except Exception:
            return ch, residue_number_key(res_str)
    return default_chain, residue_number_key(res_str)


# =============================================================================
# Structure loading + CA coordinate extraction
# =============================================================================

@dataclass(frozen=True)
class StructureConfig:
    structure_dir: Path
    fmt: str = "auto"  # auto|pdb|cif
    prefer_cif: bool = True


class StructureCache:
    def __init__(self, cfg: StructureConfig):
        self.cfg = cfg
        self._cache: Dict[str, Any] = {}

        self._pdb_parser = PDBParser(QUIET=True)
        self._cif_parser = MMCIFParser(QUIET=True)

    def _resolve_path(self, pdb: str, chain: str = None) -> Tuple[Path, str]:
        pdb_l = pdb.lower()
        candidates = []
    
        if self.cfg.fmt in ("auto", "cif") or (self.cfg.fmt == "auto" and self.cfg.prefer_cif):
            candidates += [
                (self.cfg.structure_dir / f"{pdb}.cif", "cif"),
                (self.cfg.structure_dir / f"{pdb_l}.cif", "cif"),
                (self.cfg.structure_dir / f"{pdb}.mmcif", "cif"),
                (self.cfg.structure_dir / f"{pdb_l}.mmcif", "cif"),
                
            ]
            if chain is not None:
                candidates += [
                    (self.cfg.structure_dir / f"{pdb}_{chain}.cif", "cif"),
                    (self.cfg.structure_dir / f"{pdb_l}_{chain}.cif", "cif"),
                    (self.cfg.structure_dir / f"{pdb}_{chain}.mmcif", "cif"),
                    (self.cfg.structure_dir / f"{pdb_l}_{chain}.mmcif", "cif"),
                ]
        if self.cfg.fmt in ("auto", "pdb") or (self.cfg.fmt == "auto" and not self.cfg.prefer_cif):
            candidates += [
                (self.cfg.structure_dir / f"{pdb}.pdb", "pdb"),
                (self.cfg.structure_dir / f"{pdb_l}.pdb", "pdb"),
            ]
            if chain is not None:
                candidates += [
                    (self.cfg.structure_dir / f"{pdb}_{chain}.pdb", "pdb"),
                    (self.cfg.structure_dir / f"{pdb_l}_{chain}.pdb", "pdb"),
                ]

        for p, kind in candidates:
            if p.exists():
                return p, kind
        print(candidates)
        raise FileNotFoundError(f"Structure file not found for {pdb} in {self.cfg.structure_dir}")

    def load(self, pdb: str, chain: str = None) -> Any:
        if pdb in self._cache:
            return self._cache[pdb]
        path, kind = self._resolve_path(pdb, chain=chain)
        struct_id = pdb
        if kind == "cif":
            s = self._cif_parser.get_structure(struct_id, str(path))
        else:
            s = self._pdb_parser.get_structure(struct_id, str(path))
        self._cache[pdb] = s
        return s

    def get_ca_coords(
        self,
        pdb: str,
        chain: str,
        residue_nums: List[int],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Returns:
          coords: (M,3) float32 for residues found
          kept_resnums: aligned residue numbers found in structure (same order as coords)
        """
        s = self.load(pdb, chain)
        model = next(iter(s.get_models()))
        chains = {c.id: c for c in model.get_chains()}

        # If chain == ALL, try to find residues across all chains (rarely desired).
        # In practice you likely have chain, so we prioritize it.
        search_chains = [chain] if (chain != "ALL" and chain in chains) else list(chains.keys())

        wanted = set(int(x) for x in residue_nums)
        coords = []
        kept = []

        for ch_id in search_chains:
            if ch_id not in chains:
                continue
            ch = chains[ch_id]
            for res in ch.get_residues():
                # Bio.PDB residue id: (hetflag, resseq, icode)
                rid = res.id
                resseq = int(rid[1])
                if resseq in wanted:
                    if "CA" in res:
                        ca = res["CA"].get_coord()
                        coords.append(ca)
                        kept.append(resseq)

        if not coords:
            return np.zeros((0, 3), dtype=np.float32), []

        # Preserve order by residue number
        order = np.argsort(np.array(kept, dtype=int))
        coords = np.array(coords, dtype=np.float32)[order]
        kept = list(np.array(kept, dtype=int)[order])
        return coords, kept


# =============================================================================
# Graph/contact metrics
# =============================================================================

@dataclass(frozen=True)
class MetricConfig:
    contact_cutoff: float = 10.0   # Å (CA-CA)
    ca_pair_sample_max: int = 4000  # for mean intra distance if segment huge
    eps: float = 1e-8


def _pairwise_mean_distance(coords: np.ndarray, sample_max: int, rng: np.random.Generator) -> float:
    n = coords.shape[0]
    if n < 2:
        return float("nan")
    # Full pairwise if small
    if n <= 300:
        d = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=-1))
        # upper triangle mean
        iu = np.triu_indices(n, k=1)
        return float(dist[iu].mean()) if iu[0].size else float("nan")

    # Sample pairs if large
    P = min(sample_max, n * 10)
    i = rng.integers(0, n, size=P)
    j = rng.integers(0, n, size=P)
    m = i != j
    i, j = i[m], j[m]
    if i.size == 0:
        return float("nan")
    diff = coords[i] - coords[j]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    return float(dist.mean())


def _build_contacts(coords: np.ndarray, cutoff: float) -> List[Tuple[int, int]]:
    """
    Return list of undirected edges (i,j), i<j, where CA distance <= cutoff.
    Uses cKDTree if available, else O(n^2) for small n.
    """
    n = coords.shape[0]
    if n < 2:
        return []
  
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff)
    return [(int(i), int(j)) for (i, j) in pairs]


def _radius_of_gyration(coords: np.ndarray, eps: float) -> float:
    n = coords.shape[0]
    if n == 0:
        return float("nan")
    mu = coords.mean(axis=0, keepdims=True)
    d2 = np.sum((coords - mu) ** 2, axis=1)
    return float(np.sqrt(d2.mean() + eps))




# =============================================================================
# Segment extraction + size-matched random baseline
# =============================================================================

def _segments_from_labels(
    residue_nums: List[int],
    labels: List[int],
    pad_label: int = -1,
) -> Dict[int, List[int]]:
    """
    Returns {label: [residue_num,...]} excluding pad_label.
    """
    segs: Dict[int, List[int]] = {}
    for rn, lab in zip(residue_nums, labels):
        if lab == pad_label:
            continue
        segs.setdefault(int(lab), []).append(int(rn))
    return segs


def _size_matched_random_labels(labels: List[int], pad_label: int, rng: np.random.Generator) -> List[int]:
    """
    Random partition baseline preserving per-protein segment sizes:
      - Keep same multiset of labels (excluding PAD), but permute their positions.
    """
    labs = np.array(labels, dtype=int)
    mask = labs != int(pad_label)
    vals = labs[mask].copy()
    rng.shuffle(vals)
    out = labs.copy()
    out[mask] = vals
    return out.tolist()


# =============================================================================
# Per-protein evaluation
# =============================================================================

def compute_segment_metrics_for_protein(
    *,
    pdb: str,
    chain: str,
    residue_nums: List[int],
    labels: List[int],
    struct_cache: StructureCache,
    mc: MetricConfig,
    rng: np.random.Generator,
    pad_label: int = -1,
) -> List[Dict[str, Any]]:
    coords_all, kept = struct_cache.get_ca_coords(pdb, chain, residue_nums)
    if coords_all.shape[0] < 2:
        return []

    # Align labels to kept residues (by residue number)
    rn_to_lab = {int(rn): int(lab) for rn, lab in zip(residue_nums, labels)}
    kept_labels = [rn_to_lab.get(int(rn), pad_label) for rn in kept]

    # Global contact graph on protein (over kept residues)
    edges = _build_contacts(coords_all, cutoff=mc.contact_cutoff)
    # Precompute incident counts per node
    deg = np.zeros(coords_all.shape[0], dtype=int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1

    segs = _segments_from_labels(kept, kept_labels, pad_label=pad_label)

    # index map residue_num -> index in coords_all
    idx_map = {int(rn): int(i) for i, rn in enumerate(kept)}

    out_rows: List[Dict[str, Any]] = []
    for seg_id, seg_resnums in segs.items():
        if len(seg_resnums) < 2:
            continue
        idxs = np.array([idx_map[rn] for rn in seg_resnums if rn in idx_map], dtype=int)
        if idxs.size < 2:
            continue

        seg_coords = coords_all[idxs]

        # mean intra-segment CA distance
        mean_intra_ca = _pairwise_mean_distance(seg_coords, sample_max=mc.ca_pair_sample_max, rng=rng)

        # contact density (within segment)
        # count intra edges among idxs
        idx_set = set(idxs.tolist())
        intra_edges = 0
        cut_edges = 0
        for i, j in edges:
            in_i = i in idx_set
            in_j = j in idx_set
            if in_i and in_j:
                intra_edges += 1
            elif in_i != in_j:
                # one in, one out
                cut_edges += 1

        # possible pairs inside segment
        n = int(idxs.size)
        possible_pairs = n * (n - 1) / 2
        contact_density = float(intra_edges / possible_pairs) if possible_pairs > 0 else float("nan")

        # cut ratio: inter / (inter + intra incident)
        # Here "incident edges" to the segment = intra_edges + cut_edges (counting undirected edges once)
        cut_ratio = float(cut_edges / max(1, (cut_edges + intra_edges)))

        # compactness proxies
        seg_resnums = sorted(set(seg_resnums))
        idxs = np.array([idx_map[rn] for rn in seg_resnums if rn in idx_map], dtype=int)
        if idxs.size < 3:
            continue

        seg_coords = coords_all[idxs]
        if np.unique(seg_coords, axis=0).shape[0] < 3:
            continue

        rg = _radius_of_gyration(seg_coords, eps=mc.eps)
        rg_clip = max(float(rg), 1.0)  # 1 Å lower bound to prevent explosions
        packing = float(n / (rg_clip ** 3 + mc.eps))
        log_packing = float(np.log(n / (rg_clip ** 3 + mc.eps)))




        out_rows.append(
            {
                "pdb": pdb,
                "chain": chain,
                "segment_id": int(seg_id),
                "n_res": int(n),
                "mean_intra_ca_dist": float(mean_intra_ca),
                "contact_density": float(contact_density),
                "intra_edges": float(intra_edges),
                "cut_edges": float(cut_edges),
                "cut_ratio": float(cut_ratio),
                "rg": float(rg),
                "rg_clipped": float(rg_clip),
                "packing_density": float(packing),
                "log_packing_density": float(log_packing),
            }
        )

    return out_rows


# =============================================================================
# Run evaluation for a clustering
# =============================================================================

def load_residue_assignments(run_dir: Path, prefix: str) -> pd.DataFrame:
    p = run_dir / f"{prefix}_residue_assignments.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    require_columns(df, ["pdb_id", "residue_ids", "cluster_ids"], "residue_assignments")
    return df


def eval_run_segments(
    *,
    run_dir: Path,
    prefix: str,
    struct_cache: StructureCache,
    mc: MetricConfig,
    seeds: List[int],
    pad_label: int = -1,
    random_baseline: bool = False,
) -> pd.DataFrame:
    df = load_residue_assignments(run_dir, prefix)

    all_rows: List[Dict[str, Any]] = []
    # Use multiple seeds mainly for random baseline stability and intra-distance sampling stability
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for _, r in df.iterrows():
            pdb, chain = split_id(str(r["pdb_id"]))
            residue_ids = parse_csv_str_list(r["residue_ids"])
            labels = parse_csv_int_list(r["cluster_ids"])
            if not residue_ids or not labels:
                continue
            # sort by residue index
            order = np.argsort([residue_number_key(x) for x in residue_ids])
            L = min(len(order), len(labels))
            residue_ids = [residue_ids[i] for i in order[:L]]
            labels = [labels[i] for i in order[:L]]

            residue_nums = [residue_chain_key(x, default_chain=chain)[1] for x in residue_ids]

            if random_baseline:
                labels = _size_matched_random_labels(labels, pad_label=pad_label, rng=rng)

            try:
                rows = compute_segment_metrics_for_protein(
                    pdb=pdb,
                    chain=chain,
                    residue_nums=residue_nums,
                    labels=labels,
                    struct_cache=struct_cache,
                    mc=mc,
                    rng=rng,
                    pad_label=pad_label,
                )
            except FileNotFoundError as e:
                # structure missing -> skip protein
                log.warning(str(e))
                continue
            except Exception as e:
                log.warning(f"Failed {pdb}-{chain}: {e}")
                continue

            for row in rows:
                row["seed"] = int(seed)
                row["run_prefix"] = str(prefix)
                row["baseline"] = "random_size_matched" if random_baseline else "model"
                all_rows.append(row)

    out = pd.DataFrame(all_rows)
    return out


# =============================================================================
# Summaries + comparisons
# =============================================================================

def summarize_segments(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
    if df.empty:
        return {"n_segments": 0.0}

    out: Dict[str, Any] = {"n_segments": float(len(df))}
    for c in metric_cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna().astype(float)
        out[c] = {
            "mean": float(x.mean()) if len(x) else float("nan"),
            "median": float(x.median()) if len(x) else float("nan"),
            "p10": float(x.quantile(0.10)) if len(x) else float("nan"),
            "p90": float(x.quantile(0.90)) if len(x) else float("nan"),
        }
    # size distribution
    n = pd.to_numeric(df["n_res"], errors="coerce").dropna().astype(float)
    out["segment_size"] = {
        "mean": float(n.mean()) if len(n) else float("nan"),
        "median": float(n.median()) if len(n) else float("nan"),
        "p10": float(n.quantile(0.10)) if len(n) else float("nan"),
        "p90": float(n.quantile(0.90)) if len(n) else float("nan"),
    }
    return out


def delta_summary(dfA: pd.DataFrame, dfB: pd.DataFrame, key_cols: List[str], metric_cols: List[str]) -> Dict[str, Any]:
    """
    Join segment-level rows by (pdb,chain,seed,segment_id) is NOT well-defined across runs
    because segment ids are not aligned. So we compare DISTRIBUTIONS:
      - overall mean/median differences
      - per-protein summaries could be added later if needed
    """
    out: Dict[str, Any] = {}
    for c in metric_cols + ["n_res"]:
        xA = pd.to_numeric(dfA[c], errors="coerce").dropna().astype(float)
        xB = pd.to_numeric(dfB[c], errors="coerce").dropna().astype(float)
        out[c] = {
            "A_mean": float(xA.mean()) if len(xA) else float("nan"),
            "B_mean": float(xB.mean()) if len(xB) else float("nan"),
            "delta_mean": float((xB.mean() - xA.mean())) if (len(xA) and len(xB)) else float("nan"),
            "A_median": float(xA.median()) if len(xA) else float("nan"),
            "B_median": float(xB.median()) if len(xB) else float("nan"),
            "delta_median": float((xB.median() - xA.median())) if (len(xA) and len(xB)) else float("nan"),
        }
    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--runA_dir", type=str, required=True)
    ap.add_argument("--runA_prefix", type=str, required=True)
    ap.add_argument("--runB_dir", type=str, required=True)
    ap.add_argument("--runB_prefix", type=str, required=True)

    ap.add_argument("--structure_dir", type=str, required=True)
    ap.add_argument("--structure_fmt", type=str, default="auto", choices=["auto", "pdb", "cif"])
    ap.add_argument("--prefer_cif", action="store_true", default=False)

    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--contact_cutoff", type=float, default=8.0)
    ap.add_argument("--seeds", type=str, default="0")  # used for sampling stability
    ap.add_argument("--random_baseline_seeds", type=str, default="0,1,2,3,4")

    ap.add_argument("--pad_label", type=int, default=-1)

    return ap.parse_args()


def main():
    a = parse_args()

    seeds = [int(x.strip()) for x in str(a.seeds).split(",") if x.strip() != ""]
    rnd_seeds = [int(x.strip()) for x in str(a.random_baseline_seeds).split(",") if x.strip() != ""]

    out_dir = ensure_dir(Path(a.output_dir))
    eval_dir = ensure_dir(out_dir / "structural_eval")

    struct_cache = StructureCache(
        StructureConfig(
            structure_dir=Path(a.structure_dir),
            fmt=str(a.structure_fmt),
            prefer_cif=bool(a.prefer_cif),
        )
    )
    mc = MetricConfig(contact_cutoff=float(a.contact_cutoff))

    metric_cols = [
        "mean_intra_ca_dist",
        "contact_density",
        "cut_ratio",
        "rg",
        "packing_density",
    ]

    # ---- Model outputs ----
    log.info("Evaluating A (model)...")
    dfA = eval_run_segments(
        run_dir=Path(a.runA_dir),
        prefix=str(a.runA_prefix),
        struct_cache=struct_cache,
        mc=mc,
        seeds=seeds,
        pad_label=int(a.pad_label),
        random_baseline=False,
    )
    dfA.to_csv(eval_dir / "segments_A.csv", index=False)

    log.info("Evaluating B (model)...")
    dfB = eval_run_segments(
        run_dir=Path(a.runB_dir),
        prefix=str(a.runB_prefix),
        struct_cache=struct_cache,
        mc=mc,
        seeds=seeds,
        pad_label=int(a.pad_label),
        random_baseline=False,
    )
    dfB.to_csv(eval_dir / "segments_B.csv", index=False)

    # ---- Random size-matched baselines ----
    log.info("Evaluating random baseline for A (size-matched)...")
    dfA_r = eval_run_segments(
        run_dir=Path(a.runA_dir),
        prefix=str(a.runA_prefix),
        struct_cache=struct_cache,
        mc=mc,
        seeds=rnd_seeds,
        pad_label=int(a.pad_label),
        random_baseline=True,
    )
    dfA_r.to_csv(eval_dir / "segments_random_A.csv", index=False)

    log.info("Evaluating random baseline for B (size-matched)...")
    dfB_r = eval_run_segments(
        run_dir=Path(a.runB_dir),
        prefix=str(a.runB_prefix),
        struct_cache=struct_cache,
        mc=mc,
        seeds=rnd_seeds,
        pad_label=int(a.pad_label),
        random_baseline=True,
    )
    dfB_r.to_csv(eval_dir / "segments_random_B.csv", index=False)

    def bin_summary(df, cols, bin_edges):
        d = df.copy()
        d["n_res_bin"] = pd.cut(d["n_res"], bins=bin_edges, include_lowest=True)
        out = {}
        for b, g in d.groupby("n_res_bin"):
            out[str(b)] = {"n": float(len(g))}
            for c in cols:
                x = pd.to_numeric(g[c], errors="coerce").dropna().astype(float)
                out[str(b)][c] = {
                    "mean": float(x.mean()) if len(x) else float("nan"),
                    "median": float(x.median()) if len(x) else float("nan"),
                }
        return out

    bin_edges = [5,10,15,20,25,30,35,40,45,50,60,70,80,999]
    metric_cols2 = ["mean_intra_ca_dist","contact_density","cut_ratio","rg","log_packing_density"]

    summary["A"]["by_size_bin"] = bin_summary(dfA, metric_cols2, bin_edges)
    summary["B"]["by_size_bin"] = bin_summary(dfB, metric_cols2, bin_edges)
    summary["A"]["random_by_size_bin"] = bin_summary(dfA_r, metric_cols2, bin_edges)
    summary["B"]["random_by_size_bin"] = bin_summary(dfB_r, metric_cols2, bin_edges)


    # ---- Summaries ----
    summary = {
        "config": {
            "contact_cutoff": float(a.contact_cutoff),
            "seeds_model": seeds,
            "seeds_random": rnd_seeds,
            "structure_dir": str(a.structure_dir),
            "structure_fmt": str(a.structure_fmt),
            "prefer_cif": bool(a.prefer_cif),
        },
        "A": {
            "run": {"dir": str(a.runA_dir), "prefix": str(a.runA_prefix)},
            "summary": summarize_segments(dfA, metric_cols),
            "random_size_matched_summary": summarize_segments(dfA_r, metric_cols),
        },
        "B": {
            "run": {"dir": str(a.runB_dir), "prefix": str(a.runB_prefix)},
            "summary": summarize_segments(dfB, metric_cols),
            "random_size_matched_summary": summarize_segments(dfB_r, metric_cols),
        },
        "distribution_deltas_B_minus_A": {
            "model": delta_summary(dfA, dfB, key_cols=["pdb", "chain"], metric_cols=metric_cols),
            "random_baseline": delta_summary(dfA_r, dfB_r, key_cols=["pdb", "chain"], metric_cols=metric_cols),
        },
        "interpretation_hints": {
            "mean_intra_ca_dist": "Lower is more compact within segments.",
            "contact_density": "Higher means more intra-segment contacts (PP-like coherence).",
            "cut_ratio": "Lower means fewer cut edges relative to internal edges (better separation).",
            "rg": "Lower is more compact (size-dependent). Use packing_density for normalization.",
            "packing_density": "Higher indicates tighter packing for given size.",
        },
    }

    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote: {eval_dir / 'summary.json'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
