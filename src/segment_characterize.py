# src/segment_characterize.py
# -----------------------------------------------------------------------------
# Purpose
#   Characterize discovered units (segments/clusters) from residue assignments.
#   Reports:
#     - size + contiguity + fragmentation metrics
#     - label usage inequality (gini/entropy)
#     - OPTIONAL structural coherence metrics per segment from PDB/mmCIF:
#         * mean intra-segment Cα distance
#         * contact density (Cα contacts / possible pairs)
#         * cut ratio (inter contacts / (inter+intra) incident to segment)
#         * radius of gyration (Rg)
#         * packing density N / (Rg^3)
#     - OPTIONAL size-matched random baseline per protein (shuffle labels, preserve counts)
#
# Inputs
#   - <cluster_dir>/<prefix>_residue_assignments.csv
#       columns: pdb_id, residue_ids, cluster_ids, (optional) K
#
# Outputs (under <output_dir>/)
#   - proteins.csv
#   - segments.csv
#   - summary.json
#   - plots/*.pdf and plots/*.png
#
# Notes
#   - "cluster_ids" are treated as *residue-level labels*.
#   - Segment = set of residues with same label within a protein.
#   - Contiguity is computed on residue index order (after sorting).
# -----------------------------------------------------------------------------


import os
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm


from loguru import logger as log
import warnings
# Silence BioPython warnings
warnings.filterwarnings("ignore", module="Bio")
warnings.filterwarnings("ignore", module="Bio.PDB")
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib as mpl
import matplotlib.pyplot as plt


from Bio.PDB import PDBParser, MMCIFParser  
from scipy.spatial import cKDTree  



# =============================================================================
# Generic utils
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Found: {list(df.columns)}")


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
    Accepts: 3ONG-B_B, 1AD3-A_A, 3ONG-B, 3ONG
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
    """
    Parses residue string of form CHAIN:RESNAME:RESNUM
    Returns residue_number as int, or large number if not found.
    """
    parts = str(res_str).split(":")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except Exception:
            pass
    m = re.findall(r"-?\d+", str(res_str))
    return int(m[-1]) if m else 10**9


def residue_chain_and_num(res_str: str, default_chain: str) -> Tuple[str, int]:
    """
    Parses residue string of form CHAIN:RESNAME:RESNUM
    Returns (chain, residue_number)
    """
    parts = str(res_str).split(":")
    if len(parts) >= 3:
        ch = str(parts[0]) if str(parts[0]) else default_chain
        try:
            return ch, int(parts[2])
        except Exception:
            return ch, residue_number_key(res_str)
    return default_chain, residue_number_key(res_str)

def subsample_df(
    df: pd.DataFrame,
    max_n: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if max_n is None or len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=seed)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def quantiles(x: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q*100):02d}": float("nan") for q in qs}
    out = np.quantile(x, qs)
    return {f"q{int(q*100):02d}": float(v) for q, v in zip(qs, out.tolist())}


def trimmed_mean(x: np.ndarray, trim: float = 0.1) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    lo = np.quantile(x, trim)
    hi = np.quantile(x, 1.0 - trim)
    y = x[(x >= lo) & (x <= hi)]
    return float(y.mean()) if y.size else float("nan")


def gini_coefficient(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return float("nan")
    s = x.sum()
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(g)


def normalized_entropy(counts: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(counts, dtype=float)
    x = x[np.isfinite(x)]
    s = x.sum()
    if s <= 0:
        return float("nan")
    p = x / s
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log(p))
    Hmax = np.log(len(p)) if len(p) > 1 else 1.0
    return float(H / max(eps, Hmax))


# =============================================================================
# Publication-ready plotting style
# =============================================================================

def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 350,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,  # embed fonts (TrueType) for Illustrator-friendly PDFs
            "ps.fonttype": 42,
        }
    )


def savefig_dual(fig: plt.Figure, out_base: Path) -> None:
    ensure_dir(out_base.parent)
    fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Structural: load coords and compute graph metrics
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
        self.failed: set = set()

    def _resolve_path(self, pdb: str, chain: Optional[str] = None) -> Tuple[Path, str]:
        pdb_u = pdb.upper()
        pdb_l = pdb.lower()
        cands: List[Tuple[Path, str]] = []

        def add(kind: str, stem: str) -> None:
            if kind == "cif":
                cands.extend(
                    [
                        (self.cfg.structure_dir / f"{stem}.cif", "cif"),
                        (self.cfg.structure_dir / f"{stem}.mmcif", "cif"),
                    ]
                )
            else:
                cands.append((self.cfg.structure_dir / f"{stem}.pdb", "pdb"))

        # prefer cif if requested
        if self.cfg.fmt in ("auto", "cif") and self.cfg.prefer_cif:
            add("cif", pdb_u)
            add("cif", pdb_l)
            if chain:
                add("cif", f"{pdb_u}_{chain}")
                add("cif", f"{pdb_l}_{chain}")

        if self.cfg.fmt in ("auto", "pdb"):
            add("pdb", pdb_u)
            add("pdb", pdb_l)
            if chain:
                add("pdb", f"{pdb_u}_{chain}")
                add("pdb", f"{pdb_l}_{chain}")

        # fallback to cif if fmt=auto but prefer_cif=False
        if self.cfg.fmt == "auto" and not self.cfg.prefer_cif:
            add("cif", pdb_u)
            add("cif", pdb_l)
            if chain:
                add("cif", f"{pdb_u}_{chain}")
                add("cif", f"{pdb_l}_{chain}")

        for p, kind in cands:
            if p.exists():
                return p, kind

        raise FileNotFoundError(f"Structure file not found for {pdb_u} in {self.cfg.structure_dir}")

    def load(self, pdb: str, chain: Optional[str] = None) -> Any:
        try: 
            pdb_u = pdb.upper()
            if pdb_u in self._cache:
                return self._cache[pdb_u]
            path, kind = self._resolve_path(pdb_u, chain=chain)
            if kind == "cif":
                s = self._cif_parser.get_structure(pdb_u, str(path))
            else:
                s = self._pdb_parser.get_structure(pdb_u, str(path))
            self._cache[pdb_u] = s
            return s
        except Exception as e:
            self.failed.add(pdb)
            # log.warning(f"Failed to load structure for {pdb} chain={chain}: {e}")
            return None

    def get_ca_coords(
        self,
        pdb: str,
        chain: str,
        residue_nums: List[int],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Returns:
          coords: (M,3) float32
          kept_resnums: residue numbers found in structure (sorted)
        """
        s = self.load(pdb, chain=chain)
        if s is None:
            return np.zeros((0, 3), dtype=np.float32), []
        # model = next(iter(s.get_models()))
        model = next(s.get_models(), None)
        chains = {c.id: c for c in model.get_chains()}

        search_chains = [chain] if (chain != "ALL" and chain in chains) else list(chains.keys())
        wanted = set(int(x) for x in residue_nums)

        coords: List[np.ndarray] = []
        kept: List[int] = []
        for ch_id in search_chains:
            if ch_id not in chains:
                continue
            ch = chains[ch_id]
            for res in ch.get_residues():
                rid = res.id  # (hetflag, resseq, icode)
                resseq = int(rid[1])
                if resseq in wanted and "CA" in res:
                    coords.append(res["CA"].get_coord())
                    kept.append(resseq)

        if not coords:
            return np.zeros((0, 3), dtype=np.float32), []

        order = np.argsort(np.array(kept, dtype=int))
        coords_arr = np.array(coords, dtype=np.float32)[order]
        kept_sorted = list(np.array(kept, dtype=int)[order])
        return coords_arr, kept_sorted



@dataclass(frozen=True)
class ProteinStatic:
    pdb_id: str
    pdb: str
    chain: str
    residue_nums_sorted: List[int]
    labels_sorted: List[int]


def _prepare_static(res_df: pd.DataFrame) -> List[ProteinStatic]:
    out: List[ProteinStatic] = []
    for r in res_df.itertuples(index=False):
        pdb_id = str(getattr(r, "pdb_id"))
        pdb, chain = split_id(pdb_id)

        residue_ids = parse_csv_str_list(getattr(r, "residue_ids"))
        labels = parse_csv_int_list(getattr(r, "cluster_ids"))
        if not residue_ids or not labels:
            continue

        order = np.argsort([residue_number_key(x) for x in residue_ids])
        L = min(len(order), len(labels))
        idx = order[:L]

        residue_ids_sorted = [residue_ids[i] for i in idx]
        labels_sorted = [labels[i] for i in idx]

        residue_nums_sorted = [
            residue_chain_and_num(x, default_chain=chain)[1] for x in residue_ids_sorted
        ]

        out.append(
            ProteinStatic(
                pdb_id=pdb_id,
                pdb=pdb,
                chain=chain,
                residue_nums_sorted=residue_nums_sorted,
                labels_sorted=labels_sorted,
            )
        )
    return out

def _preload_coords(
    items: List[ProteinStatic],
    struct_cache: Optional[StructureCache],
    max_workers: Optional[int] = None,
    show_progress: bool = True,
):
    if struct_cache is None:
        return None  # no preload possible

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 8) * 4)

    coords_map: Dict[Tuple[str, str, Tuple[int, ...]], Tuple[np.ndarray, List[int]]] = {}

    def _one(it: ProteinStatic):
        key = (it.pdb, it.chain, tuple(it.residue_nums_sorted))
        try:
            coords_all, kept_resnums = struct_cache.get_ca_coords(it.pdb, it.chain, it.residue_nums_sorted)
            return key, (coords_all, kept_resnums)
        except Exception:
            return key, None  # mark as unavailable

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, it) for it in items]
        it_futs = as_completed(futs)
        if show_progress:
            it_futs = tqdm(it_futs, total=len(futs), desc="Preloading CA coords", unit="protein")
        for f in it_futs:
            key, val = f.result()
            if val is not None:
                coords_map[key] = val

    return coords_map


class PreloadedStructureCache:
    def __init__(self, base: StructureCache, coords_map):
        self.base = base
        self.coords_map = coords_map
        self.failed = set()

    def get_ca_coords(self, pdb, chain, residue_nums_sorted):
        key = (pdb, chain, tuple(residue_nums_sorted))
        if key in self.coords_map:
            return self.coords_map[key]
        # fallback (and still safe)
        coords_all, kept_resnums = self.base.get_ca_coords(pdb, chain, residue_nums_sorted)
        if coords_all.shape[0] == 0:
            self.failed.add(pdb)
        elif kept_resnums is None:
            self.failed.add(pdb)
        return coords_all, kept_resnums


@dataclass(frozen=True)
class StructuralMetricConfig:
    contact_cutoff: float = 10.0
    ca_pair_sample_max: int = 4000
    eps: float = 1e-8


def pairwise_mean_distance(coords: np.ndarray, sample_max: int, rng: np.random.Generator) -> float:
    """
    Mean pairwise distance between coords.
    For small n, computes full pairwise distances.
    For large n, samples up to sample_max pairs.
    Args:
        coords: (N,3) float32
        sample_max: maximum number of pairs to sample when n is large
        rng: random generator
    Returns:
        Mean pairwise distance as float, or nan if not computable.
    """
    n = coords.shape[0]
    if n < 2:
        return float("nan")

    # full pairwise when small
    if n <= 300:
        d = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=-1))
        iu = np.triu_indices(n, k=1)
        return float(dist[iu].mean()) if iu[0].size else float("nan")

    # sample pairs when large
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


def build_contacts(coords: np.ndarray, cutoff: float) -> List[Tuple[int, int]]:
    """
    Builds contact edges between coords within cutoff.
    Args:
        coords: (N,3) float32
        cutoff: distance cutoff for contacts
    Returns:
        List[Tuple[int, int]]: list of contact edges as (i, j) tuples
    """
    n = coords.shape[0]
    if n < 2:
        return []
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff)
    return [(int(i), int(j)) for (i, j) in pairs]


def radius_of_gyration(coords: np.ndarray, eps: float) -> float:
    """
    Computes radius of gyration of coords.
    Args:
        coords: (N,3) float32
        eps: small value to avoid zero division
    Returs:
        Rg as float, or nan if not computable.
    """
    n = coords.shape[0]
    if n == 0:
        return float("nan")
    mu = coords.mean(axis=0, keepdims=True)
    d2 = np.sum((coords - mu) ** 2, axis=1)
    return float(np.sqrt(d2.mean() + eps))


# =============================================================================
# Core characterization
# =============================================================================

@dataclass(frozen=True)
class CharacterizeConfig:
    pad_label: int = -1
    random_baseline: bool = False
    random_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    max_segments_per_protein: Optional[int] = None  # safety valve for pathological labels
    size_bins: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 40, 60, 80, 120, 200, 999999)


def runs_of_equal_labels(labels: Sequence[int], pad_label: int) -> List[Tuple[int, int, int]]:
    """
    Runs as (label, start_idx, length). PAD resets run.
    Args:
        labels: sequence of integer labels
        pad_label: label to ignore / reset run
    Returns:
        List[Tuple[int, int, int]]: list of runs as (label, start_idx, length) tuples
    """
    runs: List[Tuple[int, int, int]] = []
    prev = None
    start = 0
    for i, lab in enumerate(labels):
        if lab == pad_label:
            prev = None
            continue
        if prev is None:
            prev = lab
            start = i
            continue
        if lab != prev:
            runs.append((int(prev), int(start), int(i - start)))
            prev = lab
            start = i
    if prev is not None:
        runs.append((int(prev), int(start), int(len(labels) - start)))
    return runs


def size_matched_shuffle(labels: List[int], pad_label: int, rng: np.random.Generator) -> List[int]:
    """
    Returns a size-matched random shuffle of labels, preserving counts of each label (except pad_label).
    """

    labs = np.asarray(labels, dtype=int)
    mask = labs != int(pad_label)
    vals = labs[mask].copy()
    rng.shuffle(vals)
    out = labs.copy()
    out[mask] = vals
    return out.tolist()


def segment_membership_from_labels(
    residue_nums_sorted: List[int],
    labels_sorted: List[int],
    pad_label: int,
) -> Dict[int, List[int]]:
    """
    Returns mapping: segment_label -> list of residue_numbers
    Args:
        residue_nums_sorted: list of residue numbers (sorted)
        labels_sorted: list of labels (same order as residue_nums_sorted)
        pad_label: label to ignore
    Returns:
        Dict[int, List[int]]: mapping segment_label -> list of residue_numbers
    """
    segs: Dict[int, List[int]] = {}
    for rn, lab in zip(residue_nums_sorted, labels_sorted):
        if lab == pad_label:
            continue
        segs.setdefault(int(lab), []).append(int(rn))
    return segs


def compute_protein_level_metrics(
    labels_sorted: List[int],
    pad_label: int,
) -> Dict[str, float]:
    """
    Computes protein-level metrics from residue-level labels.
    Args:
        labels_sorted: list of labels (sorted by residue number)
        pad_label: label to ignore
    Returns:
        Dict[str, float]: protein-level metrics
        n_res: total number of residues
        coverage_nonpad: fraction of residues not PAD
        unique_labels: number of unique non-PAD labels
        boundaries: number of segment boundaries (runs - 1)
        mean_run_length: mean run length of segments
        median_run_length: median run length of segments
        usage_gini: gini coefficient of label usage counts
        usage_entropy_norm: normalized entropy of label usage counts
    """
    n = len(labels_sorted)
    if n == 0:
        return {
            "n_res": 0.0,
            "coverage_nonpad": float("nan"),
            "unique_labels": float("nan"),
            "boundaries": float("nan"),
            "mean_run_length": float("nan"),
            "median_run_length": float("nan"),
            "usage_gini": float("nan"),
            "usage_entropy_norm": float("nan"),
        }

    valid = [lab for lab in labels_sorted if lab != pad_label]
    coverage = len(valid) / float(n) if n else 0.0

    if valid:
        vals, cnts = np.unique(np.asarray(valid, dtype=int), return_counts=True)
        uniq = float(len(vals))
        gini = gini_coefficient(cnts)
        ent = normalized_entropy(cnts)
    else:
        uniq = 0.0
        gini = float("nan")
        ent = float("nan")

    runs = runs_of_equal_labels(labels_sorted, pad_label=pad_label)
    run_lengths = [rlen for (_lab, _st, rlen) in runs]
    boundaries = max(0, len(runs) - 1)

    return {
        "n_res": float(n),
        "coverage_nonpad": float(coverage),
        "unique_labels": float(uniq),
        "boundaries": float(boundaries),
        "mean_run_length": float(np.mean(run_lengths)) if run_lengths else float("nan"),
        "median_run_length": float(np.median(run_lengths)) if run_lengths else float("nan"),
        "usage_gini": float(gini),
        "usage_entropy_norm": float(ent),
    }


def compute_segment_table_for_protein(
    *,
    pdb: str,
    chain: str,
    residue_nums_sorted: List[int],
    labels_sorted: List[int],
    pad_label: int,
    struct_cache: Optional[StructureCache],
    smc: StructuralMetricConfig,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Builds per-segment rows for one protein. If struct_cache is given, computes structural metrics.
    Args:
        pdb: PDB ID
        chain: chain ID
        residue_nums_sorted: list of residue numbers (sorted)
        labels_sorted: list of labels (same order as residue_nums_sorted)
        pad_label: label to ignore
        struct_cache: optional StructureCache for loading structure and coords
        smc: StructuralMetricConfig for structural metric parameters
        rng: random generator for sampling
    Returns:
        List[Dict[str, Any]]: list of per-segment metric rows
        n_res: number of residues in segment
        seq_span: span of residue numbers in segment
        seq_compactness: n_res / seq_span
        run_count: number of runs (contiguous stretches) of segment
        max_run_length: maximum run length of segment
        mean_run_length: mean run length of segment
        fragmentation_ratio: run_count / n_res
        mean_intra_ca_dist: mean pairwise Cα distance within segment
        contact_density: fraction of possible Cα contacts within segment
        cut_ratio: fraction of cut edges incident to segment
        intra_edges: number of intra-segment contact edges
        cut_edges: number of cut edges incident to segment
        rg: radius of gyration of segment Cα coords
        packing_density: n_coords_used / (rg^3)
        log_packing_density: log(n_coords_used / (rg^3))
        n_coords_used: number of Cα coords used for structural metrics
    """
    segs = segment_membership_from_labels(residue_nums_sorted, labels_sorted, pad_label=pad_label)
    out_rows: List[Dict[str, Any]] = []

    # contiguity via runs: a label can appear in multiple runs => fragmented segment
    runs = runs_of_equal_labels(labels_sorted, pad_label=pad_label)
    label_to_run_count: Dict[int, int] = {}
    label_to_run_lengths: Dict[int, List[int]] = {}
    for lab, _st, ln in runs:
        label_to_run_count[lab] = label_to_run_count.get(lab, 0) + 1
        label_to_run_lengths.setdefault(lab, []).append(int(ln))

    coords_all = None
    kept_resnums: List[int] = []
    edges: List[Tuple[int, int]] = []
    idx_map: Dict[int, int] = {}

    if struct_cache is not None:
        try: 
            coords_all, kept_resnums = struct_cache.get_ca_coords(pdb, chain, residue_nums_sorted)
            if coords_all.shape[0] >= 2:
                edges = build_contacts(coords_all, cutoff=smc.contact_cutoff)
                idx_map = {int(rn): int(i) for i, rn in enumerate(kept_resnums)}
            else:
                coords_all = None
        except Exception as e:
            # log.warning(f"Failed to compute structural metrics for {pdb} chain={chain}: {e}")
            coords_all = None

    for seg_id, members in segs.items():
        if len(members) < 2:
            continue

        members_sorted = sorted(set(int(x) for x in members))
        n_res = len(members_sorted)

        # contiguity/fragmentation metrics
        run_count = int(label_to_run_count.get(int(seg_id), 0))
        run_lengths = label_to_run_lengths.get(int(seg_id), [])
        max_run = int(max(run_lengths)) if run_lengths else 0
        mean_run = float(np.mean(run_lengths)) if run_lengths else float("nan")
        frag_ratio = float(run_count / max(1, n_res))  # tiny proxy, mostly for sanity
        # "span" in sequence space (residue numbers)
        span = int(members_sorted[-1] - members_sorted[0] + 1) if members_sorted else 0
        compactness_seq = float(n_res / max(1, span))  # 1.0 means perfectly contiguous w/o gaps

        row: Dict[str, Any] = {
            "pdb": pdb,
            "chain": chain,
            "segment_id": int(seg_id),
            "n_res": int(n_res),
            "seq_span": int(span),
            "seq_compactness": float(compactness_seq),
            "run_count": int(run_count),
            "max_run_length": int(max_run),
            "mean_run_length": float(mean_run),
            "fragmentation_ratio": float(frag_ratio),
        }

        # structural metrics (optional)
        if coords_all is not None and idx_map:
            idxs = np.asarray([idx_map[rn] for rn in members_sorted if rn in idx_map], dtype=int)
            if idxs.size >= 2:
                seg_coords = coords_all[idxs]
                mean_intra = pairwise_mean_distance(seg_coords, sample_max=smc.ca_pair_sample_max, rng=rng)

                idx_set = set(idxs.tolist())
                intra_edges = 0
                cut_edges = 0
                for i, j in edges:
                    in_i = i in idx_set
                    in_j = j in idx_set
                    if in_i and in_j:
                        intra_edges += 1
                    elif in_i != in_j:
                        cut_edges += 1

                possible_pairs = idxs.size * (idxs.size - 1) / 2
                contact_density = float(intra_edges / possible_pairs) if possible_pairs > 0 else float("nan")
                cut_ratio = float(cut_edges / max(1, (cut_edges + intra_edges)))

                rg = radius_of_gyration(seg_coords, eps=smc.eps)
                rg_clip = max(float(rg), 1.0)
                packing = float(idxs.size / (rg_clip ** 3 + smc.eps))
                log_packing = float(np.log(idxs.size / (rg_clip ** 3 + smc.eps)))

                row.update(
                    {
                        "mean_intra_ca_dist": float(mean_intra),
                        "contact_density": float(contact_density),
                        "cut_ratio": float(cut_ratio),
                        "intra_edges": int(intra_edges),
                        "cut_edges": int(cut_edges),
                        "rg": float(rg),
                        "packing_density": float(packing),
                        "log_packing_density": float(log_packing),
                        "n_coords_used": int(idxs.size),
                    }
                )
            else:
                row.update({"n_coords_used": int(idxs.size)})
        out_rows.append(row)

    return out_rows


# =============================================================================
# Summaries
# =============================================================================

def summarize_numeric(df: pd.DataFrame, col: str) -> Dict[str, float]:
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0.0, "mean": float("nan"), "median": float("nan"), "trimmed_mean": float("nan"), **quantiles(x, [0.1, 0.25, 0.75, 0.9])}
    return {
        "n": float(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "trimmed_mean": float(trimmed_mean(x, 0.1)),
        **quantiles(x, [0.10, 0.25, 0.75, 0.90]),
    }


def size_binned_summary(df: pd.DataFrame, metric_cols: List[str], bins: Sequence[int]) -> Dict[str, Any]:
    if df.empty:
        return {}
    d = df.copy()
    d["n_res_bin"] = pd.cut(d["n_res"], bins=list(bins), include_lowest=True)
    out: Dict[str, Any] = {}
    for b, g in d.groupby("n_res_bin"):
        if g.empty:
            continue
        out[str(b)] = {"n_segments": float(len(g))}
        for c in metric_cols:
            out[str(b)][c] = summarize_numeric(g, c)
    return out


def distribution_delta(dfA: pd.DataFrame, dfB: pd.DataFrame, col: str) -> Dict[str, float]:
    a = pd.to_numeric(dfA[col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(dfB[col], errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"delta_mean": float("nan"), "delta_median": float("nan")}
    return {"delta_mean": float(b.mean() - a.mean()), "delta_median": float(np.median(b) - np.median(a))}


# =============================================================================
# Plotting
# =============================================================================

def plot_hist(df: pd.DataFrame, col: str, title: str, xlabel: str, out_base: Path, logx: bool = False) -> None:
    set_pub_style()
    x = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()
    fig = plt.figure(figsize=(6.2, 3.2))
    ax = plt.gca()
    if x.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        if logx:
            x = x[x > 0]
        ax.hist(x, bins=50)
        if logx:
            ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    savefig_dual(fig, out_base)


def plot_ecdf(df: pd.DataFrame, col: str, title: str, xlabel: str, out_base: Path, logx: bool = False) -> None:
    set_pub_style()
    x = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()
    fig = plt.figure(figsize=(6.2, 3.2))
    ax = plt.gca()
    if x.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        if logx:
            x = x[x > 0]
        x = np.sort(x)
        y = np.arange(1, x.size + 1) / float(x.size)
        ax.plot(x, y)
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    savefig_dual(fig, out_base)


def plot_box_by_sizebin(
    df: pd.DataFrame,
    col: str,
    bins: Sequence[int],
    title: str,
    ylabel: str,
    out_base: Path,
) -> None:
    set_pub_style()
    d = df.copy()
    d["n_res_bin"] = pd.cut(d["n_res"], bins=list(bins), include_lowest=True)
    groups = [(str(b), g) for b, g in d.groupby("n_res_bin") if len(g) >= 10]
    fig = plt.figure(figsize=(8.0, 3.4))
    ax = plt.gca()
    if not groups:
        ax.text(0.5, 0.5, "Insufficient data per bin", ha="center", va="center")
    else:
        data = [pd.to_numeric(g[col], errors="coerce").dropna().astype(float).to_numpy() for _, g in groups]
        labels = [name for name, _ in groups]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Segment size bin (# residues)")
    savefig_dual(fig, out_base)


# =============================================================================
# Orchestration
# =============================================================================

def load_residue_assignments(cluster_dir: Path, prefix: str) -> pd.DataFrame:
    p = cluster_dir / f"{prefix}_residue_assignments.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    require_columns(df, ["pdb_id", "residue_ids", "cluster_ids"], "residue_assignments")
    return df

def _stable_seed_from_str(s: str) -> int:
    # deterministic across runs (unlike Python's built-in hash, which is salted)
    # 32-bit seed is enough for numpy RNG
    return int(np.uint32(np.frombuffer(s.encode("utf-8"), dtype=np.uint8).sum()))


def _characterize_one(
    r: Any,
    *,
    cfg: CharacterizeConfig,
    struct_cache: Optional[StructureCache],
    smc: StructuralMetricConfig,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Worker for a single row. Returns (protein_row or None, segment_rows).
    """
    try: 
        pdb_id = str(getattr(r, "pdb_id"))
        pdb, chain = split_id(pdb_id)

        residue_ids = parse_csv_str_list(getattr(r, "residue_ids"))
        labels = parse_csv_int_list(getattr(r, "cluster_ids"))
        if not residue_ids or not labels:
            return None, []

        # sort by residue number for sequence-based metrics
        # (avoid building big intermediate lists more than needed)
        keys = [residue_number_key(x) for x in residue_ids]
        order = np.argsort(keys)

        L = min(len(order), len(labels))
        idx = order[:L]

        residue_ids_sorted = [residue_ids[i] for i in idx]
        labels_sorted = [labels[i] for i in idx]

        residue_nums_sorted = [
            residue_chain_and_num(x, default_chain=chain)[1] for x in residue_ids_sorted
        ]

        # protein-level metrics
        pm = compute_protein_level_metrics(labels_sorted, pad_label=cfg.pad_label)
        pm.update({"pdb": pdb, "chain": chain, "pdb_id": pdb_id})

        # deterministic *per protein* sampling
        rng = np.random.default_rng(_stable_seed_from_str(pdb_id))

        seg_rows = compute_segment_table_for_protein(
            pdb=pdb,
            chain=chain,
            residue_nums_sorted=residue_nums_sorted,
            labels_sorted=labels_sorted,
            pad_label=cfg.pad_label,
            struct_cache=struct_cache,
            smc=smc,
            rng=rng,
        )
        
        return pm, seg_rows
    except Exception as e:
        log.warning(f"Failed to characterize protein {getattr(r, 'pdb_id', 'unknown')}: {e}")  
        return None, []

def _rand_worker(
    it: ProteinStatic,
    *,
    cfg: CharacterizeConfig,
    smc: StructuralMetricConfig,
    struct_cache: Optional[StructureCache],
    seed: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:

    # deterministic per protein+seed
    # (cheap stable hash; avoids python's salted hash)
    stable = (np.uint32(seed) * np.uint32(2654435761) + np.uint32(sum(it.pdb_id.encode("utf-8"))))
    rng = np.random.default_rng(int(stable))

    labels_shuf = size_matched_shuffle(it.labels_sorted, pad_label=cfg.pad_label, rng=rng)

    pm = compute_protein_level_metrics(labels_shuf, pad_label=cfg.pad_label)
    pm.update({"pdb": it.pdb, "chain": it.chain, "pdb_id": it.pdb_id, "seed": int(seed)})

    try:
        seg_rows = compute_segment_table_for_protein(
            pdb=it.pdb,
            chain=it.chain,
            residue_nums_sorted=it.residue_nums_sorted,
            labels_sorted=labels_shuf,
            pad_label=cfg.pad_label,
            struct_cache=struct_cache,
            smc=smc,
            rng=rng,
        )
    except Exception:
        # bad/empty structure: skip structural rows for this protein
        seg_rows = []

    return pm, seg_rows


def characterize(
    *,
    res_df: pd.DataFrame,
    cfg: CharacterizeConfig,
    struct_cache: Optional[StructureCache],
    smc: StructuralMetricConfig,
    max_workers: Optional[int] = None,
    chunksize: int = 256,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel per-protein characterization.

    Returns:
      proteins_df: per-protein metrics
      segments_df: per-segment metrics (one row per protein-segment label)
    """
    # faster than iterrows: namedtuples without index
    rows = list(res_df.itertuples(index=False, name="ResRow"))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 8) * 4)

    prot_rows: List[Dict[str, Any]] = []
    seg_rows_all: List[Dict[str, Any]] = []

    total = len(rows)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _characterize_one,
                r,
                cfg=cfg,
                struct_cache=struct_cache,
                smc=smc,
            )
            for r in rows
        ]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total,
                desc="Characterizing proteins",
                unit="protein",
            )

        for fut in iterator:
            pm, seg_rows = fut.result()
            if pm is not None:
                prot_rows.append(pm)
            if seg_rows:
                seg_rows_all.extend(seg_rows)

    log.info(f"Characterized {len(prot_rows)} proteins, {len(seg_rows_all)} segments")
    log.info(f"    Failed loads: {len(struct_cache.failed) if struct_cache else 0}")

    proteins_df = pd.DataFrame(prot_rows)
    segments_df = pd.DataFrame(seg_rows_all)
    return proteins_df, segments_df

# def characterize(
#     *,
#     res_df: pd.DataFrame,
#     cfg: CharacterizeConfig,
#     struct_cache: Optional[StructureCache],
#     smc: StructuralMetricConfig,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Returns:
#       proteins_df: per-protein metrics
#       segments_df: per-segment metrics (one row per protein-segment label)
#     """
#     prot_rows: List[Dict[str, Any]] = []
#     seg_rows: List[Dict[str, Any]] = []

#     for _, r in res_df.iterrows():
#         pdb_id = str(r["pdb_id"])
#         pdb, chain = split_id(pdb_id)

#         residue_ids = parse_csv_str_list(r["residue_ids"])
#         labels = parse_csv_int_list(r["cluster_ids"])
#         if not residue_ids or not labels:
#             continue

#         # sort by residue number for sequence-based metrics
#         order = np.argsort([residue_number_key(x) for x in residue_ids])
#         L = min(len(order), len(labels))
#         residue_ids = [residue_ids[i] for i in order[:L]]
#         labels_sorted = [labels[i] for i in order[:L]]

#         residue_nums_sorted = [residue_chain_and_num(x, default_chain=chain)[1] for x in residue_ids]

#         # protein-level metrics
#         pm = compute_protein_level_metrics(labels_sorted, pad_label=cfg.pad_label)
#         pm.update({"pdb": pdb, "chain": chain, "pdb_id": pdb_id})
#         prot_rows.append(pm)

#         # segment-level metrics
#         # (use a fixed seed per protein for deterministic pair sampling)
#         rng = np.random.default_rng(0)
#         seg_rows.extend(
#             compute_segment_table_for_protein(
#                 pdb=pdb,
#                 chain=chain,
#                 residue_nums_sorted=residue_nums_sorted,
#                 labels_sorted=labels_sorted,
#                 pad_label=cfg.pad_label,
#                 struct_cache=struct_cache,
#                 smc=smc,
#                 rng=rng,
#             )
#         )

#     proteins_df = pd.DataFrame(prot_rows)
#     segments_df = pd.DataFrame(seg_rows)
#     return proteins_df, segments_df


# def characterize_with_random_baseline(
#     *,
#     res_df: pd.DataFrame,
#     cfg: CharacterizeConfig,
#     struct_cache: Optional[StructureCache],
#     smc: StructuralMetricConfig,
# ) -> Dict[str, pd.DataFrame]:
#     """
#     Returns dict with:
#       - proteins_model, segments_model
#       - proteins_random, segments_random   (concatenated over seeds)
#     """
#     proteins_model, segments_model = characterize(res_df=res_df, cfg=cfg, struct_cache=struct_cache, smc=smc, show_progress=True)

#     if not cfg.random_baseline:
#         return {
#             "proteins_model": proteins_model,
#             "segments_model": segments_model,
#         }

#     prot_rows_r: List[pd.DataFrame] = []
#     seg_rows_r: List[pd.DataFrame] = []

#     for seed in cfg.random_seeds:
#         prot_rows: List[Dict[str, Any]] = []
#         seg_rows: List[Dict[str, Any]] = []
#         rng = np.random.default_rng(int(seed))

#         for _, r in res_df.iterrows():
#             pdb_id = str(r["pdb_id"])
#             pdb, chain = split_id(pdb_id)

#             residue_ids = parse_csv_str_list(r["residue_ids"])
#             labels = parse_csv_int_list(r["cluster_ids"])
#             if not residue_ids or not labels:
#                 continue

#             order = np.argsort([residue_number_key(x) for x in residue_ids])
#             L = min(len(order), len(labels))
#             residue_ids = [residue_ids[i] for i in order[:L]]
#             labels_sorted = [labels[i] for i in order[:L]]

#             # shuffle labels preserving counts (excluding PAD)
#             labels_shuf = size_matched_shuffle(labels_sorted, pad_label=cfg.pad_label, rng=rng)

#             residue_nums_sorted = [residue_chain_and_num(x, default_chain=chain)[1] for x in residue_ids]

#             pm = compute_protein_level_metrics(labels_shuf, pad_label=cfg.pad_label)
#             pm.update({"pdb": pdb, "chain": chain, "pdb_id": pdb_id, "seed": int(seed)})
#             prot_rows.append(pm)

#             seg_rows.extend(
#                 compute_segment_table_for_protein(
#                     pdb=pdb,
#                     chain=chain,
#                     residue_nums_sorted=residue_nums_sorted,
#                     labels_sorted=labels_shuf,
#                     pad_label=cfg.pad_label,
#                     struct_cache=struct_cache,
#                     smc=smc,
#                     rng=rng,
#                 )
#             )

#         dfp = pd.DataFrame(prot_rows)
#         dfs = pd.DataFrame(seg_rows)
#         if not dfp.empty:
#             dfp["baseline"] = "random_size_matched"
#         if not dfs.empty:
#             dfs["baseline"] = "random_size_matched"
#             dfs["seed"] = int(seed)
#         prot_rows_r.append(dfp)
#         seg_rows_r.append(dfs)

#     proteins_random = pd.concat(prot_rows_r, axis=0, ignore_index=True) if prot_rows_r else pd.DataFrame()
#     segments_random = pd.concat(seg_rows_r, axis=0, ignore_index=True) if seg_rows_r else pd.DataFrame()

#     if not proteins_model.empty:
#         proteins_model["baseline"] = "model"
#     if not segments_model.empty:
#         segments_model["baseline"] = "model"

#     return {
#         "proteins_model": proteins_model,
#         "segments_model": segments_model,
#         "proteins_random": proteins_random,
#         "segments_random": segments_random,
#     }

def characterize_with_random_baseline(
    *,
    res_df: pd.DataFrame,
    cfg: CharacterizeConfig,
    struct_cache: Optional[StructureCache],
    smc: StructuralMetricConfig,
) -> Dict[str, pd.DataFrame]:
    log.info(f"Starting characterization with random baseline: random_baseline={cfg.random_baseline}, random_seeds={cfg.random_seeds}")
    
    if res_df.shape[0] > 5000:
        log.info(f"Subsampling to 5000 proteins for characterization (from {res_df.shape[0]})")
        res_df = subsample_df(res_df, max_n=5000, seed=0)

    log.info("Preparing static per-protein info")
    # Precompute static per-protein info once
    items = _prepare_static(res_df)

    # OPTIONAL: preload CA coords once to amortize across seeds
    # (huge speedup if random_seeds > 1)
    # If your StructureCache is not thread-safe, preloading also reduces contention.
    log.info("Preloading CA coordinates for structural metrics")
    coords_map = _preload_coords(items, struct_cache, show_progress=True) if struct_cache is not None else None
    if coords_map is not None:
        struct_cache_eff: Optional[StructureCache] = PreloadedStructureCache(struct_cache, coords_map)
    else:
        struct_cache_eff = struct_cache
        log.info("Using original structure cache")
 

    proteins_model, segments_model = characterize(
        res_df=res_df, cfg=cfg, struct_cache=struct_cache_eff, smc=smc, show_progress=True
    )

    if not cfg.random_baseline:
        return {"proteins_model": proteins_model, "segments_model": segments_model}


    max_workers = min(32, (os.cpu_count() or 8) * 4)

    prot_rows_r: List[pd.DataFrame] = []
    seg_rows_r: List[pd.DataFrame] = []

    for seed in cfg.random_seeds:
        prot_rows: List[Dict[str, Any]] = []
        seg_rows: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    _rand_worker,
                    it,
                    cfg=cfg,
                    smc=smc,
                    struct_cache=struct_cache_eff,
                    seed=int(seed),
                )
                for it in items
            ]

            it_futs = tqdm(
                as_completed(futs),
                total=len(futs),
                desc=f"Random baseline (seed={seed})",
                unit="protein",
            )

            for f in it_futs:
                pm, segs = f.result()
                if pm is not None:
                    prot_rows.append(pm)
                if segs:
                    seg_rows.extend(segs)

        dfp = pd.DataFrame(prot_rows)
        dfs = pd.DataFrame(seg_rows)
        if not dfp.empty:
            dfp["baseline"] = "random_size_matched"
        if not dfs.empty:
            dfs["baseline"] = "random_size_matched"
            dfs["seed"] = int(seed)

        prot_rows_r.append(dfp)
        seg_rows_r.append(dfs)

    proteins_random = pd.concat(prot_rows_r, axis=0, ignore_index=True) if prot_rows_r else pd.DataFrame()
    segments_random = pd.concat(seg_rows_r, axis=0, ignore_index=True) if seg_rows_r else pd.DataFrame()

    if not proteins_model.empty:
        proteins_model["baseline"] = "model"
    if not segments_model.empty:
        segments_model["baseline"] = "model"

    return {
        "proteins_model": proteins_model,
        "segments_model": segments_model,
        "proteins_random": proteins_random,
        "segments_random": segments_random,
    }



def build_summary(
    proteins_model: pd.DataFrame,
    segments_model: pd.DataFrame,
    *,
    proteins_random: Optional[pd.DataFrame] = None,
    segments_random: Optional[pd.DataFrame] = None,
    cfg: CharacterizeConfig,
    has_structure: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_proteins": float(len(proteins_model)) if proteins_model is not None else 0.0,
        "n_segments": float(len(segments_model)) if segments_model is not None else 0.0,
        "has_structure_metrics": bool(has_structure),
        "config": {
            "pad_label": int(cfg.pad_label),
            "random_baseline": bool(cfg.random_baseline),
            "random_seeds": list(cfg.random_seeds),
            "size_bins": list(cfg.size_bins),
        },
        "protein_level": {},
        "segment_level": {},
        "size_binned": {},
        "random_baseline": {},
    }

    # protein-level summaries
    prot_cols = ["n_res", "coverage_nonpad", "unique_labels", "boundaries", "mean_run_length", "usage_gini", "usage_entropy_norm"]
    for c in prot_cols:
        if c in proteins_model.columns:
            summary["protein_level"][c] = summarize_numeric(proteins_model, c)

    # segment-level summaries
    seg_cols = ["n_res", "seq_span", "seq_compactness", "run_count", "max_run_length", "fragmentation_ratio"]
    if has_structure:
        seg_cols += ["mean_intra_ca_dist", "contact_density", "cut_ratio", "rg", "packing_density", "log_packing_density"]

    for c in seg_cols:
        if c in segments_model.columns:
            summary["segment_level"][c] = summarize_numeric(segments_model, c)

    # size-binned (segments)
    metric_cols = [c for c in seg_cols if c != "n_res"]
    summary["size_binned"]["model"] = size_binned_summary(segments_model, metric_cols=metric_cols, bins=cfg.size_bins)

    # random baseline comparison
    if cfg.random_baseline and segments_random is not None and not segments_random.empty:
        summary["size_binned"]["random_size_matched"] = size_binned_summary(segments_random, metric_cols=metric_cols, bins=cfg.size_bins)

        deltas: Dict[str, Any] = {}
        for c in metric_cols:
            if c in segments_model.columns and c in segments_random.columns:
                deltas[c] = distribution_delta(segments_random, segments_model, c)  # (random -> model) means "model - random"
        summary["random_baseline"]["model_minus_random"] = deltas

    return summary


def write_outputs(
    out_dir: Path,
    prefix: str,
    proteins_model: pd.DataFrame,
    segments_model: pd.DataFrame,
    summary: Dict[str, Any],
    *,
    proteins_random: Optional[pd.DataFrame] = None,
    segments_random: Optional[pd.DataFrame] = None,
    cfg: CharacterizeConfig,
    has_structure: bool,
) -> None:
    char_dir = ensure_dir(out_dir)
    plot_dir = ensure_dir(char_dir / "plots")

    # tables
    proteins_model.to_csv(char_dir / f"{prefix}_proteins.csv", index=False)
    segments_model.to_csv(char_dir / f"{prefix}_segments.csv", index=False)
    log.info(f"Wrote: {char_dir / f'{prefix}_proteins.csv'}")
    log.info(f"Wrote: {char_dir / f'{prefix}_segments.csv'}")

    if cfg.random_baseline and proteins_random is not None and segments_random is not None:
        proteins_random.to_csv(char_dir / f"{prefix}_proteins_random.csv", index=False)
        segments_random.to_csv(char_dir / f"{prefix}_segments_random.csv", index=False)
        log.info(f"Wrote: {char_dir / f'{prefix}_proteins_random.csv'}")
        log.info(f"Wrote: {char_dir / f'{prefix}_segments_random.csv'}")

    # summary json
    (char_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote: {char_dir / f'{prefix}_summary.json'}")

    # plots
    # sizes
    plot_hist(segments_model, "n_res", "Segment size distribution", "Segment size (# residues)", plot_dir / f"{prefix}_hist_segment_size", logx=False)
    plot_ecdf(segments_model, "n_res", "Segment size ECDF", "Segment size (# residues)", plot_dir / f"{prefix}_ecdf_segment_size", logx=False)

    # contiguity
    if "seq_compactness" in segments_model.columns:
        plot_hist(segments_model, "seq_compactness", "Sequence compactness (n_res/span)", "n_res / span", plot_dir / f"{prefix}_hist_seq_compactness")

    if has_structure:
        # structural distributions
        for col, title, xlabel, logx in [
            ("mean_intra_ca_dist", "Mean intra-segment Cα distance", "Å", False),
            ("contact_density", "Intra-segment contact density", "contacts / possible pairs", False),
            ("cut_ratio", "Cut ratio (inter / (inter+intra))", "ratio", False),
            ("rg", "Radius of gyration (Rg)", "Å", False),
            ("packing_density", "Packing density (N / Rg^3)", "1/Å^3 (proxy)", True),
        ]:
            if col in segments_model.columns:
                plot_hist(segments_model, col, title, xlabel, plot_dir / f"{prefix}_hist_{col}", logx=logx)
                plot_ecdf(segments_model, col, f"{title} ECDF", xlabel, plot_dir / f"{prefix}_ecdf_{col}", logx=logx)

        # size-binned boxplots for key structural metrics
        for col, title, ylabel in [
            ("mean_intra_ca_dist", "Mean intra-segment Cα distance by size bin", "Å"),
            ("contact_density", "Contact density by size bin", "contacts / possible pairs"),
            ("cut_ratio", "Cut ratio by size bin", "ratio"),
            ("log_packing_density", "Log packing density by size bin", "log(N / Rg^3)"),
        ]:
            if col in segments_model.columns:
                plot_box_by_sizebin(
                    segments_model,
                    col=col,
                    bins=cfg.size_bins,
                    title=title,
                    ylabel=ylabel,
                    out_base=plot_dir / f"{prefix}_box_{col}_by_sizebin",
                )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)

    ap.add_argument("--pad_label", type=int, default=-1)

    # structural
    ap.add_argument("--structure_dir", type=str, default=None)
    ap.add_argument("--structure_fmt", type=str, default="auto", choices=["auto", "pdb", "cif"])
    ap.add_argument("--prefer_cif", action="store_true", default=False)
    ap.add_argument("--contact_cutoff", type=float, default=8.0)

    # baseline
    ap.add_argument("--random_baseline", action="store_true", default=False)
    ap.add_argument("--random_seeds", type=str, default="0,1,2,3,4")

    return ap.parse_args()


def main() -> None:
    a = parse_args()

    cluster_dir = Path(a.cluster_dir)
    out_dir = ensure_dir(Path(a.output_dir))
    prefix = str(a.prefix)

    log.info(f"Characterizing residue assignments in: {cluster_dir}")
    res_df = load_residue_assignments(cluster_dir, prefix)

    log.info("Configuration:")
    log.info(f"  pad_label = {a.pad_label}")
    log.info(f"  random_baseline = {a.random_baseline}")
    log.info(f"  random_seeds = {a.random_seeds}")
    log.info(f"  structure_dir = {a.structure_dir}")
    log.info(f"  structure_fmt = {a.structure_fmt}")
    log.info(f"  prefer_cif = {a.prefer_cif}")
    log.info(f"  contact_cutoff = {a.contact_cutoff}")

    cfg = CharacterizeConfig(
        pad_label=int(a.pad_label),
        random_baseline=bool(a.random_baseline),
        random_seeds=tuple(int(x.strip()) for x in str(a.random_seeds).split(",") if x.strip() != ""),
    )

    struct_cache: Optional[StructureCache] = None
    has_structure = False
    smc = StructuralMetricConfig(contact_cutoff=float(a.contact_cutoff))

    if a.structure_dir is not None:
        try:
            log.info("Initializing structure cache...")
            struct_cache = StructureCache(
                StructureConfig(
                    structure_dir=Path(a.structure_dir),
                    fmt=str(a.structure_fmt),
                    prefer_cif=bool(a.prefer_cif),
                )
            )
            has_structure = True
            log.info(f"Structural metrics enabled. structure_dir={a.structure_dir}")
        except Exception as e:
            log.warning(f"Could not enable structural metrics: {e}. Continuing without structure.")
            struct_cache = None
            has_structure = False

    log.info("Characterizing segments...")
    out = characterize_with_random_baseline(res_df=res_df, cfg=cfg, struct_cache=struct_cache, smc=smc)
    proteins_model = out["proteins_model"]
    segments_model = out["segments_model"]
    proteins_random = out.get("proteins_random")
    segments_random = out.get("segments_random")

    log.info("Building summary...")
    summary = build_summary(
        proteins_model=proteins_model,
        segments_model=segments_model,
        proteins_random=proteins_random,
        segments_random=segments_random,
        cfg=cfg,
        has_structure=has_structure,
    )
    log.info("Writing outputs...")
    write_outputs(
        out_dir=out_dir,
        prefix=prefix,
        proteins_model=proteins_model,
        segments_model=segments_model,
        proteins_random=proteins_random,
        segments_random=segments_random,
        summary=summary,
        cfg=cfg,
        has_structure=has_structure,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
