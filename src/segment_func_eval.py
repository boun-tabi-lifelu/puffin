#!/usr/bin/env python3
"""
segment_func_reports_knn.py
---------------------------

Standalone: Unit-level kNN neighborhood GO enrichment on segment embeddings
(no prototypes, no clustering).

Do segments exhibit functional coherence beyond discrete prototype assignments?

Method:
- Each segment has an embedding.
- For each queried segment, find k nearest neighbor segments (cosine).
- Collect neighbor proteins, deduplicate.
- Evaluate functional coherence in two ways:
  (1) Neighborhood overlap: fraction of neighbor proteins sharing ANY GO term with the query protein.
  (2) Neighborhood enrichment: Fisher enrichment for the query protein's GO terms among neighbor proteins
      vs background proteins.

Controls:
- shuffled_embeddings: shuffle embedding rows, compute kNN in shuffled space (breaks alignment)
- random_neighbors: random neighbor segments (chance baseline)

Inputs:
- segments_root/<model>/<split>/
    <split>_segment_embeddings.npy OR <split>_segment_embeddings.pt (dict with key "embeddings")
    <split>_segment_metadata.csv  (must include: pdb_id, n_residues_assigned; ideally global_seg_index)
- annotation_dir/nrPDB-GO_annot.tsv (CAFA-style) -> we auto-detect skiprows (12/14/13)

Outputs:
- segment_func_reports/<model>/<split>/
    unit_knn_go_enrichment_<split>.json
    unit_knn_go_enrichment_<split>_per_query.csv
    unit_knn_go_enrichment_<split>_summary.csv   (one row per tag: true + controls)

Example:
  python segment_func_reports_knn.py \
    --segments_root segments \
    --model_name puffin_K64 \
    --split test \
    --annotation_dir data/go \
    --go_aspect MF \
    --out_root segment_func_reports \
    --device cpu \
    --k_neighbors 50 \
    --n_queries 5000 \
    --min_assigned 5

Notes:
- meta["protein_key"] is derived from pdb_id -> legacy_id (PDB-CHAIN).
- exclude_same_protein defaults to True (recommended).
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.neighbors import NearestNeighbors

import tqdm
import faiss
import matplotlib.pyplot as plt
import torch


# -------------------------
# small IO utils
# -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}. Found: {list(df.columns)}")


# -------------------------
# GO file loading (same idea as your pipeline)
# -------------------------
def load_go_annotations(annotation_dir: Path, go_file: str = "nrPDB-GO_annot.tsv") -> pd.DataFrame:
    p = annotation_dir / go_file
    if not p.exists():
        raise FileNotFoundError(f"GO annotation file not found: {p}")

    for skip in (12, 14, 13):
        try:
            df = pd.read_csv(p, sep="\t", skiprows=skip)
            if df.shape[1] >= 4:
                df = df.iloc[:, :4].copy()
                df.columns = ["PDB", "MF", "BP", "CC"]
                df["PDB"] = df["PDB"].astype(str)
                return df.set_index("PDB")
        except Exception:
            pass
    raise ValueError(f"Could not parse GO TSV with common skiprows values: {p}")


def safe_go_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    return [t for t in s.split(",") if t]


# -------------------------
# PDB id helpers -> legacy_id (PDB-CHAIN)
# -------------------------
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


def legacy_id_from_new(pdb_id: str) -> str:
    pdb, chain = split_id(pdb_id)
    return f"{pdb}-{chain}"


def add_id_columns(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    require_columns(m, ["pdb_id"], "segment_metadata")
    m["legacy_id"] = m["pdb_id"].astype(str).apply(legacy_id_from_new)
    m["protein_key"] = m["legacy_id"].astype(str)
    return m


# -------------------------
# Embeddings loading
# -------------------------
def load_segment_outputs(
    split_dir: Path,
    split: str,
    *,
    as_torch: bool = False,
    device: str = "cpu",
) -> Tuple[Union[np.ndarray, "torch.Tensor"], pd.DataFrame]:
    meta_path = split_dir / f"{split}_segment_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing segment metadata: {meta_path}")
    meta = pd.read_csv(meta_path)

    # minimal required columns; we don't force global_seg_index here, but keep if present
    require_columns(meta, ["pdb_id", "n_residues_assigned"], "segment_metadata")

    npy_path = split_dir / f"{split}_segment_embeddings.npy"
    pt_path = split_dir / f"{split}_segment_embeddings.pt"

    if as_torch:
        if torch is None:
            raise ImportError("torch is required for as_torch=True")

        if npy_path.exists():
            E = np.load(npy_path).astype("float32")
            E_t = torch.from_numpy(E).to(device=device, non_blocking=True)
        elif pt_path.exists():
            obj: Any = torch.load(pt_path, map_location="cpu")
            if not (isinstance(obj, dict) and "embeddings" in obj):
                raise ValueError(f"{pt_path} must be a dict with key 'embeddings'")
            E_t = obj["embeddings"].detach().to(device=device, non_blocking=True).float()
        else:
            raise FileNotFoundError(f"Missing embeddings file: {npy_path} or {pt_path}")

        if len(meta) != int(E_t.shape[0]):
            raise ValueError(f"Mismatch: meta rows={len(meta)} vs embeddings rows={int(E_t.shape[0])}")

        return E_t, meta

    # numpy path
    if npy_path.exists():
        E = np.load(npy_path).astype("float32")
    elif pt_path.exists():
        if torch is None:
            raise ImportError("torch required to load .pt embeddings")
        obj = torch.load(pt_path, map_location="cpu")
        if not (isinstance(obj, dict) and "embeddings" in obj):
            raise ValueError(f"{pt_path} must be a dict with key 'embeddings'")
        E = obj["embeddings"].detach().cpu().numpy().astype("float32")
    else:
        raise FileNotFoundError(f"Missing embeddings file: {npy_path} or {pt_path}")

    if len(meta) != E.shape[0]:
        raise ValueError(f"Mismatch: meta rows={len(meta)} vs embeddings rows={E.shape[0]}")
    return E, meta


# -------------------------
# Filtering
# -------------------------
def filter_segments_by_assigned(
    E: np.ndarray,
    meta: pd.DataFrame,
    *,
    min_assigned: int,
    max_assigned: Optional[int],
) -> Tuple[np.ndarray, pd.DataFrame]:
    n_assigned = pd.to_numeric(meta["n_residues_assigned"], errors="coerce").fillna(0).astype(int)
    mask = n_assigned >= int(min_assigned)
    if max_assigned is not None:
        mask &= n_assigned <= int(max_assigned)
    kept = np.where(mask.to_numpy())[0]
    if kept.size == 0:
        raise ValueError(f"No segments kept after filter min_assigned={min_assigned}, max_assigned={max_assigned}")
    return E[kept], meta.iloc[kept].copy().reset_index(drop=True)


# -------------------------
# Math helpers
# -------------------------
def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)

def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini for nonnegative array x.
    Returns 0 if all zeros or length<2.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x[x >= 0]
    if x.size < 2:
        return float("nan")
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    # classic Gini formula
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * (idx * xs).sum()) / (n * s) - (n + 1.0) / n
    return float(g)

def top_frac_share(x: np.ndarray, frac: float = 0.01) -> float:
    """
    Share of mass in top frac items of x (nonnegative).
    """
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    if s <= 0:
        return 0.0
    n = x.size
    m = max(1, int(math.ceil(frac * n)))
    xs = np.sort(x)[::-1]
    return float(xs[:m].sum() / s)

def neff_from_counts(counts: np.ndarray) -> float:
    """
    Effective number of categories from counts using Simpson inverse.
    """
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return float("nan")
    p = counts / tot
    return float(1.0 / np.maximum((p * p).sum(), 1e-12))

def jaccard_set(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union > 0 else 0.0

def compute_reuse_gap(
    query_proteins: np.ndarray,
    query_neighbor_protein_sets: List[set],
    prot_go_all: Dict[str, set],
    *,
    max_pairs: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Sample query pairs and compute Jaccard overlap of neighbor protein sets.
    Split pairs into sharedGO vs noSharedGO based on query proteins' GO sets.
    """
    rng = np.random.default_rng(seed)
    n = len(query_neighbor_protein_sets)
    if n < 2:
        return {
            "reuse_pairs_sampled": 0,
            "reuse_shared_go_mean": float("nan"),
            "reuse_noshared_go_mean": float("nan"),
            "reuse_gap_mean": float("nan"),
            "reuse_shared_go_median": float("nan"),
            "reuse_noshared_go_median": float("nan"),
            "reuse_shared_go_n": 0,
            "reuse_noshared_go_n": 0,
        }

    # sample pairs without building all nC2 when n is large
    # strategy: random pairs
    pairs = set()
    target = min(int(max_pairs), n * (n - 1) // 2)
    while len(pairs) < target:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))

    shared_vals = []
    noshared_vals = []

    for i, j in pairs:
        pi = str(query_proteins[i])
        pj = str(query_proteins[j])
        gi = prot_go_all.get(pi, set())
        gj = prot_go_all.get(pj, set())
        shared = (len(gi & gj) > 0)
        jac = jaccard_set(query_neighbor_protein_sets[i], query_neighbor_protein_sets[j])
        if shared:
            shared_vals.append(jac)
        else:
            noshared_vals.append(jac)

    shared_vals = np.asarray(shared_vals, dtype=np.float64)
    noshared_vals = np.asarray(noshared_vals, dtype=np.float64)

    return {
        "reuse_pairs_sampled": int(len(pairs)),
        "reuse_shared_go_mean": float(shared_vals.mean()) if shared_vals.size else float("nan"),
        "reuse_noshared_go_mean": float(noshared_vals.mean()) if noshared_vals.size else float("nan"),
        "reuse_gap_mean": float(shared_vals.mean() - noshared_vals.mean())
            if (shared_vals.size and noshared_vals.size) else float("nan"),
        "reuse_shared_go_median": float(np.median(shared_vals)) if shared_vals.size else float("nan"),
        "reuse_noshared_go_median": float(np.median(noshared_vals)) if noshared_vals.size else float("nan"),
        "reuse_shared_go_n": int(shared_vals.size),
        "reuse_noshared_go_n": int(noshared_vals.size),
    }


def _build_protein_go_map(go_df: pd.DataFrame, go_aspect: str) -> Dict[str, set]:
    go_aspect = go_aspect.upper()
    go_col = go_df[go_aspect]
    prot_go: Dict[str, set] = {}
    for pid in go_col.index.astype(str):
        prot_go[str(pid)] = set(safe_go_list(go_col.loc[pid]))
    return prot_go


# -------------------------
# Neighbor builders
# -------------------------
def neighbors_cosine(X: np.ndarray, k: int, *, exclude_self: bool = True) -> np.ndarray:
    """
    X: (N,H), L2-normalized recommended.
    Returns idx: (N,k), neighbor indices.
    """
    n_neighbors = k + (1 if exclude_self else 0)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="auto")
    nn.fit(X)
    _, idx = nn.kneighbors(X, return_distance=True)
    if exclude_self:
        idx = idx[:, 1:]
    return idx[:, :k].astype(np.int64)


def neighbors_random(N: int, k: int, *, seed: int, exclude_self: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.empty((N, k), dtype=np.int64)
    for i in range(N):
        if exclude_self:
            pool = np.concatenate([np.arange(0, i, dtype=np.int64), np.arange(i + 1, N, dtype=np.int64)])
        else:
            pool = np.arange(N, dtype=np.int64)
        replace = pool.size <= k
        idx[i] = rng.choice(pool, size=k, replace=replace)
    return idx


def neighbors_shuffled_embeddings(X: np.ndarray, k: int, *, seed: int) -> np.ndarray:
    """
    Shuffle X rows (breaking alignment), compute kNN in shuffled space,
    map neighbor indices back to original index space.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    Xp = X[perm]
    idx_p = neighbors_cosine(Xp, k=k, exclude_self=True)
    idx_orig = perm[idx_p]
    return idx_orig


import math
from collections import Counter

def go_entropy_for_neighborhood(
    neighbor_proteins: list[str],
    prot_go_map: dict[str, set],
    *,
    eps: float = 1e-12,
    normalize: bool = True,
):
    """
    Compute GO entropy for a neighborhood.
    """
    terms = []
    for p in neighbor_proteins:
        terms.extend(prot_go_map.get(p, []))

    if len(terms) == 0:
        return float("nan")

    cnt = Counter(terms)
    total = sum(cnt.values())
    probs = [v / total for v in cnt.values()]

    H = -sum(p * math.log(p + eps) for p in probs)

    if normalize:
        H /= math.log(len(cnt) + eps)

    return float(H)

# -------------------------
# Core evaluation
# -------------------------
def eval_unit_knn_go(
    X: np.ndarray,
    meta: pd.DataFrame,
    go_df: pd.DataFrame,
    *,
    go_aspect: str,
    k_neighbors: int,
    seed: int,
    n_queries: int,
    exclude_same_protein: bool,
    neighbor_idx: np.ndarray,
    tag: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Extended evaluation:
    - existing: shared_go_frac_neighbors, hit_any_shared_go, fisher enrichment for query terms
    - new (compositionality / collapse diagnostics):
        * unique_neighbor_proteins (protein-dedup count)
        * neff_neighbor_proteins (Simpson inverse based on segment multiplicity per protein)
        * top1_protein_share, top3_protein_share
        * reuse gap: mean(J|sharedGO) - mean(J|noSharedGO) on sampled query pairs
        * hubness: neighbor appearance distribution across proteins (Gini, top1% share, max/median)
    Returns:
      rep: summary dict (includes rep["_hubness_counts"] = dict[protein->count])
      dfq: per-query dataframe (extended columns)
    """
    go_aspect = go_aspect.upper()
    prot_go_all = _build_protein_go_map(go_df, go_aspect=go_aspect)

    background_proteins = sorted(set(meta["protein_key"].astype(str).unique().tolist()))
    bg_set = set(background_proteins)

    # term->proteins index (within background)
    term_to_prots: Dict[str, set] = defaultdict(set)
    for p in background_proteins:
        for t in prot_go_all.get(p, set()):
            term_to_prots[t].add(p)

    rng = np.random.default_rng(seed)
    N = int(X.shape[0])
    qN = min(int(n_queries), N)
    q_idx = rng.choice(np.arange(N), size=qN, replace=False)

    rows: List[Dict[str, Any]] = []

    # For reuse-gap + hubness
    query_neighbor_sets: List[set] = []
    query_proteins: List[str] = []
    neighbor_appearance = defaultdict(int)  # protein -> count appearances across query neighbor sets

    for qi in q_idx:
        q_prot = str(meta.iloc[qi]["protein_key"])
        q_terms = prot_go_all.get(q_prot, set())

        # neighbor segments -> neighbor proteins (with multiplicity)
        nseg = neighbor_idx[qi].tolist()
        nprot_all = [str(meta.iloc[j]["protein_key"]) for j in nseg]

        if exclude_same_protein:
            nprot_all = [p for p in nprot_all if p != q_prot]

        # restrict to proteins present in this split background
        nprot_all = [p for p in nprot_all if p in bg_set]

        # protein-dedup set (for coherence + reuse)
        nprot_set = set(nprot_all)

        go_H = go_entropy_for_neighborhood(
            neighbor_proteins=list(nprot_set),
            prot_go_map=prot_go_all,
            normalize=True,
        )

        m = int(len(nprot_set))

        # segment-multiplicity stats by protein (dominance / collapse)
        if len(nprot_all) > 0:
            uniq, cnt = np.unique(np.asarray(nprot_all, dtype=object), return_counts=True)
            neff = neff_from_counts(cnt)
            top1 = float(cnt.max() / cnt.sum())
            top3 = float(np.sort(cnt)[-3:].sum() / cnt.sum()) if cnt.size >= 3 else 1.0
        else:
            neff = float("nan")
            top1 = float("nan")
            top3 = float("nan")

        # query segment id best-effort
        q_seg_id = int(meta.iloc[qi]["global_seg_index"]) if "global_seg_index" in meta.columns else int(qi)

        # store for reuse-gap + hubness (even if q_terms empty; reuse split uses GO later)
        query_neighbor_sets.append(nprot_set)
        query_proteins.append(q_prot)
        for p in nprot_set:
            neighbor_appearance[p] += 1

        if len(q_terms) == 0 or m == 0:
            rows.append({
                "tag": tag,
                "query_row": int(qi),
                "query_seg": int(q_seg_id),
                "query_protein": q_prot,
                "k": int(k_neighbors),
                "neighbor_proteins": int(m),
                "unique_neighbor_proteins": int(m),
                "neff_neighbor_proteins": float(neff),
                "go_entropy_neighborhood": go_H,
                "top1_protein_share": float(top1),
                "top3_protein_share": float(top3),
                "has_go": int(len(q_terms) > 0),
                "hit_any_shared_go": 0,
                "shared_go_frac_neighbors": float("nan"),
                "best_term": "",
                "best_pval": float("nan"),
                "best_odds_approx": float("nan"),
            })
            continue

        # --- coherence: any GO overlap with query terms (protein-level) ---
        shared = 0
        for p in nprot_set:
            if len(prot_go_all.get(p, set()) & q_terms) > 0:
                shared += 1
        shared_frac = shared / float(m) if m > 0 else float("nan")
        hit_any = 1 if shared > 0 else 0

        # --- enrichment: fisher for query's GO terms among neighbor proteins vs background ---
        out_set = bg_set - nprot_set
        n_in = len(nprot_set)
        n_out = len(out_set)

        best_p = 1.0
        best_t = ""
        best_odds = 1.0

        for t in q_terms:
            has_t = term_to_prots.get(t, set())
            a = len(nprot_set & has_t)
            if a == 0:
                continue
            b = n_in - a
            c_ = len(out_set & has_t)
            d = n_out - c_
            _, pval = fisher_exact([[a, b], [c_, d]], alternative="greater")
            odds = float((a * d) / max(1.0, (b * c_))) if (b * c_) > 0 else float("inf")
            if pval < best_p:
                best_p = float(pval)
                best_t = str(t)
                best_odds = odds

        rows.append({
            "tag": tag,
            "query_row": int(qi),
            "query_seg": int(q_seg_id),
            "query_protein": q_prot,
            "k": int(k_neighbors),
            "neighbor_proteins": int(m),
            "unique_neighbor_proteins": int(m),
            "neff_neighbor_proteins": float(neff),
            "go_entropy_neighborhood": go_H,
            "top1_protein_share": float(top1),
            "top3_protein_share": float(top3),
            "has_go": 1,
            "hit_any_shared_go": int(hit_any),
            "shared_go_frac_neighbors": float(shared_frac),
            "best_term": best_t,
            "best_pval": float(best_p) if best_t else float("nan"),
            "best_odds_approx": float(best_odds) if best_t else float("nan"),
        })

    dfq = pd.DataFrame(rows)
    used = dfq[dfq["has_go"] == 1].copy()

    # --- reuse gap on this query set ---
    reuse_stats = compute_reuse_gap(
        query_proteins=np.asarray(query_proteins, dtype=object),
        query_neighbor_protein_sets=query_neighbor_sets,
        prot_go_all=prot_go_all,
        max_pairs=10000,
        seed=seed + 777,
    )

    # --- hubness stats on appearance counts across background proteins ---
    bg_counts = np.asarray([neighbor_appearance.get(p, 0) for p in background_proteins], dtype=np.float64)
    hub_gini = gini_coefficient(bg_counts)
    hub_top1pct = top_frac_share(bg_counts, frac=0.01)
    if np.any(bg_counts > 0):
        med_pos = float(np.median(bg_counts[bg_counts > 0]))
        hub_max_over_median = float(bg_counts.max() / max(1.0, med_pos))
    else:
        hub_max_over_median = float("nan")

    rep = {
        "tag": tag,
        "go_aspect": go_aspect,
        "k_neighbors": int(k_neighbors),
        "queries_total": int(qN),
        "queries_with_go": int((dfq["has_go"] == 1).sum()),
        "exclude_same_protein": bool(exclude_same_protein),

        "mean_neighbor_proteins": float(np.nanmean(dfq["neighbor_proteins"].to_numpy())) if len(dfq) else float("nan"),

        "mean_hit_any_shared_go": float(np.nanmean(used["hit_any_shared_go"].to_numpy())) if len(used) else float("nan"),
        "mean_shared_go_frac_neighbors": float(np.nanmean(used["shared_go_frac_neighbors"].to_numpy())) if len(used) else float("nan"),
        "median_shared_go_frac_neighbors": float(np.nanmedian(used["shared_go_frac_neighbors"].to_numpy())) if len(used) else float("nan"),

        "frac_with_enriched_query_term_p<0.05": float(np.nanmean((used["best_pval"].to_numpy() < 0.05).astype(float))) if len(used) else float("nan"),
        "frac_with_enriched_query_term_p<1e-3": float(np.nanmean((used["best_pval"].to_numpy() < 1e-3).astype(float))) if len(used) else float("nan"),

        # diversity/dominance
        "median_unique_neighbor_proteins": float(np.nanmedian(used["unique_neighbor_proteins"].to_numpy())) if len(used) else float("nan"),
        "median_neff_neighbor_proteins": float(np.nanmedian(used["neff_neighbor_proteins"].to_numpy())) if len(used) else float("nan"),
        "median_top1_protein_share": float(np.nanmedian(used["top1_protein_share"].to_numpy())) if len(used) else float("nan"),
        "go_entropy_neighborhood_mean": float(np.nanmean(used["go_entropy_neighborhood"])),
        "go_entropy_neighborhood_median": float(np.nanmedian(used["go_entropy_neighborhood"])),
        "go_entropy_neighborhood_std": float(np.nanstd(used["go_entropy_neighborhood"])),

        # reuse
        **reuse_stats,

        # hubness
        "hubness_gini": float(hub_gini),
        "hubness_top1pct_share": float(hub_top1pct),
        "hubness_max_over_median": float(hub_max_over_median),
    }

    # attach raw hubness counts for writing a per-protein table later
    rep["_hubness_counts"] = {p: int(neighbor_appearance.get(p, 0)) for p in background_proteins}
    return rep, dfq




def _savefig(fig, out_base: Path):
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=250)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_diversity_violin(per_query: pd.DataFrame, out_base: Path):
    """
    Violin+box for unique_neighbor_proteins and neff_neighbor_proteins by tag.
    """
    tags = list(per_query["tag"].unique())
    metrics = ["unique_neighbor_proteins", "neff_neighbor_proteins"]

    fig = plt.figure(figsize=(10.5, 4.6))
    for mi, m in enumerate(metrics, start=1):
        ax = fig.add_subplot(1, 2, mi)
        data = []
        for t in tags:
            v = per_query.loc[per_query["tag"] == t, m].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            data.append(v)

        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        ax.boxplot(data, widths=0.15, vert=True, showfliers=False)
        ax.set_title(m)
        ax.set_xticks(np.arange(1, len(tags) + 1))
        ax.set_xticklabels(tags, rotation=30, ha="right")
        ax.set_ylabel(m)

    fig.tight_layout()
    _savefig(fig, out_base)

def plot_alignment_vs_diversity(per_query: pd.DataFrame, out_base: Path):
    """
    Scatter: shared_go_frac_neighbors vs neff_neighbor_proteins, colored by tag (no explicit colors).
    """
    fig = plt.figure(figsize=(6.8, 5.2))
    ax = fig.add_subplot(111)
    for t in per_query["tag"].unique():
        df = per_query[per_query["tag"] == t].copy()
        x = df["neff_neighbor_proteins"].to_numpy(dtype=float)
        y = df["shared_go_frac_neighbors"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=10, alpha=0.35, label=t)

    ax.set_xlabel("neff_neighbor_proteins (Simpson inverse)")
    ax.set_ylabel("shared_go_frac_neighbors")
    ax.set_title("Function alignment vs diversity")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _savefig(fig, out_base)

def plot_hubness_lorenz(hub_df: pd.DataFrame, out_base: Path):
    """
    Lorenz curve of neighbor-appearance counts across proteins, per tag.
    hub_df columns: tag, protein_key, neighbor_appearance_count
    """
    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111)

    for t in hub_df["tag"].unique():
        x = hub_df.loc[hub_df["tag"] == t, "neighbor_appearance_count"].to_numpy(dtype=float)
        x = np.sort(x)
        if x.size == 0:
            continue
        s = x.sum()
        if s <= 0:
            continue
        cum = np.cumsum(x) / s
        frac = (np.arange(1, x.size + 1) / x.size)
        g = gini_coefficient(x)
        ax.plot(frac, cum, label=f"{t} (gini={g:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")  # equality line
    ax.set_xlabel("Fraction of proteins")
    ax.set_ylabel("Fraction of neighbor appearances")
    ax.set_title("Hubness Lorenz curve (neighbor appearance)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    _savefig(fig, out_base)

def plot_reuse_gap(summary: pd.DataFrame, out_base: Path):
    """
    Bar chart of reuse_gap_mean per tag.
    """
    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)

    tags = summary["tag"].astype(str).tolist()
    vals = summary["reuse_gap_mean"].to_numpy(dtype=float)
    x = np.arange(len(tags))

    ax.bar(x, vals)
    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_ylabel("mean(J|sharedGO) - mean(J|noSharedGO)")
    ax.set_title("Function-conditional neighborhood reuse gap")
    fig.tight_layout()
    _savefig(fig, out_base)


def plot_go_entropy_hist(df, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4.5, 3.5))
    for tag, g in df.groupby("tag"):
        plt.hist(
            g["go_entropy_neighborhood"].dropna(),
            bins=40,
            alpha=0.5,
            density=True,
            label=tag,
        )

    plt.xlabel("GO entropy (normalized)")
    plt.ylabel("Density")
    plt.title("Functional diversity of neighborhoods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



# -------------------------
# Orchestration
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_root", type=str, required=True, help="e.g., segments/")
    ap.add_argument("--model_name", type=str, required=True, help="child dir under segments_root/")
    ap.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    ap.add_argument("--annotation_dir", type=str, required=True)
    ap.add_argument("--go_file", type=str, default="nrPDB-GO_annot.tsv")
    ap.add_argument("--go_aspect", type=str, default="MF", choices=["MF", "BP", "CC"])

    ap.add_argument("--out_root", type=str, default="segment_func_reports")
    ap.add_argument("--k_neighbors", type=int, default=50)
    ap.add_argument("--n_queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--min_assigned", type=int, default=3)
    ap.add_argument("--max_assigned", type=int, default=None)

    ap.add_argument("--exclude_same_protein", action="store_true", default=True)
    ap.add_argument("--include_same_protein", action="store_true", default=False,
                    help="If set, overrides exclude_same_protein to False")

    ap.add_argument("--controls", type=str, default="shuffled_embeddings,random_neighbors",
                    help="comma-separated: shuffled_embeddings,random_neighbors or empty")

    return ap.parse_args()


def main():
    a = parse_args()
    if a.include_same_protein:
        exclude_same_protein = False
    else:
        exclude_same_protein = True if a.exclude_same_protein else False

    segments_root = Path(a.segments_root)
    model_dir = segments_root / a.model_name / a.split
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {model_dir}")

    out_dir = ensure_dir(Path(a.out_root) / a.model_name / a.split)

    print(f"[INFO] Loading data from: {model_dir}")
    go_df = load_go_annotations(Path(a.annotation_dir), go_file=a.go_file)
    E, meta = load_segment_outputs(model_dir, a.split, as_torch=False)
    meta = add_id_columns(meta)

    print(f"[INFO] Original segments: {E.shape[0]}")
    E, meta = filter_segments_by_assigned(E, meta, min_assigned=a.min_assigned, max_assigned=a.max_assigned)
    print(f"[INFO] Filtered segments: {E.shape[0]} (min_assigned={a.min_assigned}, max_assigned={a.max_assigned})")

    X = l2_normalize(E.astype("float32", copy=False))

    print(f"[INFO] Building neighbors and evaluating GO enrichment...")
    idx_true = neighbors_cosine(X, k=a.k_neighbors, exclude_self=True)

    controls = [c.strip() for c in str(a.controls).split(",") if c.strip()]
    tags_and_idx = [("knn_true", idx_true)]

    if "shuffled_embeddings" in controls:
        tags_and_idx.append(
            ("control_shuffled_embeddings", neighbors_shuffled_embeddings(X, k=a.k_neighbors, seed=a.seed + 11))
        )
    if "random_neighbors" in controls:
        tags_and_idx.append(
            ("control_random_neighbors", neighbors_random(X.shape[0], k=a.k_neighbors, seed=a.seed + 23, exclude_self=True))
        )

    reports: List[Dict[str, Any]] = []
    per_query_all: List[pd.DataFrame] = []
    hubness_rows: List[pd.DataFrame] = []

    for tag, idx in tags_and_idx:
        print(f"[INFO] Evaluating tag: {tag}")
        rep, dfq = eval_unit_knn_go(
            X=X,
            meta=meta,
            go_df=go_df,
            go_aspect=a.go_aspect,
            k_neighbors=a.k_neighbors,
            seed=a.seed + 999,          # deterministic per tag run
            n_queries=a.n_queries,
            exclude_same_protein=exclude_same_protein,
            neighbor_idx=idx,
            tag=tag,
        )

        # extract hubness counts for this tag into a per-protein table
        hc = rep.pop("_hubness_counts", {})
        if hc:
            hub_df = pd.DataFrame({
                "tag": tag,
                "protein_key": list(hc.keys()),
                "neighbor_appearance_count": list(hc.values()),
            })
            hubness_rows.append(hub_df)
    
        reports.append(rep)
        per_query_all.append(dfq)

    # Save main outputs
    out_json = out_dir / f"unit_knn_go_enrichment_{a.split}.json"
    out_csv = out_dir / f"unit_knn_go_enrichment_{a.split}_per_query.csv"
    out_sum = out_dir / f"unit_knn_go_enrichment_{a.split}_summary.csv"
    out_hub = out_dir / f"unit_knn_neighbor_hubness_{a.split}.csv"
    comp_out = out_dir / f"unit_knn_go_compositionality_{a.split}.csv"


    payload = {
        "model_name": a.model_name,
        "split": a.split,
        "go_aspect": a.go_aspect,
        "k_neighbors": int(a.k_neighbors),
        "n_queries": int(min(a.n_queries, X.shape[0])),
        "exclude_same_protein": bool(exclude_same_protein),
        "filters": {
            "min_assigned": int(a.min_assigned),
            "max_assigned": None if a.max_assigned is None else int(a.max_assigned),
        },
        "reports": reports,
    }

    out_json.write_text(json.dumps(payload, indent=2))
    per_query = pd.concat(per_query_all, axis=0, ignore_index=True)
    per_query.to_csv(out_csv, index=False)
    summary = pd.DataFrame(reports)
    summary.to_csv(out_sum, index=False)


    if hubness_rows:
        hubness = pd.concat(hubness_rows, axis=0, ignore_index=True)
        hubness.to_csv(out_hub, index=False)
    else: 
        hubness = pd.DataFrame(columns=["tag", "protein_key", "neighbor_appearance_count"])
        hubness.to_csv(out_hub, index=False)


   
    plot_diversity_violin(per_query, out_dir / f"plot_diversity_violin_{a.split}")
    plot_alignment_vs_diversity(per_query, out_dir / f"plot_alignment_vs_diversity_{a.split}")
    plot_hubness_lorenz(hubness, out_dir / f"plot_hubness_lorenz_{a.split}")
    plot_reuse_gap(summary, out_dir / f"plot_reuse_gap_{a.split}")
    plot_go_entropy_hist(per_query, out_dir / f"plot_go_entropy_hist_{a.split}.png")
    print(f"[OK] Wrote plots to: {out_dir}")
    

    print(f"[OK] {out_json}")
    print(f"[OK] {out_csv}")
    print(f"[OK] {out_sum}")
    print(f"[OK] {out_hub}")


if __name__ == "__main__":
    main()

""" 
# Summary of capabilities:
Alignment (shared GO)
Diversity (Neff + GO entropy)
Controlled reuse (reuse gap)
Anti-hubness diagnostics (Lorenz/Gini)
Scalable evaluation (kNN, no clustering assumptions)
"""
