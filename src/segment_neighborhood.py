#!/usr/bin/env python3
"""
neighborhood_diagnostics.py
---------------------------
Standalone script for neighborhood diagnostics:
  1) Protein diversity per kNN neighborhood
  2) GO multiplicity per neighborhood
  3) Neighborhood reuse overlap across queries (Jaccard of neighbor protein sets)

Two kNN backends:
  - bruteforce: cosine via block matmul (O(N^2)) + multiprocessing over queries (parallel)
  - faiss: cosine kNN via FAISS (fast) + single-process metrics (no multiprocessing)

Inputs per method:
  - embeddings: .npy (N,D) or .pt containing dict["embeddings"]
  - metadata CSV: must contain a protein key column (e.g., pdb_id) used to join GO

GO annotations:
  - TSV with columns: protein_key, MF, BP, CC (comma-separated GO terms)

Outputs:
  <out_dir>/
    metrics_k<K>.csv
    diversity_k<K>.png/.pdf
    go_multiplicity_k<K>.png/.pdf
    reuse_overlap_k<K>_<pair_mode>.png/.pdf
    reuse_pairs_k<K>_<pair_mode>.csv

Example (FAISS, no parallel):
  python neighborhood_diagnostics.py \
    --backend faiss --faiss_exact \
    --go_tsv ../data/GeneOntology/nrPDB-GO_annot.tsv --aspect MF \
    --k 50 --sample_queries 5000 --exclude_same_protein --require_query_has_go \
    --out_dir segment_func_reports/neighborhoods/test \
    --method puffin ../ismb26/segments/puffin_K64/test/test_segment_embeddings.npy ../ismb26/segments/puffin_K64/test/test_segment_metadata.csv pdb_id \
    --method mincut ../ismb26/segments/mincut_K64/test/test_segment_embeddings.npy ../ismb26/segments/mincut_K64/test/test_segment_metadata.csv pdb_id \
    --method louvain ../ismb26/segments/louvain/test/test_segment_embeddings.npy ../ismb26/segments/louvain/test/test_segment_metadata.csv pdb_id

Example (Brute force kNN + parallel per-query metrics):
  python neighborhood_diagnostics.py \
    --backend bruteforce --n_jobs 8 --chunksize 256 --knn_chunk 4096 \
    --go_tsv ../data/GeneOntology/nrPDB-GO_annot.tsv --aspect MF \
    --k 50 --sample_queries 5000 --exclude_same_protein --require_query_has_go \
    --out_dir segment_func_reports/neighborhoods/test \
    --method puffin ../ismb26/segments/puffin_K64/test/test_segment_embeddings.npy ../ismb26/segments/puffin_K64/test/test_segment_metadata.csv pdb_id \
    --method mincut ../ismb26/segments/mincut_K64/test/test_segment_embeddings.npy ../ismb26/segments/mincut_K64/test/test_segment_metadata.csv pdb_id \
    --method louvain ../ismb26/segments/louvain/test/test_segment_embeddings.npy ../ismb26/segments/louvain/test/test_segment_metadata.csv pdb_id
"""

import math
import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


import torch
import faiss  


# ----------------------------
# Data containers
# ----------------------------
@dataclass
class MethodData:
    name: str
    E: np.ndarray                 # (N,D) float32
    meta: pd.DataFrame            # columns: seg_id, protein_key
    knn_idx: Optional[np.ndarray] = None  # (N,k) int32 (optional)


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig_dual(fig, out_base: Path):
    ensure_dir(out_base.parent)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=250)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)


def load_embeddings(path: Path) -> np.ndarray:
    """
    Supports:
      - .npy  -> direct numpy array
      - .pt   -> dict with key 'embeddings'
    Returns float32 numpy array (N,D)
    """
    path = Path(path)
    if path.suffix == ".npy":
        E = np.load(path).astype("float32")
    elif path.suffix == ".pt":
        if torch is None:
            raise ImportError("torch is required to load .pt embeddings")
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or "embeddings" not in obj:
            raise ValueError(f"{path} must contain dict['embeddings']")
        E = obj["embeddings"].detach().cpu().numpy().astype("float32")
    else:
        raise ValueError(f"Unsupported embedding format: {path}")
    return E


def load_segment_table(
    embeddings_path: Path,
    metadata_csv: Path,
    *,
    protein_key_col: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Returns:
      E: (N,D) float32
      meta: DataFrame with at least ['seg_id','protein_key']
    """
    E = load_embeddings(Path(embeddings_path))
    meta = pd.read_csv(Path(metadata_csv))

    if len(meta) != E.shape[0]:
        raise ValueError(f"Row mismatch: {len(meta)} metadata vs {E.shape[0]} embeddings")

    if protein_key_col not in meta.columns:
        raise ValueError(f"Missing column '{protein_key_col}' in {metadata_csv}. Found: {list(meta.columns)}")

    meta = meta.reset_index(drop=True).copy()
    meta["seg_id"] = np.arange(len(meta), dtype=int)
    meta["protein_key"] = meta[protein_key_col].astype(str)
    meta["protein_key"] = meta["protein_key"].str.split("_").str[0] 

    return E, meta


def load_go_df(go_tsv: Path) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      protein_key | MF | BP | CC
    """
    go_tsv = Path(go_tsv)
    for skip in (12, 13, 14):
        try:
            df = pd.read_csv(go_tsv, sep="\t", skiprows=skip)
            if df.shape[1] >= 4:
                df = df.iloc[:, :4].copy()
                df.columns = ["protein_key", "MF", "BP", "CC"]
                df["protein_key"] = df["protein_key"].astype(str)
                return df
        except Exception:
            pass
    raise RuntimeError(f"Could not parse GO TSV with common skiprows: {go_tsv}")


def build_go_map(go_df: pd.DataFrame, aspect: str, protein_key_col: str = "protein_key") -> Dict[str, Set[str]]:
    """
    go_df: DataFrame with columns [protein_key_col, 'MF'/'BP'/'CC'] (comma-separated or list-like)
    """
    aspect = aspect.upper()
    if aspect not in ("MF", "BP", "CC"):
        raise ValueError("aspect must be MF/BP/CC")

    m: Dict[str, Set[str]] = {}
    for _, r in go_df.iterrows():
        pk = str(r[protein_key_col])
        val = r[aspect]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            m[pk] = set()
            continue
        if isinstance(val, (list, tuple, set)):
            m[pk] = set(map(str, val))
        else:
            s = str(val).strip()
            if not s or s.lower() == "nan":
                m[pk] = set()
            else:
                m[pk] = set(t for t in s.split(",") if t)
    return m


# ----------------------------
# kNN backends
# ----------------------------
def knn_cosine_bruteforce(E: np.ndarray, k: int, *, chunk: int = 4096, exclude_self: bool = True) -> np.ndarray:
    """
    Returns knn indices for each row, based on cosine similarity.
    Complexity: O(N^2). Use only if N is moderate.
    """
    E = l2_normalize(E.astype("float32"))
    N = E.shape[0]
    if exclude_self and N <= 1:
        raise ValueError("Need N>1 to exclude self neighbors")
    k_eff = min(k + (1 if exclude_self else 0), N)

    out_cols = min(k, N - 1 if exclude_self else N)
    out = np.empty((N, out_cols), dtype=np.int32)

    for s in tqdm(range(0, N, chunk), desc=f"bruteforce kNN (k={k})", leave=False):
        e = min(N, s + chunk)
        S = E[s:e] @ E.T  # (bs, N)
        if exclude_self:
            rows = np.arange(s, e)
            S[np.arange(e - s), rows] = -1e9

        idx = np.argpartition(-S, kth=np.arange(k_eff), axis=1)[:, :k_eff]
        sim = S[np.arange(e - s)[:, None], idx]
        ord_ = np.argsort(-sim, axis=1)
        idx = idx[np.arange(e - s)[:, None], ord_]
        idx = idx[:, :out_cols]

        out[s:e] = idx.astype(np.int32)

    return out


def knn_cosine_faiss(
    E: np.ndarray,
    k: int,
    *,
    exclude_self: bool = True,
    exact: bool = True,
    use_gpu: bool = False,
    gpu_device: int = 0,
    ivf_nlist: int = 4096,
    ivf_nprobe: int = 32,
    train_max: int = 200_000,
) -> np.ndarray:
    """
    Fast cosine kNN using FAISS. No multiprocessing here.

    Cosine = inner product on L2-normalized vectors.
    - exact=True => IndexFlatIP
    - exact=False => IndexIVFFlat (approx)

    Returns:
      knn_idx: (N, k) int32
    """
    if faiss is None:
        raise ImportError("faiss not installed. pip install faiss-cpu (or faiss-gpu).")

    X = E.astype("float32", copy=False)
    faiss.normalize_L2(X)

    N, d = X.shape
    k_eff = min(k + (1 if exclude_self else 0), N)

    if exact:
        index = faiss.IndexFlatIP(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, int(ivf_nlist), faiss.METRIC_INNER_PRODUCT)
        # train
        train_X = X
        if N > int(train_max):
            rng = np.random.default_rng(0)
            train_X = X[rng.choice(N, size=int(train_max), replace=False)]
        index.train(train_X)
        index.nprobe = int(ivf_nprobe)

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("This faiss build has no GPU support. Install faiss-gpu.")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, int(gpu_device), index)

    index.add(X)

    _, idx = index.search(X, k_eff)  # (N, k_eff)

    if exclude_self:
        out = np.empty((N, min(k, N - 1)), dtype=np.int32)
        for i in range(N):
            row = idx[i]
            row = row[row != i]
            if row.size < out.shape[1]:
                pad = np.full((out.shape[1] - row.size,), row[-1] if row.size > 0 else i, dtype=row.dtype)
                row = np.concatenate([row, pad], axis=0)
            out[i] = row[: out.shape[1]].astype(np.int32)
        return out

    return idx[:, : min(k, N)].astype(np.int32)


# ----------------------------
# Metrics (parallel worker for bruteforce backend)
# ----------------------------
_G_knn_idx = None
_G_prot_ids = None
_G_go_by_prot = None
_G_require_query_has_go = None
_G_exclude_same_protein = None
_G_k = None
_G_method_name = None


def _init_worker(knn_idx, prot_ids, go_by_prot, k, method_name, exclude_same_protein, require_query_has_go):
    global _G_knn_idx, _G_prot_ids, _G_go_by_prot, _G_k, _G_method_name
    global _G_exclude_same_protein, _G_require_query_has_go
    _G_knn_idx = knn_idx
    _G_prot_ids = prot_ids
    _G_go_by_prot = go_by_prot
    _G_k = int(k)
    _G_method_name = str(method_name)
    _G_exclude_same_protein = bool(exclude_same_protein)
    _G_require_query_has_go = bool(require_query_has_go)


def _one_query(qi: int):
    qi = int(qi)
    q_pid = int(_G_prot_ids[qi])
    q_go = _G_go_by_prot[q_pid]

    if _G_require_query_has_go and len(q_go) == 0:
        return None

    neigh = _G_knn_idx[qi]  # (k,)
    neigh_pids = _G_prot_ids[neigh]

    if _G_exclude_same_protein:
        mask = neigh_pids != q_pid
        if not np.any(mask):
            return None
        neigh_pids = neigh_pids[mask]

    uniq_prots = int(np.unique(neigh_pids).size)

    union_go = set()
    any_shared = 0
    for pid in neigh_pids:
        s = _G_go_by_prot[int(pid)]
        if s:
            union_go |= s
        if q_go and s and (q_go & s):
            any_shared += 1

    go_multiplicity = float(len(union_go))
    shared_go_frac_neighbors = float(any_shared / len(neigh_pids)) if q_go else float("nan")

    return {
        "method": _G_method_name,
        "query_idx": qi,
        "query_protein": q_pid,  # int id -> mapped back later
        "k": _G_k,
        "neighbors_used": int(len(neigh_pids)),
        "protein_diversity": float(uniq_prots),
        "go_multiplicity": float(go_multiplicity),
        "shared_go_frac_neighbors": float(shared_go_frac_neighbors),
    }


# ----------------------------
# Neighborhood metrics (two variants)
# ----------------------------
def _prep_compact_ids(meta: pd.DataFrame, go_map: Dict[str, Set[str]]):
    prot_keys = meta["protein_key"].astype(str).to_numpy()
    uniq_prots, prot_ids = np.unique(prot_keys, return_inverse=True)
    prot_ids = prot_ids.astype(np.int32)
    go_by_prot = [set(go_map.get(pk, set())) for pk in uniq_prots]
    return uniq_prots, prot_ids, go_by_prot


def compute_neighborhood_metrics_bruteforce_parallel(
    method: MethodData,
    go_map: Dict[str, Set[str]],
    *,
    k: int,
    sample_queries: int = 5000,
    seed: int = 0,
    exclude_same_protein: bool = True,
    require_query_has_go: bool = True,
    n_jobs: Optional[int] = None,
    chunksize: int = 256,
    knn_chunk: int = 4096,
) -> pd.DataFrame:
    """
    Brute force kNN + multiprocessing over queries.
    """
    meta = method.meta.reset_index(drop=True).copy()
    if "protein_key" not in meta.columns:
        raise ValueError("meta must have protein_key")
    N = len(meta)
    if N == 0:
        return pd.DataFrame()

    # kNN (slow part)
    knn_idx = knn_cosine_bruteforce(method.E, k=k, exclude_self=True, chunk=knn_chunk)
    knn_idx = np.asarray(knn_idx, dtype=np.int32)

    uniq_prots, prot_ids, go_by_prot = _prep_compact_ids(meta, go_map)

    rng = np.random.default_rng(seed)
    q_idx = np.arange(N, dtype=np.int32)
    if sample_queries and sample_queries < N:
        q_idx = rng.choice(q_idx, size=int(sample_queries), replace=False)

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=int(n_jobs),
        initializer=_init_worker,
        initargs=(knn_idx, prot_ids, go_by_prot, k, method.name, exclude_same_protein, require_query_has_go),
    ) as pool:
        it = pool.imap_unordered(_one_query, q_idx.tolist(), chunksize=int(chunksize))
        out = []
        for r in tqdm(it, total=len(q_idx), desc=f"{method.name} metrics (bruteforce, k={k})"):
            if r is not None:
                out.append(r)

    df = pd.DataFrame(out)
    if not df.empty:
        df["query_protein"] = df["query_protein"].apply(lambda pid: uniq_prots[int(pid)])
    return df


def compute_neighborhood_metrics_faiss_singleprocess(
    method: MethodData,
    go_map: Dict[str, Set[str]],
    *,
    k: int,
    sample_queries: int = 5000,
    seed: int = 0,
    exclude_same_protein: bool = True,
    require_query_has_go: bool = True,
    faiss_exact: bool = True,
    faiss_use_gpu: bool = False,
    faiss_gpu_device: int = 0,
    faiss_ivf_nlist: int = 4096,
    faiss_ivf_nprobe: int = 32,
) -> pd.DataFrame:
    """
    FAISS kNN + single-process loop (no multiprocessing).
    """
    meta = method.meta.reset_index(drop=True).copy()
    if "protein_key" not in meta.columns:
        raise ValueError("meta must have protein_key")
    N = len(meta)
    if N == 0:
        return pd.DataFrame()

    knn_idx = knn_cosine_faiss(
        method.E,
        k=k,
        exclude_self=True,
        exact=bool(faiss_exact),
        use_gpu=bool(faiss_use_gpu),
        gpu_device=int(faiss_gpu_device),
        ivf_nlist=int(faiss_ivf_nlist),
        ivf_nprobe=int(faiss_ivf_nprobe),
    )

    uniq_prots, prot_ids, go_by_prot = _prep_compact_ids(meta, go_map)

    rng = np.random.default_rng(seed)
    q_idx = np.arange(N, dtype=np.int32)
    if sample_queries and sample_queries < N:
        q_idx = rng.choice(q_idx, size=int(sample_queries), replace=False)

    skip_queries_go = set()
    skip_queries_neigh = set()
    rows = []
    for qi in tqdm(q_idx.tolist(), desc=f"{method.name} metrics (faiss, k={k})"):
        q_pid = int(prot_ids[qi])
        q_go = go_by_prot[q_pid]
        if require_query_has_go and len(q_go) == 0:
            skip_queries_go.add(int(qi))
            # print(f"Skipping query {qi} (protein {uniq_prots[q_pid]}) with no GO")
            continue

        neigh = knn_idx[qi]  # (k,)
        neigh_pids = prot_ids[neigh]

        if exclude_same_protein:
            mask = neigh_pids != q_pid
            if not np.any(mask):
                skip_queries_neigh.add(int(qi))
                # print(f"Skipping query {qi} (protein {uniq_prots[q_pid]}) with no valid neighbors after excluding same protein")
                continue
            neigh_pids = neigh_pids[mask]

        uniq = int(np.unique(neigh_pids).size)

        union_go = set()
        any_shared = 0
        for pid in neigh_pids:
            s = go_by_prot[int(pid)]
            if s:
                union_go |= s
            if q_go and s and (q_go & s):
                any_shared += 1

        # print(f"Query {qi} (protein {uniq_prots[q_pid]}): diversity={uniq}, go_multiplicity={len(union_go)}, shared_go_frac_neighbors={any_shared}/{len(neigh_pids)}" )
        rows.append({
            "method": method.name,
            "query_idx": int(qi),
            "query_protein": str(uniq_prots[q_pid]),
            "k": int(k),
            "neighbors_used": int(len(neigh_pids)),
            "protein_diversity": float(uniq),
            "go_multiplicity": float(len(union_go)),
            "shared_go_frac_neighbors": float(any_shared / len(neigh_pids)) if q_go else float("nan"),
        })
    print(f"[INFO] Skipped {len(skip_queries_go)} queries with no GO annotations")
    print(f"[INFO] Skipped {len(skip_queries_neigh)} queries with no valid neighbors after excluding same protein")

    return pd.DataFrame(rows)


# ----------------------------
# Reuse overlap (backend-specific kNN; always single-process)
# ----------------------------

def _compute_knn_idx_for_method(
    method: MethodData,
    *,
    backend: str,
    k: int,
    knn_chunk: int = 4096,
    faiss_exact: bool = True,
    faiss_use_gpu: bool = False,
    faiss_gpu_device: int = 0,
    faiss_ivf_nlist: int = 4096,
    faiss_ivf_nprobe: int = 32,
) -> np.ndarray:
    """
    Compute kNN indices once and reuse for multiple metrics.
    """
    if method.knn_idx is not None and method.knn_idx.shape[1] >= k:
        return method.knn_idx[:, :k].astype(np.int32, copy=False)

    if backend == "bruteforce":
        knn_idx = knn_cosine_bruteforce(method.E, k=k, exclude_self=True, chunk=knn_chunk)
    elif backend == "faiss":
        knn_idx = knn_cosine_faiss(
            method.E,
            k=k,
            exclude_self=True,
            exact=bool(faiss_exact),
            use_gpu=bool(faiss_use_gpu),
            gpu_device=int(faiss_gpu_device),
            ivf_nlist=int(faiss_ivf_nlist),
            ivf_nprobe=int(faiss_ivf_nprobe),
        )
    else:
        raise ValueError("backend must be bruteforce or faiss")

    knn_idx = np.asarray(knn_idx, dtype=np.int32)
    method.knn_idx = knn_idx
    return knn_idx


def compute_reuse_retrieval_metrics(
    method: MethodData,
    go_map: Dict[str, Set[str]],
    *,
    knn_idx: np.ndarray,
    k: int,
    sample_queries: int = 5000,
    seed: int = 0,
    exclude_same_protein: bool = True,
    require_query_has_go: bool = True,
) -> Dict[str, float]:
    """
    Cross-protein retrieval:
      - Reuse@k: fraction of queries whose top-k contains any neighbor protein sharing >=1 GO term with query protein
      - MRR@k: mean reciprocal rank of first such hit (0 if none)

    Note: this is protein-level GO sharing, consistent with your other neighborhood metrics.
    """
    meta = method.meta.reset_index(drop=True)
    N = len(meta)
    rng = np.random.default_rng(seed)

    q_idx = np.arange(N, dtype=np.int32)
    if sample_queries and sample_queries < N:
        q_idx = rng.choice(q_idx, size=int(sample_queries), replace=False)

    hits = 0
    rr_sum = 0.0
    used = 0
    skipped_no_go = 0
    skipped_no_valid_neighbors = 0

    for qi in q_idx.tolist():
        qp = str(meta.loc[qi, "protein_key"])
        q_go = go_map.get(qp, set())
        if require_query_has_go and len(q_go) == 0:
            skipped_no_go += 1
            continue

        neigh = knn_idx[qi, :k]
        neigh_prots = meta.loc[neigh, "protein_key"].astype(str).tolist()

        if exclude_same_protein:
            neigh_prots = [p for p in neigh_prots if p != qp]
            if len(neigh_prots) == 0:
                skipped_no_valid_neighbors += 1
                continue

        used += 1

        first_rank = None
        for r, npk in enumerate(neigh_prots, start=1):
            s_go = go_map.get(npk, set())
            if q_go and s_go and (q_go & s_go):
                first_rank = r
                break

        if first_rank is not None:
            hits += 1
            rr_sum += 1.0 / float(first_rank)

    reuse_at_k = hits / used if used > 0 else float("nan")
    mrr_at_k = rr_sum / used if used > 0 else float("nan")

    return {
        "queries_used": float(used),
        "queries_skipped_no_go": float(skipped_no_go),
        "queries_skipped_no_valid_neighbors": float(skipped_no_valid_neighbors),
        f"reuse_at_{k}": float(reuse_at_k),
        f"mrr_at_{k}": float(mrr_at_k),
    }


def compute_reuse_log2_fold_enrichment(
    df_pairs_method: pd.DataFrame,
    df_pairs_baseline: pd.DataFrame,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    log2 fold enrichment of reuse overlap (Jaccard) over baseline:
      log2( mean_jaccard_method / mean_jaccard_baseline )

    You can pass shared_go-only pair tables (recommended).
    """
    m = float(df_pairs_method["jaccard_overlap"].mean()) if not df_pairs_method.empty else float("nan")
    b = float(df_pairs_baseline["jaccard_overlap"].mean()) if not df_pairs_baseline.empty else float("nan")

    if np.isnan(m) or np.isnan(b):
        return {"reuse_jaccard_mean": m, "reuse_jaccard_baseline_mean": b, "reuse_log2FE": float("nan")}

    log2fe = math.log2((m + eps) / (b + eps))
    return {
        "reuse_jaccard_mean": m,
        "reuse_jaccard_baseline_mean": b,
        "reuse_log2FE": float(log2fe),
    }


def _make_fast_shuffled_baseline_pairs(
    df_pairs: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Fast shuffle-null for reuse overlap:
    breaks any true correspondence by shuffling jaccard scores across pairs.
    (This is a conservative 'association-breaking' null for the overlap statistic itself.)

    If you prefer a *structural* null, use an explicit method 'control_shuffled_embeddings'.
    """
    if df_pairs.empty:
        return df_pairs.copy()

    rng = np.random.default_rng(seed)
    out = df_pairs.copy()
    out["jaccard_overlap"] = rng.permutation(out["jaccard_overlap"].to_numpy())
    out["method"] = out["method"].astype(str) + "_baseline_fastshuffle"
    return out


def compute_reuse_overlap(
    method: MethodData,
    go_map: Dict[str, Set[str]],
    *,
    k: int,
    backend: str,
    seed: int = 0,
    n_queries: int = 2000,
    n_pairs: int = 3000,
    exclude_same_protein: bool = True,
    require_query_has_go: bool = True,
    pair_mode: str = "shared_go",
    # bruteforce
    knn_chunk: int = 4096,
    # faiss
    faiss_exact: bool = True,
    faiss_use_gpu: bool = False,
    faiss_gpu_device: int = 0,
    faiss_ivf_nlist: int = 4096,
    faiss_ivf_nprobe: int = 32,
) -> pd.DataFrame:
    meta = method.meta.reset_index(drop=True).copy()
    N = len(meta)
    if N == 0:
        return pd.DataFrame(columns=["method", "k", "pair_mode", "jaccard_overlap", "i", "j"])

    if backend == "bruteforce":
        knn_idx = knn_cosine_bruteforce(method.E, k=k, exclude_self=True, chunk=knn_chunk)
    elif backend == "faiss":
        knn_idx = knn_cosine_faiss(
            method.E,
            k=k,
            exclude_self=True,
            exact=bool(faiss_exact),
            use_gpu=bool(faiss_use_gpu),
            gpu_device=int(faiss_gpu_device),
            ivf_nlist=int(faiss_ivf_nlist),
            ivf_nprobe=int(faiss_ivf_nprobe),
        )
    else:
        raise ValueError("backend must be bruteforce or faiss")

    rng = np.random.default_rng(seed)
    q_pool = np.arange(N)
    if n_queries < N:
        q_pool = rng.choice(q_pool, size=n_queries, replace=False)

    # filter by GO availability
    keep = []
    for qi in q_pool:
        qp = str(meta.loc[qi, "protein_key"])
        gos = go_map.get(qp, set())
        if require_query_has_go and len(gos) == 0:
            continue
        keep.append(int(qi))
    if len(keep) < 2:
        return pd.DataFrame(columns=["method", "k", "pair_mode", "jaccard_overlap", "i", "j"])

    keep = np.array(keep, dtype=int)

    neigh_sets: Dict[int, Set[str]] = {}
    q_go_map: Dict[int, Set[str]] = {}
    for qi in keep:
        qp = str(meta.loc[qi, "protein_key"])
        q_go_map[qi] = go_map.get(qp, set())

        neigh = knn_idx[qi]
        prots = meta.loc[neigh, "protein_key"].astype(str).tolist()
        if exclude_same_protein:
            prots = [p for p in prots if p != qp]
        neigh_sets[qi] = set(prots)

    rows = []
    trials = 0
    max_trials = n_pairs * 50
    while len(rows) < n_pairs and trials < max_trials:
        trials += 1
        i, j = rng.choice(keep, size=2, replace=False)
        gi, gj = q_go_map[i], q_go_map[j]

        if pair_mode == "shared_go":
            if len(gi & gj) == 0:
                continue
        elif pair_mode == "same_primary_go":
            if len(gi) == 0 or len(gj) == 0:
                continue
            if sorted(gi)[0] != sorted(gj)[0]:
                continue
        else:
            raise ValueError("pair_mode must be shared_go or same_primary_go")

        A, B = neigh_sets[i], neigh_sets[j]
        if len(A) == 0 or len(B) == 0:
            continue

        jac = len(A & B) / float(len(A | B))
        rows.append({
            "method": method.name,
            "k": int(k),
            "pair_mode": pair_mode,
            "i": int(i),
            "j": int(j),
            "jaccard_overlap": float(jac),
        })

    return pd.DataFrame(rows)


# ----------------------------
# Plotting
# ----------------------------
def plot_violin_box(df_all: pd.DataFrame, ycol: str, ylabel: str, title: str, out_base: Path):
    methods = list(df_all["method"].unique())
    data = [df_all[df_all["method"] == m][ycol].dropna().values for m in methods]

    fig = plt.figure(figsize=(7.2, 4.3))
    ax = fig.add_subplot(111)

    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    ax.boxplot(data, widths=0.15, patch_artist=False)

    ax.set_xticks(np.arange(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i, m in enumerate(methods, start=1):
        vals = df_all[df_all["method"] == m][ycol].dropna().values
        if vals.size == 0:
            continue
        med = np.median(vals)
        ax.text(i, med, f"{med:.3f}" if med < 10 else f"{med:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    savefig_dual(fig, out_base)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--backend", type=str, default="faiss", choices=["faiss", "bruteforce"],
                    help="faiss: fast kNN, single-process metrics; bruteforce: O(N^2) kNN + parallel metrics")

    ap.add_argument("--go_tsv", type=str, required=True, help="GO TSV (nrPDB-GO_annot.tsv)")
    ap.add_argument("--aspect", type=str, default="MF", choices=["MF", "BP", "CC"], help="GO aspect")

    ap.add_argument("--k", type=int, default=50, help="kNN neighborhood size")
    ap.add_argument("--sample_queries", type=int, default=5000, help="number of queries sampled per method (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_same_protein", action="store_true", default=False)
    ap.add_argument("--require_query_has_go", action="store_true", default=False)

    # Bruteforce settings (used only if backend=bruteforce)
    ap.add_argument("--n_jobs", type=int, default=None, help="parallel workers (default cpu_count-1)")
    ap.add_argument("--chunksize", type=int, default=256, help="mp chunksize")
    ap.add_argument("--knn_chunk", type=int, default=4096, help="block size for brute-force kNN matmul")

    # FAISS settings (used only if backend=faiss OR reuse overlap with backend=faiss)
    ap.add_argument("--faiss_exact", action="store_true", default=False, help="Use exact IndexFlatIP (default)")
    ap.add_argument("--faiss_ivf", action="store_true", default=False, help="Use IVF approximate index (IndexIVFFlat)")
    ap.add_argument("--faiss_nlist", type=int, default=4096, help="IVF nlist")
    ap.add_argument("--faiss_nprobe", type=int, default=32, help="IVF nprobe")
    ap.add_argument("--faiss_use_gpu", action="store_true", default=False, help="Use FAISS GPU (if available)")
    ap.add_argument("--faiss_gpu_device", type=int, default=0)

    # reuse overlap
    ap.add_argument("--reuse_n_queries", type=int, default=2000)
    ap.add_argument("--reuse_n_pairs", type=int, default=3000)
    ap.add_argument("--pair_mode", type=str, default="shared_go", choices=["shared_go", "same_primary_go"])
    ap.add_argument("--reuse_baseline_method", type=str, default="control_shuffled_embeddings",
                help="Method name to use as shuffled baseline for log2FE. "
                     "If not found, uses fast shuffle-null on the same method.")


    ap.add_argument("--out_dir", type=str, default="segment_func_reports/")


    # Repeatable method spec: name emb_path meta_csv protein_key_col
    ap.add_argument(
        "--method",
        nargs=4,
        action="append",
        metavar=("NAME", "EMB", "META", "PROT_COL"),
        required=True,
        help="Method spec: NAME EMB_PATH META_CSV PROTEIN_KEY_COL (repeatable)",
    )

    return ap.parse_args()


def main():
    a = parse_args()
    out_dir = ensure_dir(Path(a.out_dir))

    # GO map
    go_df = load_go_df(Path(a.go_tsv))
    go_map = build_go_map(go_df, aspect=a.aspect, protein_key_col="protein_key")

    print(f"[OK] Loaded GO annotations for {len(go_map)} proteins, aspect={a.aspect}")
    # Load methods
    methods: List[MethodData] = []
    for name, emb, meta_csv, prot_col in a.method:
        E, meta = load_segment_table(Path(emb), Path(meta_csv), protein_key_col=prot_col)
        methods.append(MethodData(name=str(name), E=E, meta=meta))
        print(f"[OK] Loaded method '{name}': {E.shape[0]} segments, dim={E.shape[1]}")


    backend = str(a.backend)

    # Decide FAISS exact vs IVF
    if backend == "faiss":
        # default: exact unless user forces ivf
        faiss_exact = True if (not a.faiss_ivf) else False
        if a.faiss_exact:
            faiss_exact = True
    else:
        faiss_exact = True  # not used

    # Compute per-method metrics
    dfs = []
    for m in methods:
        print(f"[..] Computing neighborhood metrics for method '{m.name}' (k={a.k}) using backend '{backend}'")
        if backend == "bruteforce":
            df = compute_neighborhood_metrics_bruteforce_parallel(
                m,
                go_map,
                k=int(a.k),
                sample_queries=int(a.sample_queries),
                seed=int(a.seed),
                exclude_same_protein=bool(a.exclude_same_protein),
                require_query_has_go=bool(a.require_query_has_go),
                n_jobs=a.n_jobs,
                chunksize=int(a.chunksize),
                knn_chunk=int(a.knn_chunk),
            )
        else:
            df = compute_neighborhood_metrics_faiss_singleprocess(
                m,
                go_map,
                k=int(a.k),
                sample_queries=int(a.sample_queries),
                seed=int(a.seed),
                exclude_same_protein=bool(a.exclude_same_protein),
                require_query_has_go=bool(a.require_query_has_go),
                faiss_exact=bool(faiss_exact),
                faiss_use_gpu=bool(a.faiss_use_gpu),
                faiss_gpu_device=int(a.faiss_gpu_device),
                faiss_ivf_nlist=int(a.faiss_nlist),
                faiss_ivf_nprobe=int(a.faiss_nprobe),
            )
        print(f"[OK] Method '{m.name}': computed metrics for {len(df)} queries")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    metrics_csv = out_dir / f"metrics_k{a.k}.csv"
    df_all.to_csv(metrics_csv, index=False)

    # Plots (diversity + multiplicity)
    if not df_all.empty:
        plot_violin_box(
            df_all,
            ycol="protein_diversity",
            ylabel="# unique proteins among neighbors",
            title=f"Protein diversity @ k={a.k} (exclude_same_protein={a.exclude_same_protein})",
            out_base=out_dir / f"diversity_k{a.k}",
        )
        plot_violin_box(
            df_all,
            ycol="go_multiplicity",
            ylabel="# unique GO terms in neighbor proteins (union)",
            title=f"GO multiplicity @ k={a.k} (exclude_same_protein={a.exclude_same_protein})",
            out_base=out_dir / f"go_multiplicity_k{a.k}",
        )

    # Reuse overlap (uses same backend as requested)
    pair_dfs = []
    for m in methods:
        dfp = compute_reuse_overlap(
            m,
            go_map,
            k=int(a.k),
            backend=backend,
            seed=int(a.seed) + 999,
            n_queries=int(a.reuse_n_queries),
            n_pairs=int(a.reuse_n_pairs),
            exclude_same_protein=bool(a.exclude_same_protein),
            require_query_has_go=bool(a.require_query_has_go),
            pair_mode=str(a.pair_mode),
            # bruteforce
            knn_chunk=int(a.knn_chunk),
            # faiss
            faiss_exact=bool(faiss_exact),
            faiss_use_gpu=bool(a.faiss_use_gpu),
            faiss_gpu_device=int(a.faiss_gpu_device),
            faiss_ivf_nlist=int(a.faiss_nlist),
            faiss_ivf_nprobe=int(a.faiss_nprobe),
        )
        pair_dfs.append(dfp)

    df_pairs_all = pd.concat(pair_dfs, ignore_index=True) if pair_dfs else pd.DataFrame()
    pairs_csv = out_dir / f"reuse_pairs_k{a.k}_{a.pair_mode}.csv"
    df_pairs_all.to_csv(pairs_csv, index=False)

    if not df_pairs_all.empty:
        plot_violin_box(
            df_pairs_all,
            ycol="jaccard_overlap",
            ylabel="Jaccard overlap of neighbor protein sets",
            title=f"Neighborhood reuse overlap @ k={a.k} ({a.pair_mode})",
            out_base=out_dir / f"reuse_overlap_k{a.k}_{a.pair_mode}",
        )
    # ----------------------------
    # Reuse@k + MRR@k + log2FE(reuse overlap)
    # ----------------------------
    # Compute kNN once per method for retrieval metrics
    knn_by_method: Dict[str, np.ndarray] = {}
    for m in methods:
        knn_by_method[m.name] = _compute_knn_idx_for_method(
            m,
            backend=backend,
            k=int(a.k),
            knn_chunk=int(a.knn_chunk),
            faiss_exact=bool(faiss_exact),
            faiss_use_gpu=bool(a.faiss_use_gpu),
            faiss_gpu_device=int(a.faiss_gpu_device),
            faiss_ivf_nlist=int(a.faiss_nlist),
            faiss_ivf_nprobe=int(a.faiss_nprobe),
        )

    baseline_name = str(a.reuse_baseline_method)
    have_baseline = baseline_name in knn_by_method

    summary_rows = []
    for m in methods:
        # Retrieval metrics
        retr = compute_reuse_retrieval_metrics(
            m,
            go_map,
            knn_idx=knn_by_method[m.name],
            k=int(a.k),
            sample_queries=int(a.sample_queries),
            seed=int(a.seed),
            exclude_same_protein=bool(a.exclude_same_protein),
            require_query_has_go=bool(a.require_query_has_go),
        )

        # Reuse overlap FE 
        df_m = df_pairs_all[df_pairs_all["method"] == m.name].copy()

        if have_baseline and m.name != baseline_name:
            df_b = df_pairs_all[df_pairs_all["method"] == baseline_name].copy()
        else:
            # Fast shuffle-null fallback (still yields a log2FE; but best is to pass control_shuffled_embeddings)
            df_b = _make_fast_shuffled_baseline_pairs(df_m, seed=int(a.seed) + 4242)

        fe = compute_reuse_log2_fold_enrichment(df_m, df_b)

        summary = {
            "method": m.name,
            "go_aspect": str(a.aspect),
            "k_neighbors": int(a.k),
            "exclude_same_protein": bool(a.exclude_same_protein),
            "require_query_has_go": bool(a.require_query_has_go),
            **retr,
            **fe,
        }
        summary_rows.append(summary)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"reuse_retrieval_summary_k{a.k}.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[OK] Wrote: {summary_csv}")

    print(f"[OK] Backend: {backend}")
    if backend == "faiss":
        print(f"[OK] FAISS mode: {'exact(IndexFlatIP)' if faiss_exact else f'IVF(IndexIVFFlat) nlist={a.faiss_nlist} nprobe={a.faiss_nprobe}'}")
    print(f"[OK] Wrote: {metrics_csv}")
    print(f"[OK] Wrote: {pairs_csv}")
    print(f"[OK] Plots in: {out_dir}")


if __name__ == "__main__":
    main()
