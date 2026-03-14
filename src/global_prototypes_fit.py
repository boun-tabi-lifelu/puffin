#!/usr/bin/env python
"""
src/global_prototypes_splitfit.py
---------------------------------
Train global prototypes on TRAIN segment embeddings, then assign VALID/TEST segments
to nearest train prototype. Sweeps K over a list and produces publication-ready
stats + plots.

Expected directory structure:
  segments/<model_name>/<split>/
    train_segment_embeddings.npy / .pt
    train_segment_metadata.csv
    valid_...
    test_...

Main outputs (per model, per K):
  <out_root>/<model_name>/K<k>/
    train_centroids.npy
    debias_transform.json
    assignments_train.csv
    assignments_valid.csv
    assignments_test.csv
    summary.json
    eval/
      go_enrichment_train_top.csv
      go_enrichment_valid_top.csv
      go_enrichment_test_top.csv
      prototype_knn_go_retrieval_train.json
      prototype_knn_go_retrieval_valid.json
      prototype_knn_go_retrieval_test.json
      stability_train.json
    plots/ (pdf + png)

Also produces (per model):
  <out_root>/<model_name>/k_sweep_metrics.csv
  <out_root>/<model_name>/k_sweep_plots/*.pdf + *.png
"""

import argparse
import json
import math
import re
import torch
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# -------------------------
# IO utils
# -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}. Found: {list(df.columns)}")


# def load_segment_outputs(cluster_dir: Path, prefix: str) -> Tuple[np.ndarray, pd.DataFrame]:
#     seg_meta_path = cluster_dir / f"{prefix}_segment_metadata.csv"
#     if not seg_meta_path.exists():
#         raise FileNotFoundError(f"Missing segment metadata: {seg_meta_path}")
#     seg_meta = pd.read_csv(seg_meta_path)
#     require_columns(seg_meta, ["global_seg_index", "pdb_id", "segment_k", "n_residues_assigned"], "segment_metadata")

#     npy_path = cluster_dir / f"{prefix}_segment_embeddings.npy"
#     pt_path = cluster_dir / f"{prefix}_segment_embeddings.pt"

#     if npy_path.exists():
#         E = np.load(npy_path).astype("float32")
#     elif pt_path.exists():
#         import torch  # local
#         obj: Any = torch.load(pt_path, map_location="cpu")
#         if not (isinstance(obj, dict) and "embeddings" in obj):
#             raise ValueError(f"{pt_path} must be a dict with key 'embeddings'")
#         E = obj["embeddings"].detach().cpu().numpy().astype("float32")
#     else:
#         raise FileNotFoundError(f"Missing embeddings file: {npy_path} or {pt_path}")

#     if len(seg_meta) != E.shape[0]:
#         raise ValueError(f"Mismatch: seg_meta rows={len(seg_meta)} vs embeddings rows={E.shape[0]}")

#     return E, seg_meta

def load_segment_outputs(
    cluster_dir: Path,
    prefix: str,
    *,
    as_torch: bool = False,
    device: str = "cpu",
) -> Tuple[Union[np.ndarray, "torch.Tensor"], pd.DataFrame]:
    seg_meta_path = cluster_dir / f"{prefix}_segment_metadata.csv"
    if not seg_meta_path.exists():
        raise FileNotFoundError(f"Missing segment metadata: {seg_meta_path}")
    seg_meta = pd.read_csv(seg_meta_path)
    require_columns(seg_meta, ["global_seg_index", "pdb_id", "segment_k", "n_residues_assigned"], "segment_metadata")

    npy_path = cluster_dir / f"{prefix}_segment_embeddings.npy"
    pt_path = cluster_dir / f"{prefix}_segment_embeddings.pt"

    if as_torch:
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

        if len(seg_meta) != int(E_t.shape[0]):
            raise ValueError(f"Mismatch: seg_meta rows={len(seg_meta)} vs embeddings rows={int(E_t.shape[0])}")

        return E_t, seg_meta

    if npy_path.exists():
        E = np.load(npy_path).astype("float32")
    elif pt_path.exists():
        if torch is None:
            raise ImportError("torch not available to load .pt embeddings.")
        obj: Any = torch.load(pt_path, map_location="cpu")
        if not (isinstance(obj, dict) and "embeddings" in obj):
            raise ValueError(f"{pt_path} must be a dict with key 'embeddings'")
        E = obj["embeddings"].detach().cpu().numpy().astype("float32")
    else:
        raise FileNotFoundError(f"Missing embeddings file: {npy_path} or {pt_path}")

    if len(seg_meta) != E.shape[0]:
        raise ValueError(f"Mismatch: seg_meta rows={len(seg_meta)} vs embeddings rows={E.shape[0]}")

    return E, seg_meta


def load_go_annotations(annotation_dir: Path, go_file: str = "nrPDB-GO_annot.tsv") -> pd.DataFrame:
    """
    Uses your earlier format assumption:
      - skiprows=12 in your newer code, but your older file used skiprows=14.
    We'll auto-detect by trying 12 then 14.
    Result indexed by legacy ID (often like '1AD3-A').
    Columns: MF,BP,CC contain comma-joined strings (we'll split later).
    """
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


# -------------------------
# ID helpers
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
# math / transforms
# -------------------------
def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)

def l2_normalize_torch(X: "torch.Tensor", eps: float = 1e-12) -> "torch.Tensor":
    n = torch.linalg.norm(X, dim=1, keepdim=True)
    return X / torch.clamp(n, min=eps)



@dataclass
class DebiasTransform:
    """
    Fit on TRAIN only; apply to val/test.
    Steps:
      1) l2 normalize
      2) subtract mean (mu) computed on train normalized
      3) project out top PCs computed on centered train
      4) l2 normalize again
    """
    mu: np.ndarray                 # (1,H)
    pcs: np.ndarray                # (r,H)  r=remove_pcs

    def apply(self, E: np.ndarray) -> np.ndarray:
        X = l2_normalize(E)
        X = X - self.mu
        if self.pcs.size > 0:
            for r in range(self.pcs.shape[0]):
                c = self.pcs[r].reshape(1, -1)
                X = X - (X @ c.T) * c
        X = l2_normalize(X)
        return X

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "mu": self.mu.astype(float).ravel().tolist(),
            "pcs": self.pcs.astype(float).tolist(),
            "remove_pcs": int(self.pcs.shape[0]),
        }

    @staticmethod
    def fit(E_train: np.ndarray, remove_pcs: int, seed: int = 0) -> "DebiasTransform":
        X = l2_normalize(E_train)
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        pcs = np.zeros((0, X.shape[1]), dtype=np.float32)
        if remove_pcs > 0:
            pca = PCA(n_components=remove_pcs, random_state=seed)
            pca.fit(Xc)
            pcs = pca.components_.astype(np.float32)
        return DebiasTransform(mu=mu.astype(np.float32), pcs=pcs)


@dataclass
class DebiasTransformTorch:
    mu: "torch.Tensor"   # (1,H) on device
    pcs: "torch.Tensor"  # (r,H) on device (r can be 0)

    def apply(self, E: "torch.Tensor") -> "torch.Tensor":
        X = l2_normalize_torch(E)
        X = X - self.mu
        if self.pcs.numel() > 0:
            # project out each PC direction
            # X <- X - (X·c) c
            for r in range(self.pcs.shape[0]):
                c = self.pcs[r].view(1, -1)  # (1,H)
                X = X - (X @ c.t()) * c
        X = l2_normalize_torch(X)
        return X

    def to_jsonable(self) -> Dict[str, Any]:
        # Save to CPU-friendly lists
        mu = self.mu.detach().float().cpu().numpy().ravel().tolist()
        pcs = self.pcs.detach().float().cpu().numpy().tolist()
        return {
            "mu": mu,
            "pcs": pcs,
            "remove_pcs": int(self.pcs.shape[0]),
        }

    @staticmethod
    def fit(E_train: "torch.Tensor", remove_pcs: int, seed: int = 0) -> "DebiasTransformTorch":
        """
        PCA is computed with torch.linalg.svd on centered, normalized train embeddings.
        remove_pcs=0 allowed.
        """
        X = l2_normalize_torch(E_train)
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu

        if remove_pcs <= 0:
            pcs = Xc.new_zeros((0, Xc.shape[1]))
            return DebiasTransformTorch(mu=mu, pcs=pcs)

        # For determinism you can set torch seed (optional)
        torch.manual_seed(int(seed))

        # SVD-based top PCs: Xc = U S Vh; PCs are rows of Vh
        # Vh shape (H,H) if full_matrices=False and M>=H; otherwise min(M,H)
        # We want top 'remove_pcs' right-singular vectors.
        # Note: for very large M, this is still heavy but usually OK at H~512..1024.
        # If M is enormous, consider random projection or sklearn PCA on CPU.
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        pcs = Vh[:remove_pcs].contiguous()  # (r,H)

        return DebiasTransformTorch(mu=mu, pcs=pcs)

# -------------------------
# filtering / sampling
# -------------------------

def filter_segments(
    E: Union[np.ndarray, "torch.Tensor"],
    meta: pd.DataFrame,
    *,
    min_assigned: int,
    max_assigned: Optional[int],
) -> Tuple[Union[np.ndarray, "torch.Tensor"], pd.DataFrame]:
    n_assigned = pd.to_numeric(meta["n_residues_assigned"], errors="coerce").fillna(0).astype(int)
    mask = n_assigned >= int(min_assigned)
    if max_assigned is not None:
        mask &= n_assigned <= int(max_assigned)
    kept = np.where(mask.to_numpy())[0]
    if kept.size == 0:
        raise ValueError(f"No segments kept after filter min_assigned={min_assigned}, max_assigned={max_assigned}")
    if isinstance(E, torch.Tensor):
        kept_t = torch.as_tensor(kept, device=E.device, dtype=torch.long)
        return E.index_select(0, kept_t), meta.iloc[kept].copy().reset_index(drop=True)
    return E[kept], meta.iloc[kept].copy().reset_index(drop=True)



def add_id_columns(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    m["pdb"] = m["pdb_id"].astype(str).apply(lambda x: split_id(x)[0])
    m["chain"] = m["pdb_id"].astype(str).apply(lambda x: split_id(x)[1])
    m["legacy_id"] = m["pdb_id"].astype(str).apply(legacy_id_from_new)
    # protein key for dedup in enrichment: legacy id is fine (pdb-chain)
    m["protein_key"] = m["legacy_id"].astype(str)
    return m


def sample_max_per_protein(
    E: Union[np.ndarray, "torch.Tensor"],
    meta: pd.DataFrame,
    *,
    max_per_protein: int,
    seed: int,
    protein_key_col: str = "protein_key",
) -> Tuple[Union[np.ndarray, "torch.Tensor"], pd.DataFrame]:
    if max_per_protein <= 0:
        return E, meta
    rng = np.random.default_rng(seed)
    rows = []
    for _, dfp in meta.groupby(protein_key_col, sort=False):
        idxs = dfp.index.to_numpy()
        if len(idxs) > max_per_protein:
            idxs = rng.choice(idxs, size=max_per_protein, replace=False)
        rows.append(idxs)
    sel = np.sort(np.concatenate(rows)) if rows else np.array([], dtype=int)
    if sel.size == 0:
        raise ValueError("Sampling produced empty selection.")
    if isinstance(E, torch.Tensor):
        sel_t = torch.as_tensor(sel, device=E.device, dtype=torch.long)
        return E.index_select(0, sel_t), meta.iloc[sel].copy().reset_index(drop=True)
    return E[sel], meta.iloc[sel].copy().reset_index(drop=True)



# -------------------------
# spherical k-means
# -------------------------
def spherical_kmeans(
    X: np.ndarray,
    k: int,
    *,
    iters: int = 40,
    n_init: int = 5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X must be L2-normalized.
    Returns labels (M,), centroids (k,H) L2-normalized.
    """
    rng = np.random.default_rng(seed)
    M, H = X.shape
    k = int(min(k, M))
    best_inertia = float("inf")
    best_labels: Optional[np.ndarray] = None
    best_C: Optional[np.ndarray] = None

    for init in range(n_init):
        idx = rng.choice(M, size=k, replace=False)
        C = X[idx].copy()

        for _ in range(iters):
            sims = X @ C.T
            labels = np.argmax(sims, axis=1)

            C_new = np.zeros((k, H), dtype=np.float32)
            counts = np.zeros((k,), dtype=np.int64)
            for i in range(M):
                j = int(labels[i])
                C_new[j] += X[i]
                counts[j] += 1

            for j in range(k):
                if counts[j] == 0:
                    C_new[j] = X[rng.integers(0, M)]
                else:
                    C_new[j] /= float(counts[j])

            C = l2_normalize(C_new)

        sims = X @ C.T
        max_sim = sims[np.arange(M), np.argmax(sims, axis=1)]
        inertia = float(np.sum(1.0 - max_sim))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = np.argmax(sims, axis=1).astype(int)
            best_C = C

    assert best_labels is not None and best_C is not None
    return best_labels, best_C

def spherical_kmeans_torch(
    X: "torch.Tensor",
    k: int,
    *,
    iters: int = 40,
    n_init: int = 5,
    seed: int = 0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    GPU spherical k-means.
    X: (M,H) torch float32 on device, assumed L2-normalized.
    Returns:
      labels: (M,) int64 on device
      centroids: (k,H) float32 on device, L2-normalized
    """
    if torch is None:
        raise ImportError("torch not available for spherical_kmeans_torch.")

    device = X.device
    M, H = X.shape
    k = int(min(k, M))
    if k <= 1:
        labels = torch.zeros((M,), dtype=torch.long, device=device)
        C = l2_normalize_torch(X[:1].clone())
        return labels, C

    best_inertia = None
    best_labels = None
    best_C = None

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    for init in range(int(n_init)):
        # init centroids by random points
        idx = torch.randperm(M, generator=g, device=device)[:k]
        C = X[idx].clone()  # already normalized if X is normalized

        for _ in range(int(iters)):
            sims = X @ C.t()              # (M,k)
            labels = torch.argmax(sims, dim=1)  # (M,)

            # accumulate sums per cluster
            C_new = torch.zeros((k, H), dtype=X.dtype, device=device)
            C_new.index_add_(0, labels, X)

            # counts per cluster
            counts = torch.bincount(labels, minlength=k).to(dtype=X.dtype).view(k, 1)

            # handle empty clusters
            empty = (counts.view(-1) == 0)
            if empty.any():
                # re-seed empties with random points
                reseed = torch.randperm(M, generator=g, device=device)[:int(empty.sum().item())]
                C_new[empty] = X[reseed]
                counts[empty] = 1.0

            C = l2_normalize_torch(C_new / counts)

        # inertia = sum(1 - max cosine sim)
        sims = X @ C.t()
        max_sim, _ = torch.max(sims, dim=1)
        inertia = torch.sum(1.0 - max_sim)

        if best_inertia is None or inertia.item() < float(best_inertia):
            best_inertia = inertia.item()
            best_labels = torch.argmax(sims, dim=1).to(torch.long)
            best_C = C.clone()

    assert best_labels is not None and best_C is not None
    return best_labels, best_C


# -------------------------
# assignment: nearest centroid (cosine)
# -------------------------
def assign_to_centroids(X: np.ndarray, C: np.ndarray, *, chunk: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      proto: (M,) int
      sim:   (M,) float  cosine similarity to chosen centroid
    """
    M = X.shape[0]
    proto = np.empty((M,), dtype=np.int32)
    sim = np.empty((M,), dtype=np.float32)

    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        S = X[s:e] @ C.T
        p = np.argmax(S, axis=1).astype(np.int32)
        proto[s:e] = p
        sim[s:e] = S[np.arange(e - s), p].astype(np.float32)

    return proto, sim

def assign_to_centroids_torch(
    X: "torch.Tensor",
    C: "torch.Tensor",
    *,
    chunk: int = 20000,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    X: (M,H) torch float32 (L2-normalized)
    C: (K,H) torch float32 (L2-normalized)
    Returns:
      proto: (M,) int64
      sim:   (M,) float32
    """
    M = X.shape[0]
    proto = torch.empty((M,), dtype=torch.long, device=X.device)
    sim = torch.empty((M,), dtype=torch.float32, device=X.device)

    for s in range(0, M, int(chunk)):
        e = min(M, s + int(chunk))
        S = X[s:e] @ C.t()
        p = torch.argmax(S, dim=1)
        proto[s:e] = p
        sim[s:e] = S[torch.arange(e - s, device=X.device), p].to(torch.float32)

    return proto, sim



# -------------------------
# evaluation
# -------------------------
def shannon_entropy(counts: np.ndarray, eps: float = 1e-12) -> float:
    s = counts.sum()
    if s <= 0:
        return float("nan")
    p = counts / max(eps, s)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def protein_entropy_summary(assign_df: pd.DataFrame) -> Dict[str, float]:
    # entropy over proteins within each prototype
    entropies = []
    entropies_w = []
    sizes = []

    for _, d in assign_df.groupby("proto"):
        vc = d["protein_key"].value_counts().to_numpy().astype(float)
        H = shannon_entropy(vc)
        entropies.append(H)
        sizes.append(len(d))
        entropies_w.append(H * len(d))

    total = float(np.sum(sizes)) if sizes else 1.0
    return {
        "n_segments": float(len(assign_df)),
        "n_prototypes_observed": float(assign_df["proto"].nunique()),
        "mean_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "mean_entropy_weighted": float(np.sum(entropies_w) / total) if entropies_w else float("nan"),
        "mean_effective_proteins_weighted": float(
            np.sum([math.exp(h) * n for h, n in zip(entropies, sizes)]) / total
        ) if entropies else float("nan"),
    }


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def go_enrichment_fisher(
    assign_df: pd.DataFrame,
    go_df: pd.DataFrame,
    *,
    go_aspect: str,
    min_proteins_per_proto: int = 10,
    min_term_proteins: int = 10,
    top_terms_per_proto: int = 10,
    qval_thresh: float = 0.05,
    out_csv_all: Optional[Path] = None,
    out_csv_top: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Protein-level Fisher enrichment per prototype, within this split only.
    Background = proteins appearing in assign_df.
    GO keys: go_df indexed by legacy id (e.g., 1AD3-A). Our protein_key is legacy id.
    """
    go_aspect = go_aspect.upper()
    if go_aspect not in ("MF", "BP", "CC"):
        raise ValueError("go_aspect must be MF/BP/CC")

    prots_all = sorted(assign_df["protein_key"].astype(str).unique().tolist())
    total_prots = len(prots_all)
    go_col = go_df[go_aspect]

    prot_go: Dict[str, set] = {}
    for p in prots_all:
        if p in go_col.index:
            prot_go[p] = set(safe_go_list(go_col.loc[p]))
        else:
            prot_go[p] = set()

    term_to_prots: Dict[str, set] = defaultdict(set)
    for p in prots_all:
        for t in prot_go[p]:
            term_to_prots[t].add(p)

    terms = [t for t, s in term_to_prots.items() if len(s) >= min_term_proteins]
    terms = sorted(terms)

    proto_to_prots: Dict[int, set] = (
        assign_df.groupby("proto")["protein_key"]
        .apply(lambda x: set(x.astype(str).tolist()))
        .to_dict()
    )

    results = []
    for proto, prots_in in proto_to_prots.items():
        if len(prots_in) < min_proteins_per_proto:
            continue
        prots_out = set(prots_all) - prots_in
        n_in = len(prots_in)
        n_out = len(prots_out)

        for t in terms:
            has_t = term_to_prots[t]
            a = len(prots_in & has_t)
            if a == 0:
                continue
            b = n_in - a
            c_ = len(prots_out & has_t)
            d = n_out - c_
            _, pval = fisher_exact([[a, b], [c_, d]], alternative="greater")
            odds = float((a * d) / max(1.0, (b * c_))) if (b * c_) > 0 else float("inf")
            results.append(
                {
                    "proto": int(proto),
                    "go_term": t,
                    "a_in_has": int(a),
                    "b_in_not": int(b),
                    "c_out_has": int(c_),
                    "d_out_not": int(d),
                    "proto_proteins": int(n_in),
                    "total_proteins": int(total_prots),
                    "pval": float(pval),
                    "odds_ratio_approx": odds,
                }
            )

    if not results:
        empty = pd.DataFrame(columns=["proto", "go_term", "pval", "qval"])
        stats = {
            "tested_terms": float(len(terms)),
            "tested_prototypes": float(len(proto_to_prots)),
            "significant_pairs_q<0.05": 0.0,
            "significant_prototypes_q<0.05": 0.0,
        }
        if out_csv_all:
            empty.to_csv(out_csv_all, index=False)
        if out_csv_top:
            empty.to_csv(out_csv_top, index=False)
        return empty, stats

    out = pd.DataFrame(results)
    out["qval"] = benjamini_hochberg(out["pval"].to_numpy())
    out = out.sort_values(["proto", "qval", "pval", "odds_ratio_approx"], ascending=[True, True, True, False])

    top = out.groupby("proto").head(top_terms_per_proto).reset_index(drop=True)

    sig = out[out["qval"] < float(qval_thresh)]
    stats = {
        "tested_terms": float(len(terms)),
        "tested_prototypes": float(len(out["proto"].unique())),
        "significant_pairs_q<0.05": float(len(sig)),
        "significant_prototypes_q<0.05": float(sig["proto"].nunique()),
    }

    if out_csv_all:
        out.to_csv(out_csv_all, index=False)
    if out_csv_top:
        top.to_csv(out_csv_top, index=False)

    return top, stats


def centroid_knn(C: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    Cn = l2_normalize(C.astype("float32"))
    S = Cn @ Cn.T
    np.fill_diagonal(S, -1e9)
    K = Cn.shape[0]
    k = min(k, K - 1)
    idx = np.argpartition(-S, kth=np.arange(k), axis=1)[:, :k]
    row = np.arange(K)[:, None]
    sim = S[row, idx]
    order = np.argsort(-sim, axis=1)
    idx = idx[row, order]
    sim = sim[row, order]
    return idx.astype(np.int64), sim.astype(np.float32)


def build_proto_go_sets(assign_df: pd.DataFrame, go_df: pd.DataFrame, go_aspect: str) -> Tuple[Dict[int, set], Dict[int, set]]:
    go_aspect = go_aspect.upper()
    go_col = go_df[go_aspect]

    proto_prots: Dict[int, set] = defaultdict(set)
    for p, proto in zip(assign_df["protein_key"].astype(str), assign_df["proto"].astype(int)):
        proto_prots[int(proto)].add(p)

    proto_go: Dict[int, set] = {}
    for proto, prots in proto_prots.items():
        gos = set()
        for p in prots:
            if p in go_col.index:
                gos.update(safe_go_list(go_col.loc[p]))
        proto_go[int(proto)] = gos

    return proto_go, proto_prots


def proto_go_retrieval_at_k(
    knn_idx: np.ndarray,
    proto_go: Dict[int, set],
    *,
    k: int,
    min_go_terms: int = 1,
    exclude_shared_proteins: bool = False,
    proto_prots: Optional[Dict[int, set]] = None,
) -> Dict[str, float]:
    K = knn_idx.shape[0]
    k = min(k, knn_idx.shape[1])

    hits, precs, jaccs = [], [], []
    queries_used = 0
    eff_ks = []

    for q in range(K):
        q_gos = proto_go.get(q, set())
        if len(q_gos) < min_go_terms:
            continue

        denom = 0
        tp = 0
        jac_sum = 0.0

        for n in knn_idx[q, :k].tolist():
            n = int(n)
            n_gos = proto_go.get(n, set())
            if len(n_gos) < min_go_terms:
                continue

            if exclude_shared_proteins:
                if proto_prots is None:
                    raise ValueError("exclude_shared_proteins=True requires proto_prots")
                if len(proto_prots.get(q, set()) & proto_prots.get(n, set())) > 0:
                    continue

            inter = len(q_gos & n_gos)
            union = len(q_gos | n_gos)
            jac = (inter / union) if union > 0 else 0.0

            denom += 1
            jac_sum += jac
            if inter > 0:
                tp += 1

        if denom == 0:
            continue
        queries_used += 1
        eff_ks.append(denom)
        hits.append(1.0 if tp > 0 else 0.0)
        precs.append(tp / float(denom))
        jaccs.append(jac_sum / float(denom))

    if queries_used == 0:
        return {"k_target": float(k), "queries_used": 0.0, "note": "no prototypes with enough GO"}

    return {
        "k_target": float(k),
        "queries_used": float(queries_used),
        "mean_effective_k": float(np.mean(eff_ks)) if eff_ks else 0.0,
        "hit_at_k": float(np.mean(hits)),
        "precision_at_k": float(np.mean(precs)),
        "mean_jaccard_at_k": float(np.mean(jaccs)),
    }


# -------------------------
# stability (TRAIN only)
# -------------------------
def stability_under_resampling_train(
    E_train_all: np.ndarray,
    meta_train_all: pd.DataFrame,
    *,
    k: int,
    min_assigned: int,
    max_assigned: Optional[int],
    max_per_protein: int,
    remove_pcs: int,
    runs: int,
    frac_proteins: float,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    meta = add_id_columns(meta_train_all)
    proteins = sorted(meta["protein_key"].unique().tolist())
    print(proteins[:5])
    print(f"Stability: total proteins in train = {len(proteins)}")
    n_keep = max(2, int(round(frac_proteins * len(proteins))))
    run_maps: List[Dict[int, int]] = []

    for r in range(runs):
        keep = set(rng.choice(proteins, size=n_keep, replace=False).tolist())
        sel = meta["protein_key"].isin(keep).to_numpy()
        E_r = E_train_all[sel]
        meta_r = meta.iloc[np.where(sel)[0]].copy().reset_index(drop=True)

        E_rf, meta_rf = filter_segments(E_r, meta_r, min_assigned=min_assigned, max_assigned=max_assigned)
        E_rs, meta_rs = sample_max_per_protein(E_rf, meta_rf, max_per_protein=max_per_protein, seed=seed + 1000 + r)

        # deb = DebiasTransform.fit(E_rs, remove_pcs=remove_pcs, seed=seed + 2000 + r)
        # Xr = deb.apply(E_rs)

        # y, _C = spherical_kmeans(Xr, k=k, iters=30, n_init=3, seed=seed + 3000 + r)      
        deb = DebiasTransformTorch.fit(E_rs, remove_pcs=remove_pcs, seed=seed + 2000 + r)
        Xr = deb.apply(E_rs)

        y, _C = spherical_kmeans_torch(Xr, k=k, iters=30, n_init=3, seed=seed + 3000 + r)
        gids = meta_rs["global_seg_index"].astype(int).to_numpy()
        run_maps.append({int(g): int(lbl) for g, lbl in zip(gids, y)})

    aris = []
    for i in range(len(run_maps)):
        for j in range(i + 1, len(run_maps)):
            common = sorted(set(run_maps[i].keys()) & set(run_maps[j].keys()))
            if len(common) < 200:
                continue
            a = np.array([run_maps[i][g] for g in common], dtype=int)
            b = np.array([run_maps[j][g] for g in common], dtype=int)
            aris.append(adjusted_rand_score(a, b))

    if not aris:
        return {"runs": float(runs), "pairs_used": 0.0, "ari_mean": float("nan"), "ari_std": float("nan")}

    return {
        "runs": float(runs),
        "pairs_used": float(len(aris)),
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
    }


# -------------------------
# plotting (publication-ready)
# -------------------------
def setup_mpl():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig_dual(fig, out_base: Path):
    import matplotlib.pyplot as plt
    ensure_dir(out_base.parent)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_k_sweep(metrics_df: pd.DataFrame, out_dir: Path, model_name: str):
    import matplotlib.pyplot as plt
    setup_mpl()
    ensure_dir(out_dir)

    # 1) hit@k over K for each split
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    for split in ["train", "valid", "test"]:
        sub = metrics_df[metrics_df["split"] == split].sort_values("K")
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["proto_knn_hit_at_k"], marker="o", label=split)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of prototypes (K)")
    ax.set_ylabel("Prototype centroid GO retrieval hit@k")
    ax.set_title(f"{model_name}: prototype GO retrieval vs K")
    ax.legend(frameon=False)
    savefig_dual(fig, out_dir / "k_sweep_proto_go_hit_at_k")

    # 2) significant prototypes (q<0.05) vs K
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    for split in ["train", "valid", "test"]:
        sub = metrics_df[metrics_df["split"] == split].sort_values("K")
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["enrich_sig_protos"], marker="o", label=split)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of prototypes (K)")
    ax.set_ylabel("# prototypes with ≥1 enriched GO term (q<0.05)")
    ax.set_title(f"{model_name}: GO-enriched prototypes vs K")
    ax.legend(frameon=False)
    savefig_dual(fig, out_dir / "k_sweep_enriched_prototypes")

    # 3) weighted protein entropy vs K
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    for split in ["train", "valid", "test"]:
        sub = metrics_df[metrics_df["split"] == split].sort_values("K")
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["entropy_weighted"], marker="o", label=split)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of prototypes (K)")
    ax.set_ylabel("Weighted protein entropy per prototype")
    ax.set_title(f"{model_name}: prototype protein entropy vs K")
    ax.legend(frameon=False)
    savefig_dual(fig, out_dir / "k_sweep_entropy_weighted")


def plot_support_and_similarity(assign_df: pd.DataFrame, out_dir: Path, title_prefix: str):
    import matplotlib.pyplot as plt
    setup_mpl()
    ensure_dir(out_dir)

    # proteins per prototype
    prot_counts = assign_df.groupby("proto")["protein_key"].nunique().astype(float)
    seg_counts = assign_df.groupby("proto").size().astype(float)

    fig = plt.figure(figsize=(6.5, 3.8))
    ax = fig.add_subplot(111)
    ax.hist(np.log10(np.clip(prot_counts.values, 1.0, None)), bins=40)
    ax.set_xlabel("log10(# proteins per prototype)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}: prototype protein support")
    savefig_dual(fig, out_dir / "hist_log10_proteins_per_proto")

    fig = plt.figure(figsize=(6.5, 3.8))
    ax = fig.add_subplot(111)
    ax.hist(np.log10(np.clip(seg_counts.values, 1.0, None)), bins=40)
    ax.set_xlabel("log10(# segments per prototype)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}: prototype segment support")
    savefig_dual(fig, out_dir / "hist_log10_segments_per_proto")

    # assignment similarity if available
    if "assign_sim" in assign_df.columns:
        fig = plt.figure(figsize=(6.5, 3.8))
        ax = fig.add_subplot(111)
        x = pd.to_numeric(assign_df["assign_sim"], errors="coerce").dropna().astype(float).values
        ax.hist(x, bins=60)
        ax.set_xlabel("Cosine similarity to assigned prototype")
        ax.set_ylabel("Count")
        ax.set_title(f"{title_prefix}: assignment similarity")
        savefig_dual(fig, out_dir / "hist_assignment_similarity")


# -------------------------
# per-K run
# -------------------------
def run_one_k(
    *,
    model_dir: Path,
    out_dir_k: Path,
    annotation_dir: Path,
    go_file: str,
    go_aspect: str,
    K: int,
    min_assigned: int,
    max_assigned: Optional[int],
    max_per_protein_train: int,
    max_per_protein_eval: int,
    remove_pcs: int,
    seed: int,
    proto_knn_k: int,
    proto_min_go_terms: int,
    proto_exclude_shared_proteins: bool,
    enrich_top_terms: int,
    enrich_min_proteins_per_proto: int,
    enrich_min_term_proteins: int,
    enrich_qval: float,
    stability_runs: int,
    stability_frac_proteins: float,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      metrics_rows: list of rows for k_sweep_metrics.csv (one per split)
      summary: nested summary for this K
    """
    ensure_dir(out_dir_k)
    eval_dir = ensure_dir(out_dir_k / "eval")
    plots_dir = ensure_dir(out_dir_k / "plots")
    print(f"Running K={K}, output to {out_dir_k}")
    
    print("Loading GO annotations...")
    go_df = load_go_annotations(annotation_dir, go_file=go_file)

    print("Loading and processing training segments...")
    # ---- load train ----
    train_dir = model_dir / "train"
    E_tr_all, meta_tr_all = load_segment_outputs(train_dir, "train", as_torch=True, device=device)
    meta_tr_all = add_id_columns(meta_tr_all)

    print("Filtering training segments... within assigned residue limits")
    # filter + (train) sample
    E_tr_f, meta_tr_f = filter_segments(E_tr_all, meta_tr_all, min_assigned=min_assigned, max_assigned=max_assigned)
    
    print("Sampling training segments... max per protein")
    E_tr_s, meta_tr_s = sample_max_per_protein(
        E_tr_f, meta_tr_f, max_per_protein=max_per_protein_train, seed=seed, protein_key_col="protein_key"
    )

    print("Fitting debias transform on training segments...")
    # fit debias on sampled train, apply to sampled train
    # deb = DebiasTransform.fit(E_tr_s, remove_pcs=remove_pcs, seed=seed)
    # X_tr = deb.apply(E_tr_s)
    deb = DebiasTransformTorch.fit(E_tr_s, remove_pcs=remove_pcs, seed=seed)
    X_tr = deb.apply(E_tr_s)

    print("Clustering training segments to obtain prototypes...")
    # train prototypes
    # y_tr, C = spherical_kmeans(X_tr, k=K, iters=40, n_init=5, seed=seed)
    # C = l2_normalize(C.astype("float32"))
    y_tr, C = spherical_kmeans_torch(X_tr, k=K, iters=40, n_init=5, seed=seed)
    C = l2_normalize_torch(C).to(torch.float32)

    # np.save(out_dir_k / "train_centroids.npy", C.astype("float32"))
    # (out_dir_k / "debias_transform.json").write_text(json.dumps(deb.to_jsonable(), indent=2))
    np.save(out_dir_k / "train_centroids.npy", C.detach().cpu().numpy().astype("float32"))
    (out_dir_k / "debias_transform.json").write_text(json.dumps(deb.to_jsonable(), indent=2))


    print("Assigning training segments to prototypes...")
    # assign train (sampled) with similarities
    # proto_tr, sim_tr = assign_to_centroids(X_tr, C)
    # assert np.all(proto_tr == y_tr)
    proto_tr, sim_tr = assign_to_centroids_torch(X_tr, C)
    # same labels (up to exact centroid matching)
    # assert torch.all(proto_tr == y_tr).item()


    print("Saving training assignments...")
    assign_tr = meta_tr_s[
        ["global_seg_index", "pdb_id", "legacy_id", "protein_key", "segment_k", "n_residues_assigned"]
    ].copy()
    # assign_tr["proto"] = proto_tr.astype(int)
    # assign_tr["assign_sim"] = sim_tr.astype(float)
    assign_tr["proto"] = proto_tr.detach().cpu().numpy().astype(int)
    assign_tr["assign_sim"] = sim_tr.detach().cpu().numpy().astype(float)

    assign_tr.to_csv(out_dir_k / "assignments_train.csv", index=False)

    # per-split evaluation helper
    def eval_split(split: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        print(f"Evaluating split: {split}...")
        if split == "train":
            df = assign_tr.copy()
        else:
            d = model_dir / split
            print(f"Loading and processing {split} segments...")
            E, meta = load_segment_outputs(d, split, as_torch=True, device=device)
            meta = add_id_columns(meta)
            E_f, meta_f = filter_segments(E, meta, min_assigned=min_assigned, max_assigned=max_assigned)

            # optional cap per protein for eval stability/compute
            if max_per_protein_eval > 0:
                print(f"Sampling {split} segments... max per protein")
                E_f, meta_f = sample_max_per_protein(
                    E_f, meta_f, max_per_protein=max_per_protein_eval, seed=seed + 777, protein_key_col="protein_key"
                )

            print(f"Applying debias transform and assigning {split} segments to prototypes...")
            X = deb.apply(E_f)  # apply train-fitted debias
            print(f"Assigning {split} segments to prototypes...")
            # proto, sim = assign_to_centroids(X, C)
            proto, sim = assign_to_centroids_torch(X, C)
            df = meta_f[["global_seg_index", "pdb_id", "legacy_id", "protein_key", "segment_k", "n_residues_assigned"]].copy()
            # df["proto"] = proto.astype(int)
            # df["assign_sim"] = sim.astype(float)
            df["proto"] = proto.detach().cpu().numpy().astype(int)
            df["assign_sim"] = sim.detach().cpu().numpy().astype(float)

            df.to_csv(out_dir_k / f"assignments_{split}.csv", index=False)

        # entropy
        ent = protein_entropy_summary(df)
        print(f"{split} entropy summary: {ent}")

        print(f"Performing GO enrichment for {split}...")
        # enrichment
        top, enrich_stats = go_enrichment_fisher(
            assign_df=df,
            go_df=go_df,
            go_aspect=go_aspect,
            min_proteins_per_proto=enrich_min_proteins_per_proto,
            min_term_proteins=enrich_min_term_proteins,
            top_terms_per_proto=enrich_top_terms,
            qval_thresh=enrich_qval,
            out_csv_all=eval_dir / f"go_enrichment_{split}_all.csv",
            out_csv_top=eval_dir / f"go_enrichment_{split}_top.csv",
        )
        
        enrich_stats = {
            "tested_prototypes": float(top["proto"].nunique()) if not top.empty else 0.0,
        }
        
        print(f"Performing prototype GO retrieval evaluation for {split}...")
        # proto knn GO retrieval
        C_cpu = C.detach().cpu().numpy().astype("float32")
        idx, _sim = centroid_knn(C_cpu, k=proto_knn_k)

        # idx, _sim = centroid_knn(C, k=proto_knn_k)
        proto_go, proto_prots = build_proto_go_sets(df, go_df, go_aspect=go_aspect)
        knn_report = proto_go_retrieval_at_k(
            knn_idx=idx,
            proto_go=proto_go,
            k=proto_knn_k,
            min_go_terms=proto_min_go_terms,
            exclude_shared_proteins=bool(proto_exclude_shared_proteins),
            proto_prots=proto_prots if proto_exclude_shared_proteins else None,
        )
        (eval_dir / f"prototype_knn_go_retrieval_{split}.json").write_text(json.dumps(knn_report, indent=2))

        # plots per split
        plot_support_and_similarity(df, plots_dir / split, title_prefix=f"K={K} {split}")

        split_summary = {
            "entropy": ent,
            "enrichment": enrich_stats,
            "proto_knn_go": knn_report,
            "n_segments": float(len(df)),
            "n_proteins": float(df["protein_key"].nunique()),
        }
        return df, split_summary, knn_report

    # evaluate splits
    _df_train, sum_train, knn_train = eval_split("train")
    _df_valid, sum_valid, knn_valid = eval_split("valid") if (model_dir / "valid").exists() else (pd.DataFrame(), {"note": "missing"}, {})
    _df_test, sum_test, knn_test = eval_split("test") if (model_dir / "test").exists() else (pd.DataFrame(), {"note": "missing"}, {})

    print("Computing stability under resampling on training data...")
    # stability on train-all (not just sampled), using only train data (no leakage)
    stab = stability_under_resampling_train(
        E_train_all=E_tr_all,
        meta_train_all=meta_tr_all,
        k=K,
        min_assigned=min_assigned,
        max_assigned=max_assigned,
        max_per_protein=max_per_protein_train,
        remove_pcs=remove_pcs,
        runs=stability_runs,
        frac_proteins=stability_frac_proteins,
        seed=seed,
    )
    (eval_dir / "stability_train.json").write_text(json.dumps(stab, indent=2))

    # overall per-K summary
    summary = {
        "model_dir": str(model_dir),
        "K": int(K),
        "config": {
            "min_assigned": int(min_assigned),
            "max_assigned": None if max_assigned is None else int(max_assigned),
            "remove_pcs": int(remove_pcs),
            "max_per_protein_train": int(max_per_protein_train),
            "max_per_protein_eval": int(max_per_protein_eval),
            "seed": int(seed),
            "go_aspect": str(go_aspect),
            "proto_knn_k": int(proto_knn_k),
            "proto_min_go_terms": int(proto_min_go_terms),
            "proto_exclude_shared_proteins": bool(proto_exclude_shared_proteins),
            "enrich_top_terms": int(enrich_top_terms),
            "enrich_min_proteins_per_proto": int(enrich_min_proteins_per_proto),
            "enrich_min_term_proteins": int(enrich_min_term_proteins),
            "enrich_qval": float(enrich_qval),
            "stability_runs": int(stability_runs),
            "stability_frac_proteins": float(stability_frac_proteins),
        },
        "train_fit": {
            "n_train_segments_used_for_fit": float(len(assign_tr)),
            "n_train_proteins_used_for_fit": float(assign_tr["protein_key"].nunique()),
        },
        "split_eval": {
            "train": sum_train,
            "valid": sum_valid,
            "test": sum_test,
        },
        "stability_train": stab,
    }
    (out_dir_k / "summary.json").write_text(json.dumps(summary, indent=2))

    # k-sweep metrics rows (one row per split)
    def row(split: str, ssum: Dict[str, Any]) -> Dict[str, Any]:
        ent = ssum.get("entropy", {})
        enr = ssum.get("enrichment", {})
        knn = ssum.get("proto_knn_go", {})
        return {
            "K": int(K),
            "split": split,
            "n_segments": float(ssum.get("n_segments", float("nan"))),
            "n_proteins": float(ssum.get("n_proteins", float("nan"))),
            "entropy_weighted": float(ent.get("mean_entropy_weighted", float("nan"))),
            "entropy_mean": float(ent.get("mean_entropy", float("nan"))),
            "effective_proteins_weighted": float(ent.get("mean_effective_proteins_weighted", float("nan"))),
            "enrich_sig_pairs": float(enr.get("significant_pairs_q<0.05", float("nan"))),
            "enrich_sig_protos": float(enr.get("significant_prototypes_q<0.05", float("nan"))),
            "proto_knn_hit_at_k": float(knn.get("hit_at_k", float("nan"))),
            "proto_knn_precision_at_k": float(knn.get("precision_at_k", float("nan"))),
            "proto_knn_mean_jaccard_at_k": float(knn.get("mean_jaccard_at_k", float("nan"))),
            "stability_ari_mean_train": float(stab.get("ari_mean", float("nan"))) if split == "train" else float("nan"),
        }

    metrics_rows = [row("train", sum_train)]
    if isinstance(sum_valid, dict) and "entropy" in sum_valid:
        metrics_rows.append(row("valid", sum_valid))
    if isinstance(sum_test, dict) and "entropy" in sum_test:
        metrics_rows.append(row("test", sum_test))

    return metrics_rows, summary


# -------------------------
# CLI / orchestration
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_root", type=str, required=True, help="e.g. segments")
    ap.add_argument("--model_name", type=str, required=True, help="child dir under segments_root")
    ap.add_argument("--out_root", type=str, required=True, help="e.g. results/global_prototypes")

    ap.add_argument("--annotation_dir", type=str, required=True)
    ap.add_argument("--go_file", type=str, default="nrPDB-GO_annot.tsv")
    ap.add_argument("--go_aspect", type=str, default="MF", choices=["MF", "BP", "CC"])

    ap.add_argument("--k_list", type=str, default="128,256,512,1024,2048,4096")
    ap.add_argument("--min_assigned", type=int, default=5)
    ap.add_argument("--max_assigned", type=int, default=None)

    ap.add_argument("--max_per_protein_train", type=int, default=50)
    ap.add_argument("--max_per_protein_eval", type=int, default=0, help="0=use all (can be heavy)")
    ap.add_argument("--remove_pcs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--proto_knn_k", type=int, default=20)
    ap.add_argument("--proto_min_go_terms", type=int, default=1)
    ap.add_argument("--proto_exclude_shared_proteins", action="store_true", default=False)

    ap.add_argument("--enrich_top_terms", type=int, default=10)
    ap.add_argument("--enrich_min_proteins_per_proto", type=int, default=10)
    ap.add_argument("--enrich_min_term_proteins", type=int, default=10)
    ap.add_argument("--enrich_qval", type=float, default=0.05)

    ap.add_argument("--stability_runs", type=int, default=8)
    ap.add_argument("--stability_frac_proteins", type=float, default=0.9)

    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda | cuda:0 ...")


    return ap.parse_args()


def main():
    a = parse_args()

    segments_root = Path(a.segments_root)
    model_dir = segments_root / a.model_name
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    out_model = ensure_dir(Path(a.out_root) / a.model_name)
    annotation_dir = Path(a.annotation_dir)

    Ks = [int(x.strip()) for x in str(a.k_list).split(",") if x.strip()]
    all_rows: List[Dict[str, Any]] = []

    summaries: List[Dict[str, Any]] = []
    for K in Ks:
        out_k = ensure_dir(out_model / f"K{K}")
        rows, summary = run_one_k(
            model_dir=model_dir,
            out_dir_k=out_k,
            annotation_dir=annotation_dir,
            go_file=str(a.go_file),
            go_aspect=str(a.go_aspect),
            K=int(K),
            min_assigned=int(a.min_assigned),
            max_assigned=None if a.max_assigned is None else int(a.max_assigned),
            max_per_protein_train=int(a.max_per_protein_train),
            max_per_protein_eval=int(a.max_per_protein_eval),
            remove_pcs=int(a.remove_pcs),
            seed=int(a.seed),
            proto_knn_k=int(a.proto_knn_k),
            proto_min_go_terms=int(a.proto_min_go_terms),
            proto_exclude_shared_proteins=bool(a.proto_exclude_shared_proteins),
            enrich_top_terms=int(a.enrich_top_terms),
            enrich_min_proteins_per_proto=int(a.enrich_min_proteins_per_proto),
            enrich_min_term_proteins=int(a.enrich_min_term_proteins),
            enrich_qval=float(a.enrich_qval),
            stability_runs=int(a.stability_runs),
            stability_frac_proteins=float(a.stability_frac_proteins),
            device=a.device,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    metrics_df = pd.DataFrame(all_rows)
    metrics_csv = out_model / "k_sweep_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    (out_model / "k_sweep_summary.json").write_text(json.dumps(summaries, indent=2))

    # K-sweep plots
    plot_k_sweep(metrics_df, out_model / "k_sweep_plots", model_name=a.model_name)

    print(f"[OK] Wrote: {metrics_csv}")
    print(f"[OK] Wrote: {out_model / 'k_sweep_plots'}")
    print("[DONE]")


if __name__ == "__main__":
    main()
