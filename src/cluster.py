# cluster.py
# -----------------------------------------------------------------------------
# Purpose
#   - Run clustering on protein graphs and persist:
#       (1) per-residue hard assignments
#       (2) per-segment embeddings + metadata (ready for FAISS)
#
# Notes
#   - Protygus encoder output convention:
#       clusters:   list([B, N]) PAD=-1
#       seg_mask:   list([B, K]) bool
#       node_embedding: [B, K, H] segment embeddings
#   - Baselines:
#       Louvain: produces residue labels, segment embeddings via centroid of node features
#       ESM:     produces residue labels, optional per-residue embeddings; segment embeddings via centroid
# -----------------------------------------------------------------------------


import json
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import hydra
import lightning as L
import omegaconf
import pandas as pd
import rootutils
import torch
from loguru import logger as log
from tqdm import tqdm
from torch_geometric.data import Data
from graphein.protein.tensor.dataloader import ProteinDataLoader
from graphein import verbose

from proteinworkshop import register_custom_omegaconf_resolvers
from proteinworkshop.models.graph_encoders.esm_embeddings import EvolutionaryScaleModeling
from src import register_custom_omegaconf_resolvers as src_register_custom_omegaconf_resolvers
from src.utils import extras
from src.utils.model_utils import load_model
from src.models.esm_cluster import ESMCluster
from src.models.faiss_esm_cluster import FaissESMCluster
from src.utils.cluster_utils import find_communities
from src.data.protein_dataset import ProteinDataset
from proteinworkshop.features.factory import ProteinFeaturiser

verbose(False)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PAD = -1


# =========================
# Domain records (FAIR-ish)
# =========================

@dataclass(frozen=True)
class ResidueAssignments:
    pdb_id: str
    n_residues: int
    residue_ids: str
    cluster_ids: str
    K: int


@dataclass(frozen=True)
class SegmentMeta:
    global_seg_index: int
    pdb_id: str
    segment_k: int
    n_residues_assigned: int
    is_active_mask: bool
    H: int


@dataclass(frozen=True)
class RunOutput:
    residue_rows: List[Dict[str, Any]]
    segment_rows: List[Dict[str, Any]]
    segment_embeddings: torch.Tensor  # [M, H]
    run_meta: Dict[str, Any]


# =========================
# Small protocols (SOLID)
# =========================

class BatchClusterer(Protocol):
    """Produces residue-level labels (flat) and optionally residue embeddings aligned with labels."""
    def cluster(self, batch: Any) -> Tuple[Sequence[int], Optional[torch.Tensor]]:
        ...


class SegmentExtractor(Protocol):
    """Extracts residue assignments + segment embeddings from a model output."""
    def extract(self, batch: Any, model_output: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          hard_assign: [B, N] long, PAD=-1
          seg_mask:    [B, K] bool
          seg_emb:     [B, K, H] float
        """
        ...


# =========================
# Utilities
# =========================

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_int_list(x: Any) -> List[int]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [int(v) for v in list(x)]


def _pdb_id(item: Data) -> str:
    return str(getattr(item, "id", "UNKNOWN"))


def _residue_ids(item: Data) -> List[str]:
    if not hasattr(item, "residue_id"):
        raise AttributeError("Item has no attribute 'residue_id'. Check dataset formatting.")
    return list(item.residue_id)


def _safe_parse_residue_numeric(residue_id: str) -> str:
    # Keep original if format differs; do not crash.
    # Typical: "PDB:CHAIN:123"
    parts = str(residue_id).split(":")
    return parts[2] if len(parts) >= 3 else str(residue_id)


def _get_node_embeddings(item: Data, residue_emb_slice: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Returns per-residue embeddings for one item: [N, H]
    Priority:
      1) residue_emb_slice (already aligned)
      2) item.x node features
      3) item.pos CA coords (H=3)
      4) None
    """
    if residue_emb_slice is not None:
        if residue_emb_slice.dim() != 2:
            raise ValueError(f"Expected residue_emb_slice [N,H], got {tuple(residue_emb_slice.shape)}")
        return residue_emb_slice.detach().cpu().float()

    x = getattr(item, "x", None)
    if torch.is_tensor(x) and x.dim() == 2:
        return x.detach().cpu().float()

    pos = getattr(item, "pos", None)
    if torch.is_tensor(pos) and pos.dim() == 2 and pos.size(-1) == 3:
        return pos.detach().cpu().float()

    return None


def _segment_centroids(
    labels: List[int],
    node_emb: Optional[torch.Tensor],
) -> Tuple[List[int], List[int], List[torch.Tensor]]:
    """
    Compute centroids per cluster_id for a single protein.
    Returns:
      cluster_ids_sorted, counts_sorted, centroid_embeddings_sorted
    Fallback:
      if node_emb is None or misaligned -> embedding = [cluster_id] scalar tensor
    """
    valid = [int(c) for c in labels if c != PAD]
    if len(valid) == 0:
        return [], [], []

    from collections import Counter
    counts = Counter(valid)
    cluster_ids = sorted(counts.keys())

    centroids: List[torch.Tensor] = []
    counts_sorted: List[int] = []

    aligned = node_emb is not None and node_emb.size(0) == len(labels)

    for cid in cluster_ids:
        counts_sorted.append(int(counts[cid]))
        if aligned:
            idx = [i for i, c in enumerate(labels) if int(c) == cid]
            emb = node_emb[idx].mean(dim=0) if idx else node_emb.mean(dim=0)
        else:
            emb = torch.tensor([float(cid)], dtype=torch.float32)
        centroids.append(emb.contiguous().float())

    return cluster_ids, counts_sorted, centroids


# =========================
# Model extractor
# =========================

class ModelExtractor:
    @torch.no_grad()
    def extract(self, batch: Any, model_output: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = model_output
        clusters = out["clusters"][0] if isinstance(out.get("clusters"), list) else out["clusters"]
        seg_mask = out["seg_mask"][0] if isinstance(out.get("seg_mask"), list) else out["seg_mask"]
        seg_emb = out["node_embedding"]

        if clusters.dim() != 2:
            raise ValueError(f"Expected clusters [B,N], got {tuple(clusters.shape)}")
        if seg_mask.dim() != 2:
            raise ValueError(f"Expected seg_mask [B,K], got {tuple(seg_mask.shape)}")
        if seg_emb.dim() != 3:
            raise ValueError(f"Expected node_embedding [B,K,H], got {tuple(seg_emb.shape)}")

        return clusters, seg_mask, seg_emb


# =========================
# Baseline clusterers
# =========================

class LouvainClusterer:
    def __init__(self, esm_model_path=None) -> None:
        self._featuriser = ProteinFeaturiser(
            representation="CA",
            scalar_node_features=[],
            vector_node_features=[],
            edge_types=["eps_10"],
            scalar_edge_features=[],
            vector_edge_features=[],
        )
        if not esm_model_path:
            self.esm_model = None
        else:
            model_path = Path(esm_model_path)
            self.esm_model = EvolutionaryScaleModeling(model_path.parent, model=model_path.name, mlp_post_embed=False, finetune=False)
            for param in self.esm_model.model.parameters():
                param.requires_grad = False

            self.esm_model = self.esm_model.cuda()

    @torch.no_grad()
    def cluster(self, batch: Any) -> Tuple[Sequence[int], Optional[torch.Tensor]]:
        processed = self._featuriser(batch)
        labels = find_communities(processed)
        if self.esm_model is None:
            return labels, None

        batch_path = Path(f"data/esm/ESM-1b/{batch.id[0].split('_')[0]}.pt")
        if batch_path.exists():
            embeddings = torch.load(batch_path).cpu()
        else:
            device = next(self.esm_model.parameters()).device
            batch = batch.to(device)
            embeddings = self.esm_model.esm_embed(batch).cpu()
            torch.save(embeddings, batch_path)

        return labels, embeddings


class ESMBaselineClusterer:
    def __init__(self, esm_model_path: str, algorithm: str, device: torch.device) -> None:
        if not esm_model_path:
            raise ValueError("baseline_model.esm_model_path must be set for ESM clustering.")
        # self._model = ESMCluster(esm_model_path, use_gpu=(device.type == "cuda"), cache_dir="data/esm/ESM-1b")
        self._model = FaissESMCluster(esm_model_path, use_gpu=(device.type == "cuda"), cache_dir="data/esm/ESM-1b")
        self._algorithm = algorithm
        self._device = device

    @torch.no_grad()
    def cluster(self, batch: Any) -> Tuple[Sequence[int], Optional[torch.Tensor]]:
        batch_device = batch.to(self._device)
        labels, residue_emb = self._model.cluster(batch_device, self._algorithm)
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return _as_int_list(labels), residue_emb  # residue_emb may be [total_nodes,H] or None


# =========================
# Writers
# =========================

class OutputWriter:
    def __init__(self, output_dir: Path, stem: str) -> None:
        self._dir = output_dir
        self._stem = stem
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, out: RunOutput) -> None:
        residue_csv = self._dir / f"{self._stem}_residue_assignments.csv"
        pd.DataFrame(out.residue_rows).to_csv(residue_csv, index=False)
        log.info(f"Saved residue assignments: {residue_csv}")

        segment_csv = self._dir / f"{self._stem}_segment_metadata.csv"
        pd.DataFrame(out.segment_rows).to_csv(segment_csv, index=False)
        log.info(f"Saved segment metadata: {segment_csv}")

        seg_pt = self._dir / f"{self._stem}_segment_embeddings.pt"
        torch.save({"embeddings": out.segment_embeddings, "meta": out.run_meta}, seg_pt)
        log.info(f"Saved segment embeddings (pt): {seg_pt}")

        try:
            import numpy as np
            seg_npy = self._dir / f"{self._stem}_segment_embeddings.npy"
            np.save(seg_npy, out.segment_embeddings.detach().cpu().numpy().astype("float32"))
            meta_json = self._dir / f"{self._stem}_segment_embeddings.meta.json"
            meta_json.write_text(json.dumps(out.run_meta, indent=2))
            log.info(f"Saved segment embeddings (npy): {seg_npy}")
            log.info(f"Saved embeddings meta (json): {meta_json}")
        except Exception as e:
            log.warning(f"Could not save .npy embeddings: {e}")


# =========================
# Pipelines
# =========================

@torch.no_grad()
def run_baseline(cfg: omegaconf.DictConfig, dataloader: ProteinDataLoader) -> RunOutput:
    baseline_cfg = cfg.baseline_model
    raw_algo = str(getattr(baseline_cfg, "algorithm", "louvain"))
    algo = raw_algo.lower()
    device = _device()

    if algo == "louvain":
        clusterer: BatchClusterer = LouvainClusterer(
            esm_model_path=baseline_cfg.esm_model_path,
        )
    else:
        clusterer = ESMBaselineClusterer(
            esm_model_path=baseline_cfg.esm_model_path,
            algorithm=raw_algo,
            device=device,
        )

    run_meta = {
        "encoder": "baseline",
        "baseline_algorithm": raw_algo,
        "esm_model_path": str(baseline_cfg.get("esm_model_path", "")),
    }

    residue_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []
    seg_emb_list: List[torch.Tensor] = []

    for batch in tqdm(dataloader, desc="Baseline Clustering"):
        batch_cpu = batch.to("cpu")
        labels_flat, residue_emb_all = clusterer.cluster(batch_cpu if algo == "louvain" else batch)

        items = (batch_cpu if algo == "louvain" else batch.to("cpu")).to_data_list()
        node_counts = [len(_residue_ids(it)) for it in items]
        total_nodes = sum(node_counts)

        if total_nodes != len(labels_flat):
            log.warning("Label count (%d) != residue count (%d) for batch %s",
                        len(labels_flat), total_nodes, getattr(batch_cpu, "id", "UNKNOWN"))

        # If residue_emb_all is [total_nodes, H], we slice by the same flat ordering.
        emb_is_flat = torch.is_tensor(residue_emb_all) and residue_emb_all.dim() == 2 and residue_emb_all.size(0) == len(labels_flat)
        emb_cursor = 0

        cursor = 0
        for it, n in zip(items, node_counts):
            if n == 0:
                continue

            sl = labels_flat[cursor: cursor + n]
            cursor += n

            rid = [_safe_parse_residue_numeric(r) for r in _residue_ids(it)]
            pdb = _pdb_id(it)

            residue_rows.append(ResidueAssignments(
                pdb_id=pdb,
                n_residues=len(rid),
                residue_ids=",".join(map(str, rid)),
                cluster_ids=",".join(map(str, sl)),
                K=len(set(sl)),
            ).__dict__)

            residue_emb_slice = None
            if emb_is_flat:
                residue_emb_slice = residue_emb_all[emb_cursor: emb_cursor + n]
                emb_cursor += n

            node_emb = _get_node_embeddings(it, residue_emb_slice)
            cids, cnts, cents = _segment_centroids(sl, node_emb)

            for cid, cnt, emb in zip(cids, cnts, cents):
                gidx = len(seg_emb_list)
                seg_emb_list.append(emb)
                segment_rows.append(SegmentMeta(
                    global_seg_index=gidx,
                    pdb_id=pdb,
                    segment_k=int(cid),
                    n_residues_assigned=int(cnt),
                    is_active_mask=True,
                    H=int(emb.numel()),
                ).__dict__)

    seg_mat = torch.stack(seg_emb_list, dim=0).float() if seg_emb_list else torch.empty((0, 0), dtype=torch.float32)
    return RunOutput(residue_rows=residue_rows, segment_rows=segment_rows, segment_embeddings=seg_mat, run_meta=run_meta)


@torch.no_grad()
def run_model(cfg: omegaconf.DictConfig, dataloader: ProteinDataLoader) -> RunOutput:
    device = _device()
    log.info(f"Using device: {device}")

    model = load_model(cfg, batch=next(iter(dataloader)))
    model.eval().to(device)
    extractor: SegmentExtractor = ModelExtractor()

    run_meta = {"encoder": str(cfg.get("encoder", "unknown")), "ckpt_path": str(cfg.get("ckpt_path", ""))}

    residue_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []
    seg_emb_list: List[torch.Tensor] = []

    for batch in tqdm(dataloader, desc="Clustering"):
        batch = batch.to(device)

        if hasattr(model, "featurise"):
            batch = model.featurise(batch)

        enc_out = model.encoder.forward(batch, return_clusters=True)
        hard, seg_mask, seg_emb = extractor.extract(batch, enc_out)

        batch_cpu = batch.to("cpu")
        items = batch_cpu.to_data_list()

        hard = hard.to("cpu")
        seg_mask = seg_mask.to("cpu")
        seg_emb = seg_emb.to("cpu")

        K = int(seg_mask.size(1))
        H = int(seg_emb.size(-1))

        for i, it in enumerate(items):
            pdb = _pdb_id(it)
            rid_full = _residue_ids(it)
            n_res = len(rid_full)

            a = hard[i, :n_res].long()
            valid = a != PAD

            rid = [_safe_parse_residue_numeric(r) for r, v in zip(rid_full, valid.tolist()) if v]
            a_valid = a[valid]
            labels = a_valid.tolist()
            
            residue_rows.append(ResidueAssignments(
                pdb_id=pdb,
                n_residues=len(rid),
                residue_ids=",".join(map(str, rid)),
                cluster_ids=",".join(map(str, labels)),
                K=K,
            ).__dict__)

            counts = torch.bincount(a_valid, minlength=K) if a_valid.numel() > 0 else torch.zeros(K, dtype=torch.long)
            active_idx = torch.nonzero(seg_mask[i].bool(), as_tuple=False).view(-1).tolist()

            for k in active_idx:
                emb_k = seg_emb[i, k].contiguous().float()
                gidx = len(seg_emb_list)
                seg_emb_list.append(emb_k)
                segment_rows.append(SegmentMeta(
                    global_seg_index=gidx,
                    pdb_id=pdb,
                    segment_k=int(k),
                    n_residues_assigned=int(counts[k].item()),
                    is_active_mask=True,
                    H=H,
                ).__dict__)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    seg_mat = torch.stack(seg_emb_list, dim=0).float() if seg_emb_list else torch.empty((0, 0), dtype=torch.float32)
    return RunOutput(residue_rows=residue_rows, segment_rows=segment_rows, segment_embeddings=seg_mat, run_meta=run_meta)


# =========================
# Input parsing + dataset
# =========================

def read_pdb_chain_table(input_file: str) -> pd.DataFrame:
    """
    Supports:
      (A) GO-style list: single column 'PDB-chain' (e.g., 1AD3-A)
      (B) SCOP lookup: columns ['pdb_id','label'] where pdb_id is a domain id
          (e.g., d1dlwa_, d2fk4a1). These are treated as *native structure IDs*.

    Returns df with columns:
      - id_key        : the ID to load from pdb_dir (either PDB-chain or domain id)
      - chain         : 'all' for domain ids, or extracted chain for PDB-chain
      - scop_label    : optional (only for SCOP lookup)
    """
    p = Path(input_file)
    if not p.exists():
        raise FileNotFoundError(f"cluster.input_file not found: {input_file}")

    # Load flexibly
    if p.suffix.lower() == ".csv":
        df0 = pd.read_csv(p)
    elif p.suffix.lower() in [".tsv", ".tab"]:
        df0 = pd.read_csv(p, sep="\t")
    elif p.suffix.lower() in [".txt"]:
        df0 = pd.read_csv(p, sep="\t", header=None)
        if df0.shape[1] == 1:
            df0.columns = ["pdb-chain"]
    else:
        df0 = pd.read_csv(p, delim_whitespace=True)

    cols_lower = [c.lower() for c in df0.columns.tolist()]

    # --- Case B: SCOP lookup ---
    if ("pdb_id" in cols_lower) and ("label" in cols_lower):
        rename = {df0.columns[i]: cols_lower[i] for i in range(len(df0.columns))}
        df0 = df0.rename(columns=rename)

        df = pd.DataFrame({
            "id_key": df0["pdb_id"].astype(str),
            "chain": "all",  # domain structures are usually single-chain already
            "scop_label": df0["label"].astype(str),
        }).reset_index(drop=True)
        return df

    # --- Case A: GO-style list ---

    df = pd.DataFrame({"id_key": df0["pdb-chain"].astype(str)})

    # hard excludes 
    for bad in ["5JM5-A", "5O61-K", "5O61-I", "5O61-R", "3OHM-B", "2MNT-A"]:
        df = df[df["id_key"] != bad]
    df = df.reset_index(drop=True)
    df["chain"] = df["id_key"].apply(lambda s: str(s).split("-")[1] if "-" in str(s) else "all")
    # keep only sane chains for this mode
    df = df[df["chain"].astype(str).str.len().isin([1, 3])].reset_index(drop=True)  # 'A' or 'all'
    return df



def build_dataset(cfg: omegaconf.DictConfig, df: pd.DataFrame) -> ProteinDataset:
    """
    df must have:
      - id_key : the identifier used to locate structure files in cfg.cluster.pdb_dir
      - chain  : 'all' (domain ids) or a single-letter chain (PDB-chain mode)
    """
    return ProteinDataset(
        root=cfg.cluster.pdb_dir,
        pdb_dir=cfg.cluster.pdb_dir,
        pdb_codes=df["id_key"].tolist(),   # IMPORTANT: domain ids go here directly
        chains=df["chain"].tolist(),
        overwrite=False,
        format="pdb",
        in_memory=False,
        esm_embedding_dir=cfg.cluster.esm_embedding_dir,
    )


def build_dataloader(cfg: omegaconf.DictConfig, dataset: ProteinDataset, baseline_mode: bool) -> ProteinDataLoader:
    return ProteinDataLoader(
        dataset,
        batch_size=1 if baseline_mode else int(cfg.cluster.get("batch_size", 16)),
        shuffle=False,
        pin_memory=True,
        num_workers=int(cfg.cluster.get("num_workers", 8)),
    )


# =========================
# Orchestration
# =========================

def evaluate(cfg: omegaconf.DictConfig) -> None:
    assert cfg.output_dir, "No output_dir provided."
    L.seed_everything(int(cfg.seed))

    df = read_pdb_chain_table(str(cfg.cluster.input_file))
    dataset = build_dataset(cfg, df)

    baseline_mode = str(cfg.cluster.model) == "baseline_model"
    dataloader = build_dataloader(cfg, dataset, baseline_mode)

    if baseline_mode:
        out = run_baseline(cfg, dataloader)
    else:
        out = run_model(cfg, dataloader)

    writer = OutputWriter(Path(cfg.output_dir), str(cfg.cluster.output_file))
    writer.write(out)


@hydra.main(version_base="1.3", config_path="../configs", config_name="cluster")
def main(cfg: omegaconf.DictConfig) -> None:
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    src_register_custom_omegaconf_resolvers()
    main()
