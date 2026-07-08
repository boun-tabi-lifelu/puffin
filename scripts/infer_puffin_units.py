import argparse
import copy
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from graphein import verbose
from graphein.protein.tensor.data import ProteinBatch
from graphein.protein.tensor.io import protein_to_pyg
from huggingface_hub import hf_hub_download, list_repo_files
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from proteinworkshop import register_custom_omegaconf_resolvers
from proteinworkshop.features.sequence_features import amino_acid_one_hot

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "lifelu/puffin"
DEFAULT_ESM_MODEL_PATH = Path("/cta/share/users/esm/ESM-1b")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import register_custom_omegaconf_resolvers as register_src_resolvers  # noqa: E402
from src.utils.model_utils import load_model  # noqa: E402

verbose(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign PUFFIN units to residues in one PDB file."
    )
    parser.add_argument("pdb", type=Path, help="Path to the input PDB file.")
    parser.add_argument(
        "--chain",
        default="all",
        help="Chain to load from the PDB file. Use 'all' to keep all chains. Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("puffin_units"),
        help="Directory for output files. Default: puffin_units.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output files. Default: input PDB stem.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local PUFFIN checkpoint. If omitted, downloads from --repo-id.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo id used when --checkpoint is omitted. Default: {DEFAULT_REPO_ID}.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default=None,
        help="Specific checkpoint filename inside the Hugging Face repo.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    parser.add_argument(
        "--esm-model-path",
        type=Path,
        default=DEFAULT_ESM_MODEL_PATH,
        help=(
            "Local ESM model identifier/path used to instantiate the ESM module. "
            "The ESM weights are loaded from the PUFFIN checkpoint. "
            f"Default: {DEFAULT_ESM_MODEL_PATH}."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Temporary directory for Hydra runtime paths. Default: a temporary directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto, cuda, cuda:0, or cpu. Default: auto.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override, e.g. --override encoder.num_clusters=64. Can be repeated.",
    )
    parser.add_argument(
        "--unit-cluster-artifact",
        type=Path,
        default=None,
        help=(
            "Optional unit cluster-function artifact directory. When provided, "
            "active PUFFIN units are assigned to global unit clusters and GO functions."
        ),
    )
    parser.add_argument(
        "--unit-cluster-top-n",
        type=int,
        default=5,
        help="Maximum GO terms to attach per assigned unit cluster. Default: 5.",
    )
    parser.add_argument(
        "--unit-cluster-max-qval",
        type=float,
        default=None,
        help="Optional maximum q-value for attached unit cluster GO terms.",
    )
    parser.add_argument(
        "--unit-cluster-batch-size",
        type=int,
        default=8192,
        help="Batch size for centroid assignment. Default: 8192.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        ckpt = args.checkpoint.expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    cache_dir = str(args.cache_dir.expanduser()) if args.cache_dir else None
    if args.checkpoint_filename:
        return Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=args.checkpoint_filename,
                revision=args.revision,
                cache_dir=cache_dir,
            )
        )

    repo_files = list_repo_files(repo_id=args.repo_id, revision=args.revision)
    candidates = sorted(
        name for name in repo_files if name.endswith((".ckpt", ".pt", ".pth"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt/.pt/.pth checkpoint found in Hugging Face repo {args.repo_id}. "
            "Pass --checkpoint-filename or --checkpoint."
        )
    return Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=candidates[0],
            revision=args.revision,
            cache_dir=cache_dir,
        )
    )


def hydra_path(value: Path) -> str:
    return str(value.expanduser().resolve())


def build_cfg(args: argparse.Namespace, ckpt_path: Path, work_dir: Path):
    os.environ.setdefault("ROOT_DIR", str(REPO_ROOT))
    os.environ.setdefault("DATA_PATH", str(work_dir / "data"))
    os.environ.setdefault("RUNS_PATH", str(work_dir / "runs"))
    os.environ.setdefault("WANDB_ENTITY", "")
    os.environ.setdefault("WANDB_PROJECT", "")

    register_custom_omegaconf_resolvers()
    register_src_resolvers()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [
        f"ckpt_path={hydra_path(ckpt_path)}",
        f"output_dir={hydra_path(args.output_dir)}",
        "logger=csv",
        f"env.paths.root_dir={hydra_path(REPO_ROOT)}",
        f"env.paths.data={hydra_path(work_dir / 'data')}",
        f"env.paths.output_dir={hydra_path(args.output_dir)}",
        f"env.paths.log_dir={hydra_path(work_dir / 'runs')}",
        f"env.paths.runs={hydra_path(work_dir / 'runs')}",
        "cluster.pdb_dir=.",
        "cluster.esm_embedding_dir=.",
        f"encoder.esm_model_path={hydra_path(args.esm_model_path)}",
    ]
    overrides.extend(args.override)

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        return compose(config_name="cluster", overrides=overrides)


def build_batch(pdb_path: Path, chain: str) -> ProteinBatch:
    pdb_path = pdb_path.expanduser().resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    graph = protein_to_pyg(
        path=str(pdb_path),
        chain_selection=chain,
        keep_insertions=True,
        store_het=False,
    )
    graph.id = pdb_path.stem if chain == "all" else f"{pdb_path.stem}-{chain}"
    graph.x = torch.zeros(graph.coords.shape[0])
    graph.amino_acid_one_hot = amino_acid_one_hot(graph)
    graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)
    return ProteinBatch.from_data_list([graph], None, None)


def get_output_tensor(output: Dict[str, Any], key: str) -> torch.Tensor:
    value = output[key]
    if isinstance(value, list):
        value = value[0]
    return value


def split_residue_id(residue_id: Any) -> Tuple[str, str]:
    parts = str(residue_id).split(":")
    if len(parts) >= 3:
        return parts[1], parts[2]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", str(residue_id)


def residue_name_at(residues: Any, idx: int) -> str:
    if residues is None:
        return ""
    try:
        return str(residues[idx])
    except Exception:
        return ""


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def apply_unit_cluster_debias(
    embeddings: np.ndarray, debias: Mapping[str, Any]
) -> np.ndarray:
    x = l2_normalize(embeddings.astype(np.float32, copy=False))
    mu = np.asarray(debias["mu"], dtype=np.float32).reshape(1, -1)
    pcs = np.asarray(debias.get("pcs", []), dtype=np.float32)
    if x.shape[1] != mu.shape[1]:
        raise ValueError(
            f"Unit embedding dim={x.shape[1]} does not match debias dim={mu.shape[1]}"
        )
    x = x - mu
    if pcs.size:
        if pcs.ndim != 2 or pcs.shape[1] != x.shape[1]:
            raise ValueError(f"Invalid debias pcs shape={pcs.shape}; expected (*, {x.shape[1]})")
        for pc in pcs:
            pc_2d = pc.reshape(1, -1)
            x = x - (x @ pc_2d.T) * pc_2d
    return l2_normalize(x)


def load_unit_cluster_artifact(
    artifact_dir: Path,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    artifact_dir = artifact_dir.expanduser().resolve()
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Unit cluster artifact not found: {artifact_dir}")

    config_path = artifact_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else {}
    centroids = np.load(artifact_dir / "centroids.npy").astype(np.float32)
    debias = load_json(artifact_dir / "debias_transform.json")

    unit_clusters_path = artifact_dir / "unit_clusters.json"
    legacy_prototypes_path = artifact_dir / "prototypes.json"
    if unit_clusters_path.exists():
        data = load_json(unit_clusters_path)
        records = data.get("unit_clusters", [])
        id_key = "unit_cluster_id"
    elif legacy_prototypes_path.exists():
        data = load_json(legacy_prototypes_path)
        records = data.get("prototypes", [])
        id_key = "proto"
    else:
        raise FileNotFoundError(
            f"Missing unit_clusters.json or prototypes.json in {artifact_dir}"
        )

    unit_cluster_map: Dict[int, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or id_key not in record:
            continue
        unit_cluster_id = int(record[id_key])
        normalized = dict(record)
        normalized["unit_cluster_id"] = unit_cluster_id
        unit_cluster_map[unit_cluster_id] = normalized
    return config, centroids, debias, unit_cluster_map


def assign_unit_clusters(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    debias: Mapping[str, Any],
    *,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError(f"Unit embeddings must be 2D, got shape={embeddings.shape}")
    if centroids.ndim != 2:
        raise ValueError(f"Unit cluster centroids must be 2D, got shape={centroids.shape}")

    x = apply_unit_cluster_debias(embeddings, debias)
    c = l2_normalize(centroids.astype(np.float32, copy=False))
    if x.shape[1] != c.shape[1]:
        raise ValueError(
            f"Unit embedding dim={x.shape[1]} does not match centroid dim={c.shape[1]}"
        )

    batch_size = max(1, int(batch_size))
    unit_cluster_ids = np.empty((x.shape[0],), dtype=np.int64)
    sims = np.empty((x.shape[0],), dtype=np.float32)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        scores = x[start:end] @ c.T
        unit_cluster_ids[start:end] = np.argmax(scores, axis=1)
        sims[start:end] = np.max(scores, axis=1)
    return unit_cluster_ids, sims


def terms_for_unit_cluster(
    unit_cluster_map: Mapping[int, Mapping[str, Any]],
    unit_cluster_id: int,
    *,
    top_n: Optional[int],
    max_qval: Optional[float],
) -> List[Dict[str, Any]]:
    record = unit_cluster_map.get(int(unit_cluster_id), {})
    terms = list(record.get("function_terms", []))
    if max_qval is not None:
        filtered = []
        for term in terms:
            qval = term.get("qval")
            if qval is None or qval == "inf":
                continue
            if float(qval) <= float(max_qval):
                filtered.append(term)
        terms = filtered
    if top_n is not None:
        terms = terms[: int(top_n)]
    return terms


def term_string(terms: Sequence[Mapping[str, Any]]) -> str:
    return "|".join(str(term.get("go_term", "")) for term in terms if term.get("go_term"))


def top_go_term(terms: Sequence[Mapping[str, Any]]) -> str:
    if not terms:
        return ""
    return str(terms[0].get("go_term", ""))


def run_inference(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_checkpoint(args)
    device = resolve_device(args.device)
    cache_root = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir
        else Path(tempfile.mkdtemp(prefix="puffin_infer_"))
    )

    cfg = build_cfg(args, ckpt_path, cache_root)
    batch = build_batch(args.pdb, args.chain)

    model = load_model(cfg, copy.deepcopy(batch), device=device)
    model.eval()

    with torch.no_grad():
        batch = batch.to(device)
        if hasattr(model, "featurise"):
            batch = model.featurise(batch)
        output = model.encoder.forward(batch, return_clusters=True)

    clusters = get_output_tensor(output, "clusters").detach().cpu().long()
    seg_mask = get_output_tensor(output, "seg_mask").detach().cpu().bool()
    seg_embeddings = output["node_embedding"].detach().cpu().float()

    item = batch.to("cpu").to_data_list()[0]
    residue_ids = list(getattr(item, "residue_id"))
    residues = getattr(item, "residues", None)
    assignments = clusters[0, : len(residue_ids)].tolist()
    pdb_id = str(getattr(item, "id", args.pdb.stem))

    valid_assignments = [u for u in assignments if u >= 0]
    k_total = int(seg_mask.size(1))
    counts = (
        torch.bincount(torch.tensor(valid_assignments), minlength=k_total)
        if valid_assignments
        else torch.zeros(k_total, dtype=torch.long)
    )
    active_unit_ids = [
        int(unit_id)
        for unit_id in torch.nonzero(seg_mask[0], as_tuple=False).view(-1).tolist()
    ]

    unit_cluster_by_unit: Dict[int, Dict[str, Any]] = {}
    unit_function_rows: List[Dict[str, Any]] = []
    unit_cluster_artifact_config: Optional[Dict[str, Any]] = None
    if args.unit_cluster_artifact is not None:
        unit_cluster_artifact_config, centroids, debias, unit_cluster_map = load_unit_cluster_artifact(
            args.unit_cluster_artifact
        )
        active_embeddings = seg_embeddings[0, active_unit_ids].numpy().astype(np.float32)
        unit_cluster_ids, unit_cluster_sims = assign_unit_clusters(
            active_embeddings,
            centroids,
            debias,
            batch_size=args.unit_cluster_batch_size,
        )
        for unit_id, unit_cluster_id, sim in zip(
            active_unit_ids, unit_cluster_ids, unit_cluster_sims
        ):
            terms = terms_for_unit_cluster(
                unit_cluster_map,
                int(unit_cluster_id),
                top_n=args.unit_cluster_top_n,
                max_qval=args.unit_cluster_max_qval,
            )
            row = {
                "pdb_id": pdb_id,
                "puffin_unit": int(unit_id),
                "n_residues_assigned": int(counts[unit_id].item()),
                "unit_cluster_id": int(unit_cluster_id),
                "unit_cluster_similarity": float(sim),
                "go_terms": term_string(terms),
                "top_go_term": top_go_term(terms),
                "function_terms_json": json.dumps(terms, sort_keys=True),
            }
            unit_cluster_by_unit[int(unit_id)] = row
            unit_function_rows.append(row)

    residue_rows: List[Dict[str, Any]] = []
    for idx, (residue_id, unit_id) in enumerate(zip(residue_ids, assignments), start=1):
        chain, residue_number = split_residue_id(residue_id)
        row = {
            "pdb_id": pdb_id,
            "chain": chain,
            "residue_index": idx,
            "residue_id": str(residue_id),
            "residue_number": residue_number,
            "residue_name": residue_name_at(residues, idx - 1),
            "puffin_unit": int(unit_id),
        }
        if args.unit_cluster_artifact is not None:
            unit_cluster = unit_cluster_by_unit.get(int(unit_id), {})
            row.update(
                {
                    "unit_cluster_id": unit_cluster.get("unit_cluster_id", ""),
                    "unit_cluster_similarity": unit_cluster.get("unit_cluster_similarity", ""),
                    "go_terms": unit_cluster.get("go_terms", ""),
                    "top_go_term": unit_cluster.get("top_go_term", ""),
                }
            )
        residue_rows.append(row)

    metadata_rows = []
    for unit_id in active_unit_ids:
        row = {
            "pdb_id": pdb_id,
            "puffin_unit": int(unit_id),
            "n_residues_assigned": int(counts[unit_id].item()),
            "active": True,
        }
        if args.unit_cluster_artifact is not None:
            unit_cluster = unit_cluster_by_unit.get(int(unit_id), {})
            row.update(
                {
                    "unit_cluster_id": unit_cluster.get("unit_cluster_id", ""),
                    "unit_cluster_similarity": unit_cluster.get("unit_cluster_similarity", ""),
                    "go_terms": unit_cluster.get("go_terms", ""),
                    "top_go_term": unit_cluster.get("top_go_term", ""),
                    "function_terms_json": unit_cluster.get("function_terms_json", ""),
                }
            )
        metadata_rows.append(row)

    prefix = args.output_prefix or args.pdb.stem
    residue_csv = args.output_dir / f"{prefix}_puffin_units.csv"
    metadata_csv = args.output_dir / f"{prefix}_puffin_unit_metadata.csv"
    unit_functions_csv = args.output_dir / f"{prefix}_puffin_unit_cluster_functions.csv"
    embeddings_pt = args.output_dir / f"{prefix}_puffin_unit_embeddings.pt"

    residue_fields = [
        "pdb_id",
        "chain",
        "residue_index",
        "residue_id",
        "residue_number",
        "residue_name",
        "puffin_unit",
    ]
    metadata_fields = ["pdb_id", "puffin_unit", "n_residues_assigned", "active"]
    unit_function_fields = [
        "pdb_id",
        "puffin_unit",
        "n_residues_assigned",
        "unit_cluster_id",
        "unit_cluster_similarity",
        "go_terms",
        "top_go_term",
        "function_terms_json",
    ]
    if args.unit_cluster_artifact is not None:
        residue_fields.extend(
            ["unit_cluster_id", "unit_cluster_similarity", "go_terms", "top_go_term"]
        )
        metadata_fields.extend(
            [
                "unit_cluster_id",
                "unit_cluster_similarity",
                "go_terms",
                "top_go_term",
                "function_terms_json",
            ]
        )

    write_csv(residue_csv, residue_rows, residue_fields)
    write_csv(metadata_csv, metadata_rows, metadata_fields)
    if args.unit_cluster_artifact is not None:
        write_csv(unit_functions_csv, unit_function_rows, unit_function_fields)

    torch.save(
        {
            "embeddings": seg_embeddings[0],
            "seg_mask": seg_mask[0],
            "checkpoint": str(ckpt_path),
            "esm_model_path": hydra_path(args.esm_model_path),
            "pdb": str(args.pdb),
            "chain": args.chain,
            "unit_cluster_artifact": (
                str(args.unit_cluster_artifact.expanduser().resolve())
                if args.unit_cluster_artifact is not None
                else None
            ),
            "unit_cluster_artifact_config": unit_cluster_artifact_config,
            "unit_cluster_assignments": unit_function_rows,
        },
        embeddings_pt,
    )

    print(f"Wrote residue assignments: {residue_csv}")
    print(f"Wrote unit metadata: {metadata_csv}")
    if args.unit_cluster_artifact is not None:
        print(f"Wrote unit cluster/function assignments: {unit_functions_csv}")
    print(f"Wrote unit embeddings: {embeddings_pt}")


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
