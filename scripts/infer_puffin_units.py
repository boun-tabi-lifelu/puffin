import argparse
import copy
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

    residue_rows: List[Dict[str, Any]] = []
    for idx, (residue_id, unit_id) in enumerate(zip(residue_ids, assignments), start=1):
        chain, residue_number = split_residue_id(residue_id)
        residue_rows.append(
            {
                "pdb_id": pdb_id,
                "chain": chain,
                "residue_index": idx,
                "residue_id": str(residue_id),
                "residue_number": residue_number,
                "residue_name": residue_name_at(residues, idx - 1),
                "puffin_unit": int(unit_id),
            }
        )

    valid_assignments = [u for u in assignments if u >= 0]
    k_total = int(seg_mask.size(1))
    counts = (
        torch.bincount(torch.tensor(valid_assignments), minlength=k_total)
        if valid_assignments
        else torch.zeros(k_total, dtype=torch.long)
    )
    metadata_rows = []
    for unit_id in torch.nonzero(seg_mask[0], as_tuple=False).view(-1).tolist():
        metadata_rows.append(
            {
                "pdb_id": pdb_id,
                "puffin_unit": int(unit_id),
                "n_residues_assigned": int(counts[unit_id].item()),
                "active": True,
            }
        )

    prefix = args.output_prefix or args.pdb.stem
    residue_csv = args.output_dir / f"{prefix}_puffin_units.csv"
    metadata_csv = args.output_dir / f"{prefix}_puffin_unit_metadata.csv"
    embeddings_pt = args.output_dir / f"{prefix}_puffin_unit_embeddings.pt"

    write_csv(
        residue_csv,
        residue_rows,
        ["pdb_id", "chain", "residue_index", "residue_id", "residue_number", "residue_name", "puffin_unit"],
    )
    write_csv(
        metadata_csv,
        metadata_rows,
        ["pdb_id", "puffin_unit", "n_residues_assigned", "active"],
    )
    torch.save(
        {
            "embeddings": seg_embeddings[0],
            "seg_mask": seg_mask[0],
            "checkpoint": str(ckpt_path),
            "esm_model_path": hydra_path(args.esm_model_path),
            "pdb": str(args.pdb),
            "chain": args.chain,
        },
        embeddings_pt,
    )

    print(f"Wrote residue assignments: {residue_csv}")
    print(f"Wrote unit metadata: {metadata_csv}")
    print(f"Wrote unit embeddings: {embeddings_pt}")


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
