from __future__ import annotations

import json
import html as html_utils
import math
import os
import tempfile
import uuid
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT = os.getenv(
    "PUFFIN_CHECKPOINT_PATH",
    "lifelu/puffin",
)
ESM_MODEL_PATH = os.getenv(
    "PUFFIN_ESM_MODEL_PATH",
    str(Path.home() / ".cache" / "puffin" / "ESM-1b"),
)
RESULTS_ROOT = Path(os.getenv("PUFFIN_RESULTS_DIR", str(PROJECT_ROOT / "results" / "gradio")))
UNIT_CLUSTER_ARTIFACT_DIR = Path(
    os.getenv(
        "PUFFIN_UNIT_CLUSTER_ARTIFACT_DIR",
        str(PROJECT_ROOT / "artifacts" / "puffin-unit-cluster-functions"),
    )
)
GO_OBO_PATH = os.getenv("PUFFIN_GO_OBO_PATH", "")
QUICKGO_TERM_URL = "https://www.ebi.ac.uk/QuickGO/term/{go_id}"
QUICKGO_API_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_ids}"
UNIT_COLORS = [
    "#E53935", "#00897B", "#1E88E5", "#FB8C00", "#8E24AA", "#00ACC1",
    "#D81B60", "#43A047", "#6D4C41", "#3949AB", "#F4511E", "#7CB342",
    "#5E35B1", "#039BE5", "#C0CA33", "#FDD835",
]
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _read_pdb_lines(pdb_path: str | Path) -> List[str]:
    path = Path(pdb_path)
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def validate_pdb_coordinates(pdb_path: str | Path) -> None:
    """Validate the fixed-width coordinates required by Graphein's PDB parser."""
    atom_count = 0
    for line_number, line in enumerate(_read_pdb_lines(pdb_path), start=1):
        if not line.startswith("ATOM"):
            continue
        atom_count += 1
        if len(line) < 54:
            raise ValueError(f"Invalid PDB atom record at line {line_number}: it is too short.")
        try:
            coordinates = [float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54))]
        except ValueError as error:
            raise ValueError(
                f"Invalid PDB coordinates at line {line_number}; expected numeric x, y, and z fields."
            ) from error
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            raise ValueError(f"Invalid PDB coordinates at line {line_number}.")
    if atom_count == 0:
        raise ValueError("The uploaded file contains no ATOM records in standard PDB format.")


def canonicalize_pdb_chain(input_path: str | Path, output_path: str | Path, chain_id: str) -> None:
    """Create the verbatim chain-only PDB layout used by make_chain_pdb_inputs.py."""
    wrote_atoms = 0
    with Path(input_path).open(encoding="utf-8", errors="ignore") as source, Path(output_path).open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            record = line[:6]
            if record.startswith("ATOM") or record.startswith("HETATM"):
                if len(line) > 21 and line[21].strip() == chain_id:
                    destination.write(line)
                    wrote_atoms += 1
            elif line.startswith("TER") or line.startswith("END"):
                destination.write(line)
    if wrote_atoms == 0:
        raise ValueError(f"No ATOM/HETATM records found for chain '{chain_id}'.")


def _chain_file_stem(pdb_path: str | Path, chain_id: str) -> str:
    """Return NAME-CHAIN without duplicating an existing chain suffix."""
    stem = Path(pdb_path).stem
    suffix = f"-{chain_id}"
    return stem if stem.upper().endswith(suffix.upper()) else f"{stem}{suffix}"

def parse_pdb_structure(pdb_path: str | Path) -> Dict[str, Any]:
    """Parse a lightweight PDB file and extract chain/residue metadata."""
    path = Path(pdb_path)
    lines = _read_pdb_lines(path)

    chains: List[str] = []
    protein_residue_counts: Dict[str, int] = {}
    residues: List[Dict[str, Any]] = []
    residue_numbers: List[int] = []
    seen_residues: set[Tuple[str, int]] = set()

    for line in lines:
        if not line.startswith("ATOM"):
            continue
        chain_id = line[21:22].strip()
        residue_name = line[17:20].strip()
        if chain_id and chain_id not in chains:
            chains.append(chain_id)
        if residue_name in STANDARD_AMINO_ACIDS:
            protein_residue_counts[chain_id] = protein_residue_counts.get(chain_id, 0) + 1
        residue_number = line[22:26].strip()
        try:
            residue_number_int = int(residue_number)
        except ValueError:
            residue_number_int = len(residues) + 1
        key = (chain_id, residue_number_int)
        if key in seen_residues:
            continue
        seen_residues.add(key)
        residue_numbers.append(residue_number_int)
        residues.append({"chain": chain_id, "name": residue_name, "number": residue_number_int})

    residue_count = len(residues)
    return {
        "path": str(path),
        "name": path.name,
        "chains": chains,
        "chain_ids": chains,
        "residue_count": residue_count,
        "residues": residues,
        "residue_numbers": residue_numbers,
        "protein_chains": [chain for chain in chains if protein_residue_counts.get(chain, 0) > 0],
    }


def select_chain(structure: Dict[str, Any], requested_chain: str = "") -> str:
    """Select the requested chain, defaulting to the first chain in the PDB."""
    chains = structure.get("chain_ids", [])
    protein_chains = structure.get("protein_chains", chains)
    selected_chain = requested_chain.strip() or (protein_chains[0] if protein_chains else "")
    if not selected_chain:
        raise ValueError("The uploaded PDB does not contain an ATOM record with a chain.")
    if selected_chain not in chains:
        raise ValueError(f"Chain '{selected_chain}' was not found in the uploaded PDB.")
    if selected_chain not in protein_chains:
        raise ValueError(f"Chain '{selected_chain}' is not a protein chain. Select a chain containing amino acids.")
    return selected_chain


def _puffin_config(checkpoint_path: str, esm_model_path: str | None = None) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(
            config_name="cluster",
            overrides=[
                f"ckpt_path={checkpoint_path}",
                "objective_type=dual",
                "cluster.model=model",
                "+cluster.num_workers=0",
                "encoder.gnn_type=GAT",
                "encoder.hidden_dim=512",
                "encoder.num_clusters=64",
                "encoder.num_res_gnn_layers=2",
                "encoder.num_seg_gnn_layers=2",
                "encoder.use_seg_res_cross_attn=false",
                "encoder.proj_layer=true",
                "encoder.esm_embed_dim=512",
                "encoder.input_feat_dim=512",
                "encoder.fuse_lm_method=sum",
                (
                    f"encoder.esm_model_path={esm_model_path}"
                    if esm_model_path
                    else "encoder.esm_model_path=null"
                ),
                "hydra/hydra_logging=default",
                "hydra/job_logging=none",
            ],
        )


def _proteinworkshop_esm_path(esm_model_path: str) -> str:
    """Convert an ESM-1b weights path to ProteinWorkshop's model convention."""
    path = Path(esm_model_path).expanduser()
    if path.name == "esm1b_t33_650M_UR50S.pt":
        return str(path.parent / "ESM-1b")
    return str(path)


def _resolve_checkpoint(checkpoint: str) -> str:
    if Path(checkpoint).is_file():
        return checkpoint

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=checkpoint, filename="model.ckpt")


def build_puffin_batch(pdb_path: str | Path, pdb_id: str) -> Any:
    """Build the chain-specific graph using the verified public-release path."""
    import torch
    from graphein.protein.tensor.data import ProteinBatch
    from graphein.protein.tensor.io import protein_to_pyg
    from proteinworkshop.features.sequence_features import amino_acid_one_hot

    graph = protein_to_pyg(
        path=str(Path(pdb_path).resolve()),
        chain_selection="all",
        keep_insertions=True,
        store_het=False,
    )
    graph.id = pdb_id
    graph.x = torch.zeros(graph.coords.shape[0])
    graph.amino_acid_one_hot = amino_acid_one_hot(graph)
    graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)
    return ProteinBatch.from_data_list([graph], None, None)


def load_uploaded_esm_embeddings(
    embedding_path: str | Path,
    residue_ids: List[Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Load and align an uploaded ESM-1b tensor to the structure residue order."""
    import torch

    path = Path(embedding_path)
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("ESM embeddings must be a PyTorch .pt or .pth file.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if "embeddings" not in data:
            raise ValueError("The embedding file is a dictionary but has no 'embeddings' key.")
        embeddings = data["embeddings"]
        embedding_ids = data.get("index")
    elif torch.is_tensor(data):
        embeddings = data
        embedding_ids = None
    else:
        raise ValueError(
            "Expected a tensor or a dictionary containing 'embeddings' and optional 'index'."
        )

    if not torch.is_tensor(embeddings):
        raise ValueError("The uploaded 'embeddings' value is not a tensor.")
    embeddings = embeddings.detach().cpu()
    if embeddings.ndim == 3 and embeddings.shape[0] == 1:
        embeddings = embeddings.squeeze(0)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected ESM embeddings with shape [residues, features], got {tuple(embeddings.shape)}."
        )
    if embeddings.shape[1] != 1280:
        raise ValueError(
            f"Expected ESM-1b feature dimension 1280, got {embeddings.shape[1]}."
        )

    structure_ids = [str(residue) for residue in residue_ids]
    if embedding_ids is None:
        if embeddings.shape[0] != len(structure_ids):
            raise ValueError(
                "An embedding tensor without an 'index' must have exactly one row per PDB "
                f"residue ({embeddings.shape[0]} rows versus {len(structure_ids)} residues)."
            )
        aligned = embeddings
        matched = len(structure_ids)
        alignment = "positional"
    else:
        if (
            isinstance(embedding_ids, (list, tuple))
            and len(embedding_ids) == 1
            and isinstance(embedding_ids[0], (list, tuple))
        ):
            embedding_ids = embedding_ids[0]
        embedding_ids = [str(residue) for residue in embedding_ids]
        if len(embedding_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Embedding index has {len(embedding_ids)} residues but the tensor has "
                f"{embeddings.shape[0]} rows."
            )
        positions = {residue: index for index, residue in enumerate(embedding_ids)}
        common = [residue for residue in structure_ids if residue in positions]
        if len(common) != len(structure_ids):
            missing = [residue for residue in structure_ids if residue not in positions]
            raise ValueError(
                f"Uploaded embeddings match {len(common)}/{len(structure_ids)} PDB residues; "
                f"first missing IDs: {missing[:5]}. Check that the PDB chain and embedding "
                "file correspond."
            )
        aligned = torch.stack([embeddings[positions[residue]] for residue in structure_ids])
        matched = len(common)
        alignment = "residue_id"

    return aligned.float(), {
        "source": "uploaded",
        "file_name": path.name,
        "shape": list(aligned.shape),
        "matched_residues": matched,
        "alignment": alignment,
    }


def run_puffin(
    pdb_path: str,
    chain: str = "",
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    esm_embeddings_path: str | None = None,
    esm_mode: str = "Legacy zero embeddings",
    esm_model_path: str = ESM_MODEL_PATH,
) -> Dict[str, Any]:
    """Run PUFFIN with uploaded, computed, or historical zero ESM features."""
    import torch
    import lightning as L
    from src.utils.model_utils import load_model

    pdb_file = Path(pdb_path)
    validate_pdb_coordinates(pdb_file)
    structure = parse_pdb_structure(pdb_file)
    selected_chain = select_chain(structure, chain)

    # Canonicalize first, then use the direct public-release graph path. This
    # reproduces the historical chain-specific test inputs without involving
    # ProteinDataset filename and embedding lookup behavior.
    dataset_dir = Path(tempfile.mkdtemp(prefix="puffin-upload-"))
    chain_file_stem = _chain_file_stem(pdb_file, selected_chain)
    normalized_pdb = dataset_dir / f"{chain_file_stem}.pdb"
    canonicalize_pdb_chain(pdb_file, normalized_pdb, selected_chain)
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    resolved_esm_model_path = _proteinworkshop_esm_path(esm_model_path)
    cfg = _puffin_config(
        checkpoint_path,
        resolved_esm_model_path if esm_mode == "Compute ESM" else None,
    )
    L.seed_everything(int(cfg.seed))
    batch = build_puffin_batch(normalized_pdb, chain_file_stem)
    item = batch.to_data_list()[0]
    if esm_mode == "Legacy zero embeddings":
        batch.esm_embeddings = torch.zeros((len(item.residue_id), 1280), dtype=torch.float32)
        batch.esm_id = batch.residue_id
        embedding_info = {
            "source": "zero",
            "shape": list(batch.esm_embeddings.shape),
        }
    elif esm_mode == "Uploaded embeddings":
        if not esm_embeddings_path:
            raise ValueError(
                "ESM mode is 'Uploaded embeddings', but no embedding file was provided."
            )
        batch.esm_embeddings, embedding_info = load_uploaded_esm_embeddings(
            esm_embeddings_path,
            list(item.residue_id),
        )
        batch.esm_id = batch.residue_id
    elif esm_mode == "Compute ESM":
        embedding_info = {
            "source": "computed",
            "model_path": resolved_esm_model_path,
        }
    else:
        raise ValueError(f"Unknown ESM mode: {esm_mode}")
    model = load_model(cfg, batch=batch, device="auto")
    model.eval()
    device = next(model.parameters()).device
    batch = batch.to(device)
    batch = model.featurise(batch)

    with torch.no_grad():
        output = model.encoder.forward(batch, return_clusters=True)

    labels = output["clusters"][0][0].detach().cpu().tolist()
    item = batch.to("cpu").to_data_list()[0]
    residue_ids = []
    for residue in item.residue_id:
        parts = str(residue).split(":")
        residue_ids.append(int(parts[2] if len(parts) >= 3 else parts[-1]))
    assignments = [
        {"chain": selected_chain, "residue": residue_id, "unit_id": unit_id}
        for residue_id, unit_id in zip(residue_ids, labels)
        if unit_id >= 0
    ]
    units: List[Dict[str, Any]] = []
    for unit_id in sorted({row["unit_id"] for row in assignments}):
        residues = [row["residue"] for row in assignments if row["unit_id"] == unit_id]
        units.append({"unit_id": unit_id, "residue_count": len(residues), "residues": residues})
    try:
        prototype_ids = assign_global_prototypes(output["node_embedding"][0])
    except (FileNotFoundError, KeyError, ValueError):
        prototype_ids = {}
    unit_clusters = [
        {**unit, "prototype_id": prototype_ids.get(unit["unit_id"])}
        for unit in units
    ]

    return {
        "assignments": assignments,
        "units": units,
        "unit_clusters": _enrichment_rows(
            [unit for unit in unit_clusters if unit["prototype_id"] is not None]
        ),
        "checkpoint": checkpoint_path,
        "chain": selected_chain,
        "esm_embeddings": embedding_info,
    }


def build_download_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def assign_global_prototypes(unit_embeddings: Any) -> Dict[int, int]:
    """Map local unit embeddings to the trained global prototype inventory."""
    import numpy as np

    centroids = np.load(UNIT_CLUSTER_ARTIFACT_DIR / "centroids.npy").astype("float32")
    transform = json.loads(
        (UNIT_CLUSTER_ARTIFACT_DIR / "debias_transform.json").read_text()
    )
    embeddings = unit_embeddings.detach().cpu().numpy().astype("float32")

    def normalize(values: Any) -> Any:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.clip(norms, 1e-12, None)

    transformed = normalize(embeddings) - np.asarray(transform["mu"], dtype="float32").reshape(1, -1)
    for component in np.asarray(transform.get("pcs", []), dtype="float32"):
        transformed -= (transformed @ component.reshape(-1, 1)) * component.reshape(1, -1)
    scores = normalize(transformed) @ normalize(centroids).T
    return {int(index): int(prototype) for index, prototype in enumerate(scores.argmax(axis=1))}



@lru_cache(maxsize=1)
def _local_go_names() -> Dict[str, str]:
    """Read GO IDs and names from an optional local OBO ontology."""
    if not GO_OBO_PATH or not Path(GO_OBO_PATH).is_file():
        return {}
    names: Dict[str, str] = {}
    current_id = current_name = ""
    for raw_line in Path(GO_OBO_PATH).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line == "[Term]":
            if current_id and current_name:
                names[current_id] = current_name
            current_id = current_name = ""
        elif line.startswith("id: GO:"):
            current_id = line[4:].strip()
        elif line.startswith("name: "):
            current_name = line[6:].strip()
    if current_id and current_name:
        names[current_id] = current_name
    return names


@lru_cache(maxsize=256)
def _remote_go_names(go_ids: Tuple[str, ...]) -> Dict[str, str]:
    if not go_ids:
        return {}
    request = urllib.request.Request(
        QUICKGO_API_URL.format(go_ids=",".join(go_ids)),
        headers={"Accept": "application/json", "User-Agent": "PUFFIN-Unit-Explorer/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            data = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return {}
    return {str(x["id"]): str(x["name"]) for x in data.get("results", []) if x.get("id") and x.get("name")}


def resolve_go_names(go_ids: List[str]) -> Dict[str, str]:
    unique_ids = tuple(dict.fromkeys(str(go_id) for go_id in go_ids))
    local = _local_go_names()
    names = {go_id: local[go_id] for go_id in unique_ids if go_id in local}
    names.update(_remote_go_names(tuple(go_id for go_id in unique_ids if go_id not in names)))
    return names

@lru_cache(maxsize=1)
def _unit_cluster_terms() -> Dict[int, List[Dict[str, Any]]]:
    artifact_path = UNIT_CLUSTER_ARTIFACT_DIR / "unit_clusters.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return {
        int(row["unit_cluster_id"]): list(row.get("function_terms", []))
        for row in artifact.get("unit_clusters", [])
    }


def _enrichment_rows(unit_clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    function_map = _unit_cluster_terms()
    matched = []
    for cluster in unit_clusters:
        matches = function_map.get(int(cluster["prototype_id"]), [])[:5]
        terms = [
            {
                "go_term": str(row["go_term"]),
                "name": str(row.get("go_name") or ""),
                "qval": float(row["qval"]),
                "odds_ratio": float(row["odds_ratio_approx"]),
            }
            for row in matches
        ]
        matched.append((cluster, terms))
    names = resolve_go_names(
        [
            term["go_term"]
            for _, terms in matched
            for term in terms
            if not term["name"]
        ]
    )
    rows = []
    for cluster, terms in matched:
        for term in terms:
            term["name"] = term["name"] or names.get(term["go_term"], term["go_term"])
            term["url"] = QUICKGO_TERM_URL.format(go_id=term["go_term"])
        rows.append({**cluster, "go_terms": terms})
    return rows


def _color_for(identifier: int) -> str:
    return UNIT_COLORS[int(identifier) % len(UNIT_COLORS)]


def _unit_prototypes(payload: Dict[str, Any]) -> Dict[int, int]:
    return {int(row["unit_id"]): int(row["prototype_id"]) for row in payload.get("unit_clusters", [])
            if row.get("prototype_id") is not None}


def _display_color(unit_id: int, view_mode: str, prototypes: Dict[int, int]) -> Tuple[str, str]:
    if view_mode == "Unit clusters" and unit_id in prototypes:
        prototype_id = prototypes[unit_id]
        return _color_for(prototype_id), f"Cluster {prototype_id}"
    return _color_for(unit_id), f"Unit {unit_id}"


def _color_swatch(color: str, label: str) -> str:
    return (f"<span title='{html_utils.escape(label)}' style='display:inline-block;width:14px;height:14px;"
            f"border-radius:3px;background:{color};border:1px solid #555;vertical-align:middle;margin-right:7px'></span>")

def _group_unit_clusters(unit_clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse local units assigned to the same global prototype for display."""
    grouped: Dict[int, Dict[str, Any]] = {}
    for row in unit_clusters:
        prototype_id = row.get("prototype_id")
        if prototype_id is None:
            continue
        prototype_id = int(prototype_id)
        group = grouped.setdefault(
            prototype_id,
            {"prototype_id": prototype_id, "unit_ids": [], "residue_count": 0, "go_terms": []},
        )
        group["unit_ids"].append(int(row["unit_id"]))
        group["residue_count"] += int(row.get("residue_count", 0))
        known_terms = {term["go_term"] for term in group["go_terms"]}
        group["go_terms"].extend(
            term for term in row.get("go_terms", []) if term["go_term"] not in known_terms
        )
        group["go_terms"] = sorted(
            group["go_terms"], key=lambda term: term.get("qval", float("inf"))
        )[:5]
    for group in grouped.values():
        group["unit_ids"].sort()
    return [grouped[prototype_id] for prototype_id in sorted(grouped)]


def _results_table_html(payload: Dict[str, Any], view_mode: str) -> str:
    prototypes = _unit_prototypes(payload)
    if view_mode == "Unit clusters":
        rows = []
        for cluster in _group_unit_clusters(payload.get("unit_clusters", [])):
            color = _color_for(cluster["prototype_id"])
            label = f"Cluster {cluster['prototype_id']}"
            unit_ids = ", ".join(str(unit_id) for unit_id in cluster["unit_ids"])
            terms = "<br>".join(
                f"<a href='{html_utils.escape(term.get('url', QUICKGO_TERM_URL.format(go_id=term['go_term'])))}' "
                f"target='_blank' rel='noopener noreferrer'>{html_utils.escape(term.get('name', term['go_term']))}</a> "
                f"({html_utils.escape(term['go_term'])}; q={term['qval']:.2e})"
                for term in cluster.get("go_terms", [])
            ) or "No enriched GO terms"
            rows.append(f"<tr><td>{_color_swatch(color, label)}{unit_ids}</td>"
                        f"<td>{cluster['prototype_id']}</td><td>{cluster['residue_count']}</td><td>{terms}</td></tr>")
        return """<table style='width:100%; border-collapse:collapse; font-family:sans-serif;'>
          <thead><tr><th style='text-align:left'>Units</th><th style='text-align:left'>Cluster</th>
          <th style='text-align:left'>Residues</th><th style='text-align:left'>Top GO functions</th></tr></thead>
          <tbody>""" + "".join(rows) + "</tbody></table>"
    rows = ""
    for unit in payload.get("units", []):
        color, label = _display_color(int(unit["unit_id"]), view_mode, prototypes)
        rows += (f"<tr><td>{_color_swatch(color, label)}{unit['unit_id']}</td>"
                 f"<td>{payload['chain']}</td><td>{unit['residue_count']}</td></tr>")
    return """<table style='width:100%; border-collapse:collapse; font-family:sans-serif;'>
      <thead><tr><th style='text-align:left'>Unit</th><th style='text-align:left'>Chain</th>
      <th style='text-align:left'>Residues</th></tr></thead><tbody>""" + rows + "</tbody></table>"

def _build_structure_html(
    payload: Dict[str, Any],
    view_mode: str = "Units",
    structure_display: str = "Selected chain only",
) -> str:
    structure = payload["structure"]
    assignments = payload.get("assignments", [])
    selected_chain = str(payload.get("chain", ""))
    pdb_path = Path(structure["path"])
    pdb_text = pdb_path.read_text(encoding="utf-8", errors="ignore")
    show_full_structure = structure_display == "Full PDB (other chains gray)"
    if not show_full_structure:
        selected_lines = [
            line
            for line in pdb_text.splitlines()
            if (
                (line.startswith("ATOM") and len(line) > 21 and line[21].strip() == selected_chain)
                or line.startswith("MODEL")
                or line.startswith("END")
            )
        ]
        pdb_text = "\n".join(selected_lines) + "\n"
    residue_numbers = structure.get("residue_numbers", [])
    if not residue_numbers:
        residue_numbers = list(range(1, max(1, structure.get("residue_count", 0)) + 1))

    prototypes = _unit_prototypes(payload)
    style_lines = []
    legend_items = []
    legend_labels = set()
    for unit_id in sorted({item["unit_id"] for item in assignments}):
        unit_assignments = [item for item in assignments if item["unit_id"] == unit_id]
        color, color_label = _display_color(int(unit_id), view_mode, prototypes)
        if color_label not in legend_labels:
            legend_items.append(f"<span style='display:inline-flex;align-items:center;margin:3px 12px 3px 0'>"
                                f"{_color_swatch(color, color_label)}{html_utils.escape(color_label)}</span>")
            legend_labels.add(color_label)
        for chain in sorted({item.get("chain", "") for item in unit_assignments}):
            unit_residues = [item["residue"] for item in unit_assignments if item.get("chain", "") == chain]
            selector = {"resi": unit_residues}
            if chain:
                selector["chain"] = chain
            style_lines.append(
                f"viewer.setStyle({json.dumps(selector)}, {{cartoon: {{color: '{color}'}}}});"
            )

    base_selector = {} if show_full_structure else {"chain": selected_chain}
    base_style_line = (
        f"viewer.setStyle({json.dumps(base_selector)}, "
        "{cartoon: {color: '#B0B0B0'}});"
    )

    viewer_document = f"""
            <div id="viewer" style="width:100%; height:420px; background:#fff;"></div>
            <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
            <script>
                (function() {{
                    const element = document.getElementById('viewer');
                    if (!element || typeof $3Dmol === 'undefined') return;
                    const viewer = $3Dmol.createViewer(element, {{backgroundColor: 'white'}});
                    const pdbText = {json.dumps(pdb_text)};
                    viewer.addModel(pdbText, 'pdb');
                    viewer.setStyle({{}}, {{}});
                    {base_style_line}
                    {''.join(style_lines)}
                    viewer.zoomTo();
                    viewer.render();
                }})();
            </script>
        """
    return f"""
        <div style="font-family: sans-serif;">
            <h3>{structure['name']}</h3>
            <p>Selected chain: {html_utils.escape(selected_chain)} |
               PDB chains: {', '.join(structure['chain_ids'])} |
               Residues: {structure['residue_count']}</p>
            <div style="margin:6px 0 10px">{''.join(legend_items) or 'No unit assignments'}</div>
            <iframe
                title="3D protein structure"
                srcdoc="{html_utils.escape(viewer_document, quote=True)}"
                style="width:100%; height:420px; border:1px solid #d0d7de; border-radius:8px; background:#fff;"
                sandbox="allow-scripts allow-same-origin"
            ></iframe>
        </div>
        """
def create_app() -> Any:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    def run_pipeline(
        pdb_file: str | None,
        esm_embeddings_file: str | None,
        esm_mode: str,
        chain: str,
        structure_display: str,
    ) -> Tuple[Any, Any, Any, str, str, Dict[str, Any]]:
        if pdb_file is None:
            return None, None, None, "Please upload a structure first.", "", {}

        structure = parse_pdb_structure(pdb_file)
        model_output = run_puffin(
            pdb_file,
            chain=chain,
            esm_embeddings_path=esm_embeddings_file,
            esm_mode=esm_mode,
        )
        payload = {"structure": structure, **model_output}
        table_html = _results_table_html(payload, "Units")
        run_id = uuid.uuid4().hex
        run_dir = RESULTS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        payload["run_id"] = run_id
        payload["result_directory"] = str(run_dir)
        input_path = run_dir / "input.pdb"
        input_path.write_bytes(Path(pdb_file).read_bytes())
        payload["structure"]["path"] = str(input_path)
        canonical_path = run_dir / f"{_chain_file_stem(pdb_file, model_output['chain'])}.pdb"
        canonicalize_pdb_chain(input_path, canonical_path, model_output["chain"])
        payload["canonical_pdb"] = str(canonical_path)
        results_text = build_download_payload(payload)
        results_path = run_dir / "results.json"
        results_path.write_text(results_text, encoding="utf-8")

        return (
            _build_structure_html(payload, "Units", structure_display),
            table_html,
            results_text,
            "Analysis complete",
            str(results_path),
            payload,
        )

    def render_views(
        payload: Dict[str, Any],
        unit_view: str,
        structure_display: str,
    ) -> Tuple[str, str]:
        if not payload:
            return "<p>Run the analysis first.</p>", "<p>Run the analysis first.</p>"
        return (
            _build_structure_html(payload, unit_view, structure_display),
            _results_table_html(payload, unit_view),
        )

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Protein Unit Explorer")
        gr.Markdown("Upload a protein structure to inspect predicted units and associated GO enrichment terms.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Inputs")
                pdb_input = gr.File(label="Protein structure (.pdb)", file_types=[".pdb"])
                esm_embeddings_input = gr.File(
                    label="Precomputed ESM-1b embeddings (optional .pt/.pth)",
                    file_types=[".pt", ".pth"],
                )
                esm_mode_input = gr.Radio(
                    ["Legacy zero embeddings", "Uploaded embeddings", "Compute ESM"],
                    value="Legacy zero embeddings",
                    label="ESM mode",
                    info=(
                        "Legacy mode reproduces the historical assignments using zero ESM "
                        "features."
                    ),
                )
                gr.Markdown(
                    "Upload format: a tensor `[N, 1280]`, or the dictionary produced by "
                    "`src/esm_embed.py` with `embeddings` and `index`. The file is used only "
                    "in **Uploaded embeddings** mode."
                )
                chain_input = gr.Textbox(value="", label="Chain (blank = first chain)")
                run_btn = gr.Button("Run analysis")
                download_file = gr.File(label="Downloadable results")

            with gr.Column(scale=2):
                gr.Markdown("## 3D structure")
                structure_display = gr.Radio(
                    ["Selected chain only", "Full PDB (other chains gray)"],
                    value="Selected chain only",
                    show_label=False,
                )
                structure_view = gr.HTML(value="<p>Upload a file to inspect the structure.</p>")

            with gr.Column(scale=1):
                gr.Markdown("## Predicted units")
                view_mode = gr.Radio(
                    ["Units", "Unit clusters"],
                    value="Units",
                    show_label=False,
                )
                results_table = gr.HTML(value="<p>Run the analysis to see predicted units here.</p>")
                results_json = gr.Textbox(label="Analysis output", lines=10, visible=False)
                status = gr.Textbox(label="Status", value="Waiting for input")
                results_state = gr.State({})

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                pdb_input,
                esm_embeddings_input,
                esm_mode_input,
                chain_input,
                structure_display,
            ],
            outputs=[structure_view, results_table, results_json, status, download_file, results_state],
        )
        view_mode.change(
            fn=render_views,
            inputs=[results_state, view_mode, structure_display],
            outputs=[structure_view, results_table],
        )
        structure_display.change(
            fn=render_views,
            inputs=[results_state, view_mode, structure_display],
            outputs=[structure_view, results_table],
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7865,
        share=False,
        show_error=True,
        inbrowser=False,
        prevent_thread_lock=False,
    )
