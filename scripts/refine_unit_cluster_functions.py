#!/usr/bin/env python3
"""
Export PUFFIN unit cluster centroids with refined GO function assignments.

The global unit-cluster pipeline stores centroid vectors and GO enrichment in
separate files:

  K1024/
    train_centroids.npy
    debias_transform.json
    assignments_train.csv
    eval/go_enrichment_train_all.csv

This script merges those outputs into a compact directory that can be uploaded
to Hugging Face and loaded at inference time to assign new unit embeddings to
their nearest unit cluster and enriched GO terms.

Examples
--------
Build the default artifact:

  python scripts/refine_unit_cluster_functions.py export \
    --unit-cluster-dir /cta/share/users/subclustgo/results/ismb26/prototypes/puffin_K64_v4/K1024 \
    --out-dir artifacts/puffin-unit-cluster-functions

Assign embeddings with the exported artifact:

  python scripts/refine_unit_cluster_functions.py infer \
    --artifact-dir artifacts/puffin-unit-cluster-functions \
    --embeddings unit_embeddings.npy \
    --out unit_cluster_function_assignments.csv
"""

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_UNIT_CLUSTER_DIR = Path(
    "/cta/share/users/subclustgo/results/ismb26/prototypes/puffin_K64_v4/K1024"
)
FORMAT_VERSION = "puffin-unit-cluster-functions-v1"
RUNTIME_MODULE = "puffin_unit_cluster_runtime.py"
SPLITS = ("train", "valid", "test")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}. Found: {list(df.columns)}")


def finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if pd.isna(value):
        return None
    return value


def table_value(column: str, value: Any) -> Any:
    if (
        column.endswith("_n_segments")
        or column.endswith("_n_proteins")
        or column in {"n_function_terms", "unit_cluster_id", "rank", "a_in_has", "unit_cluster_proteins", "total_proteins"}
    ):
        if value is not None and not pd.isna(value):
            return int(value)
    return finite_or_none(value)


def detect_model_name(unit_cluster_dir: Path, summary: Mapping[str, Any]) -> str:
    model_dir = summary.get("model_dir")
    if isinstance(model_dir, str) and model_dir:
        return Path(model_dir).name
    parent = unit_cluster_dir.parent.name
    return parent if parent else "puffin"


def detect_k(unit_cluster_dir: Path, centroids: np.ndarray, summary: Mapping[str, Any]) -> int:
    if "K" in summary:
        try:
            return int(summary["K"])
        except (TypeError, ValueError):
            pass
    if unit_cluster_dir.name.startswith("K"):
        try:
            return int(unit_cluster_dir.name[1:])
        except ValueError:
            pass
    return int(centroids.shape[0])


def load_term_names(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    require_file(path, "GO term-name table")
    if path.suffix.lower() == ".json":
        data = load_json(path)
        return {str(k): str(v) for k, v in data.items()}

    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    table = pd.read_csv(path, sep=sep)
    lower_to_col = {col.lower(): col for col in table.columns}
    id_col = lower_to_col.get("go_term") or lower_to_col.get("go_id") or lower_to_col.get("id")
    name_col = lower_to_col.get("name") or lower_to_col.get("term_name")
    if id_col is None or name_col is None:
        raise ValueError(
            f"{path} must contain GO ID/name columns such as go_term,name or id,name"
        )
    return dict(zip(table[id_col].astype(str), table[name_col].astype(str)))


def enrichment_path_for_split(
    unit_cluster_dir: Path, split: str, explicit_path: Optional[Path] = None
) -> Path:
    if explicit_path is not None:
        return require_file(explicit_path, "GO enrichment table")

    eval_dir = unit_cluster_dir / "eval"
    all_path = eval_dir / f"go_enrichment_{split}_all.csv"
    top_path = eval_dir / f"go_enrichment_{split}_top.csv"
    if all_path.exists():
        return all_path
    return require_file(top_path, f"GO enrichment table for split={split}")


def load_refined_functions(
    enrichment_path: Path,
    *,
    max_qval: float,
    top_terms_per_unit_cluster: int,
    min_in_unit_cluster: int,
    min_precision: float,
    min_odds_ratio: float,
    include_nonsignificant: bool,
    term_names: Mapping[str, str],
    source_split: str,
) -> pd.DataFrame:
    df = pd.read_csv(enrichment_path)
    required = [
        "proto",
        "go_term",
        "a_in_has",
        "proto_proteins",
        "total_proteins",
        "pval",
        "odds_ratio_approx",
        "qval",
    ]
    require_columns(df, required, str(enrichment_path))

    out = df.copy()
    out["unit_cluster_id"] = pd.to_numeric(out["proto"], errors="raise").astype(int)
    out["unit_cluster_proteins"] = pd.to_numeric(out["proto_proteins"], errors="coerce")
    for col in [
        "a_in_has",
        "unit_cluster_proteins",
        "total_proteins",
        "pval",
        "odds_ratio_approx",
        "qval",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["a_in_has"].fillna(0) >= int(min_in_unit_cluster)]
    out = out[out["unit_cluster_proteins"].fillna(0) > 0]
    out["term_precision"] = out["a_in_has"] / out["unit_cluster_proteins"]
    out = out[out["term_precision"].fillna(0.0) >= float(min_precision)]

    if min_odds_ratio > 0:
        out = out[out["odds_ratio_approx"].fillna(0.0) >= float(min_odds_ratio)]
    if not include_nonsignificant:
        out = out[out["qval"].fillna(1.0) <= float(max_qval)]

    sort_cols = ["unit_cluster_id", "qval", "pval", "odds_ratio_approx", "a_in_has"]
    out = out.sort_values(sort_cols, ascending=[True, True, True, False, False])
    out["rank"] = out.groupby("unit_cluster_id").cumcount() + 1
    out = out[out["rank"] <= int(top_terms_per_unit_cluster)].copy()
    out["source_split"] = source_split
    out["go_name"] = out["go_term"].astype(str).map(term_names).fillna("")

    preferred = [
        "unit_cluster_id",
        "rank",
        "go_term",
        "go_name",
        "source_split",
        "qval",
        "pval",
        "odds_ratio_approx",
        "a_in_has",
        "unit_cluster_proteins",
        "term_precision",
        "total_proteins",
        "b_in_not",
        "c_out_has",
        "d_out_not",
    ]
    cols = [col for col in preferred if col in out.columns]
    return out[cols].reset_index(drop=True)


def summarize_assignments(unit_cluster_dir: Path, k: int) -> pd.DataFrame:
    summary = pd.DataFrame({"unit_cluster_id": np.arange(k, dtype=int)})
    for split in SPLITS:
        path = unit_cluster_dir / f"assignments_{split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        require_columns(df, ["proto"], str(path))
        df["unit_cluster_id"] = pd.to_numeric(df["proto"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["unit_cluster_id"]).copy()
        df["unit_cluster_id"] = df["unit_cluster_id"].astype(int)

        agg: Dict[str, Tuple[str, str]] = {
            f"{split}_n_segments": ("unit_cluster_id", "size"),
        }
        if "protein_key" in df.columns:
            agg[f"{split}_n_proteins"] = ("protein_key", "nunique")
        if "assign_sim" in df.columns:
            agg[f"{split}_mean_assign_sim"] = ("assign_sim", "mean")
            agg[f"{split}_median_assign_sim"] = ("assign_sim", "median")
        if "n_residues_assigned" in df.columns:
            agg[f"{split}_mean_residues_assigned"] = ("n_residues_assigned", "mean")

        split_summary = df.groupby("unit_cluster_id", as_index=False).agg(**agg)
        summary = summary.merge(split_summary, on="unit_cluster_id", how="left")

    for col in summary.columns:
        if col.endswith("_n_segments") or col.endswith("_n_proteins"):
            summary[col] = summary[col].fillna(0).astype(int)
    return summary


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(k): finite_or_none(v) for k, v in row.items()})
    return records


def build_unit_cluster_records(
    *,
    k: int,
    functions: pd.DataFrame,
    support: pd.DataFrame,
) -> List[Dict[str, Any]]:
    term_records_by_unit_cluster: Dict[int, List[Dict[str, Any]]] = {}
    if not functions.empty:
        for unit_cluster_id, group in functions.groupby("unit_cluster_id"):
            term_records_by_unit_cluster[int(unit_cluster_id)] = dataframe_to_records(
                group.drop(columns=["unit_cluster_id"])
            )

    support_by_unit_cluster: Dict[int, Dict[str, Any]] = {}
    for _, row in support.iterrows():
        unit_cluster_id = int(row["unit_cluster_id"])
        support_by_unit_cluster[unit_cluster_id] = {
            str(col): table_value(str(col), value)
            for col, value in row.items()
            if col != "unit_cluster_id"
        }

    records: List[Dict[str, Any]] = []
    for unit_cluster_id in range(k):
        terms = term_records_by_unit_cluster.get(unit_cluster_id, [])
        records.append(
            {
                "unit_cluster_id": unit_cluster_id,
                "function_terms": terms,
                "n_function_terms": len(terms),
                "support": support_by_unit_cluster.get(unit_cluster_id, {}),
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def write_unit_cluster_summary(
    path: Path,
    *,
    k: int,
    functions: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = {int(unit_cluster_id): group for unit_cluster_id, group in functions.groupby("unit_cluster_id")}
    support_indexed = support.set_index("unit_cluster_id", drop=False)

    for unit_cluster_id in range(k):
        group = grouped.get(unit_cluster_id)
        row: Dict[str, Any] = {"unit_cluster_id": unit_cluster_id}
        if group is not None and not group.empty:
            top = group.sort_values("rank").iloc[0]
            row.update(
                {
                    "n_function_terms": int(len(group)),
                    "top_go_term": str(top["go_term"]),
                    "top_go_name": str(top.get("go_name", "")),
                    "top_qval": finite_or_none(top.get("qval")),
                    "top_term_precision": finite_or_none(top.get("term_precision")),
                    "go_terms": "|".join(group.sort_values("rank")["go_term"].astype(str).tolist()),
                }
            )
        else:
            row.update(
                {
                    "n_function_terms": 0,
                    "top_go_term": "",
                    "top_go_name": "",
                    "top_qval": None,
                    "top_term_precision": None,
                    "go_terms": "",
                }
            )
        if unit_cluster_id in support_indexed.index:
            for col, value in support_indexed.loc[unit_cluster_id].items():
                if col != "unit_cluster_id":
                    row[col] = table_value(str(col), value)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def apply_debias(embeddings: np.ndarray, debias: Mapping[str, Any]) -> np.ndarray:
    x = l2_normalize(embeddings.astype(np.float32, copy=False))
    mu = np.asarray(debias["mu"], dtype=np.float32).reshape(1, -1)
    pcs = np.asarray(debias.get("pcs", []), dtype=np.float32)
    if x.shape[1] != mu.shape[1]:
        raise ValueError(f"Embedding dim={x.shape[1]} does not match debias dim={mu.shape[1]}")
    x = x - mu
    if pcs.size:
        if pcs.ndim != 2 or pcs.shape[1] != x.shape[1]:
            raise ValueError(f"Invalid pcs shape={pcs.shape}; expected (*, {x.shape[1]})")
        for pc in pcs:
            pc_2d = pc.reshape(1, -1)
            x = x - (x @ pc_2d.T) * pc_2d
    return l2_normalize(x)


def assign_to_centroids(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    *,
    debias: Optional[Mapping[str, Any]],
    already_transformed: bool,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={embeddings.shape}")
    if centroids.ndim != 2:
        raise ValueError(f"Centroids must be 2D, got shape={centroids.shape}")

    if already_transformed:
        x = l2_normalize(embeddings.astype(np.float32, copy=False))
    else:
        if debias is None:
            raise ValueError("debias transform is required unless already_transformed=True")
        x = apply_debias(embeddings, debias)

    c = l2_normalize(centroids.astype(np.float32, copy=False))
    if x.shape[1] != c.shape[1]:
        raise ValueError(f"Embedding dim={x.shape[1]} does not match centroid dim={c.shape[1]}")

    unit_cluster_ids = np.empty((x.shape[0],), dtype=np.int64)
    sims = np.empty((x.shape[0],), dtype=np.float32)
    for start in range(0, x.shape[0], int(batch_size)):
        end = min(start + int(batch_size), x.shape[0])
        scores = x[start:end] @ c.T
        unit_cluster_ids[start:end] = np.argmax(scores, axis=1)
        sims[start:end] = np.max(scores, axis=1)
    return unit_cluster_ids, sims


def load_embeddings(path: Path, key: str) -> np.ndarray:
    require_file(path, "embedding file")
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if key not in obj.files:
            raise ValueError(f"{path} does not contain key={key!r}. Available keys: {obj.files}")
        return np.asarray(obj[key], dtype=np.float32)
    return np.asarray(obj, dtype=np.float32)


def load_artifact(artifact_dir: Path) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any], Dict[int, Any]]:
    config = load_json(require_file(artifact_dir / "config.json", "artifact config"))
    centroids = np.load(require_file(artifact_dir / "centroids.npy", "unit cluster centroids")).astype(np.float32)
    debias = load_json(require_file(artifact_dir / "debias_transform.json", "debias transform"))
    data = load_json(require_file(artifact_dir / "unit_clusters.json", "unit cluster function map"))
    records = data.get("unit_clusters", [])
    unit_cluster_map = {int(record["unit_cluster_id"]): record for record in records}
    return config, centroids, debias, unit_cluster_map


def terms_for_unit_cluster(
    unit_cluster_map: Mapping[int, Any],
    unit_cluster_id: int,
    *,
    top_n: Optional[int],
    max_qval: Optional[float],
) -> List[Dict[str, Any]]:
    record = unit_cluster_map.get(int(unit_cluster_id), {})
    terms = list(record.get("function_terms", []))
    if max_qval is not None:
        terms = [
            term
            for term in terms
            if term.get("qval") is not None
            and term.get("qval") != "inf"
            and float(term.get("qval")) <= float(max_qval)
        ]
    if top_n is not None:
        terms = terms[: int(top_n)]
    return terms


def term_string(terms: Sequence[Mapping[str, Any]]) -> str:
    return "|".join(str(term.get("go_term", "")) for term in terms if term.get("go_term"))


def write_runtime_module(path: Path) -> None:
    runtime = r'''#!/usr/bin/env python3
"""Standalone runtime loader for PUFFIN unit cluster function artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


class UnitClusterAssigner:
    """Assign unit embeddings to nearest unit clusters and enriched GO terms."""

    def __init__(
        self,
        centroids: np.ndarray,
        debias: Mapping[str, Any],
        unit_clusters: Mapping[int, Mapping[str, Any]],
    ) -> None:
        self.centroids = _l2_normalize(centroids.astype(np.float32, copy=False))
        self.debias = debias
        self.unit_clusters = {int(k): v for k, v in unit_clusters.items()}

    @classmethod
    def from_dir(cls, artifact_dir: Union[str, Path]) -> "UnitClusterAssigner":
        root = Path(artifact_dir)
        centroids = np.load(root / "centroids.npy").astype(np.float32)
        debias = _load_json(root / "debias_transform.json")
        data = _load_json(root / "unit_clusters.json")
        unit_clusters = {int(row["unit_cluster_id"]): row for row in data["unit_clusters"]}
        return cls(centroids=centroids, debias=debias, unit_clusters=unit_clusters)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        x = _l2_normalize(embeddings.astype(np.float32, copy=False))
        mu = np.asarray(self.debias["mu"], dtype=np.float32).reshape(1, -1)
        pcs = np.asarray(self.debias.get("pcs", []), dtype=np.float32)
        if x.shape[1] != mu.shape[1]:
            raise ValueError(f"Embedding dim={x.shape[1]} does not match debias dim={mu.shape[1]}")
        x = x - mu
        if pcs.size:
            for pc in pcs:
                pc_2d = pc.reshape(1, -1)
                x = x - (x @ pc_2d.T) * pc_2d
        return _l2_normalize(x)

    def assign_arrays(
        self,
        embeddings: np.ndarray,
        *,
        already_transformed: bool = False,
        batch_size: int = 8192,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = _l2_normalize(embeddings.astype(np.float32, copy=False)) if already_transformed else self.transform(embeddings)
        if x.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"Embedding dim={x.shape[1]} does not match centroid dim={self.centroids.shape[1]}"
            )
        unit_cluster_ids = np.empty((x.shape[0],), dtype=np.int64)
        sims = np.empty((x.shape[0],), dtype=np.float32)
        for start in range(0, x.shape[0], int(batch_size)):
            end = min(start + int(batch_size), x.shape[0])
            scores = x[start:end] @ self.centroids.T
            unit_cluster_ids[start:end] = np.argmax(scores, axis=1)
            sims[start:end] = np.max(scores, axis=1)
        return unit_cluster_ids, sims

    def terms_for_unit_cluster(
        self,
        unit_cluster_id: int,
        *,
        top_n: Optional[int] = None,
        max_qval: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        terms = list(self.unit_clusters.get(int(unit_cluster_id), {}).get("function_terms", []))
        if max_qval is not None:
            terms = [
                term
                for term in terms
                if term.get("qval") is not None
                and term.get("qval") != "inf"
                and float(term["qval"]) <= float(max_qval)
            ]
        return terms[:top_n] if top_n is not None else terms

    def assign(
        self,
        embeddings: np.ndarray,
        *,
        top_n: Optional[int] = 5,
        max_qval: Optional[float] = None,
        already_transformed: bool = False,
        batch_size: int = 8192,
    ) -> List[Dict[str, Any]]:
        unit_cluster_ids, sims = self.assign_arrays(
            embeddings,
            already_transformed=already_transformed,
            batch_size=batch_size,
        )
        return [
            {
                "row_index": int(i),
                "unit_cluster_id": int(unit_cluster_id),
                "assign_sim": float(sim),
                "function_terms": self.terms_for_unit_cluster(
                    int(unit_cluster_id), top_n=top_n, max_qval=max_qval
                ),
            }
            for i, (unit_cluster_id, sim) in enumerate(zip(unit_cluster_ids, sims))
        ]
'''
    path.write_text(runtime, encoding="utf-8")


def write_model_card(
    path: Path,
    *,
    model_name: str,
    k: int,
    embedding_dim: int,
    go_aspect: str,
    source_split: str,
    max_qval: float,
    top_terms_per_unit_cluster: int,
    source_result: str,
) -> None:
    text = f"""---
library_name: numpy
tags:
- protein
- gene-ontology
- clustering
- puffin
- subclustgo
---

# {model_name} K{k} Unit Cluster Functions

This artifact maps PUFFIN unit embeddings to global unit clusters and
unit-cluster-level enriched GO terms.

Source result: `{source_result}`

`train_centroids.npy` from the source run contains centroid vectors only. GO
enrichment is stored separately here in `unit_cluster_functions.csv` and
`unit_clusters.json`.

## Files

- `centroids.npy`: `(K, H)` normalized unit cluster centroids, `K={k}`, `H={embedding_dim}`.
- `debias_transform.json`: train-fitted preprocessing used before centroid assignment.
- `unit_cluster_functions.csv`: one row per retained unit-cluster/GO-term assignment.
- `unit_cluster_summary.csv`: one row per unit cluster with support and top GO terms.
- `unit_clusters.json`: runtime-friendly unit cluster function map.
- `puffin_unit_cluster_runtime.py`: standalone NumPy loader for inference.
- `unit_cluster_artifact.npz`: compressed bundle with centroids, debias mean, and debias PCs.

## Function Assignment Filter

- Source split: `{source_split}`
- GO aspect: `{go_aspect}`
- Max q-value: `{max_qval}`
- Top terms per unit cluster: `{top_terms_per_unit_cluster}`

## Inference

```python
import numpy as np
from puffin_unit_cluster_runtime import UnitClusterAssigner

assigner = UnitClusterAssigner.from_dir(".")
unit_embeddings = np.load("unit_embeddings.npy")
assignments = assigner.assign(unit_embeddings, top_n=5, max_qval={max_qval})
```

New embeddings should be raw unit embeddings from the same PUFFIN representation
used to fit these unit clusters. The runtime applies the train-fitted debias
transform before cosine nearest-centroid assignment.
"""
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


STALE_ARTIFACT_FILES = (
    "prototype_functions.csv",
    "prototype_support.csv",
    "prototype_summary.csv",
    "prototypes.json",
    "prototypes.jsonl",
    "prototype_artifact.npz",
    "source_summary.json",
)


def cleanup_stale_artifact_files(out_dir: Path) -> None:
    for name in STALE_ARTIFACT_FILES:
        path = out_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def export_artifact(args: argparse.Namespace) -> int:
    unit_cluster_dir = args.unit_cluster_dir.resolve()
    out_dir = ensure_dir(args.out_dir.resolve())

    centroids_path = require_file(unit_cluster_dir / "train_centroids.npy", "train centroids")
    debias_path = require_file(unit_cluster_dir / "debias_transform.json", "debias transform")
    summary_path = unit_cluster_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    centroids = np.load(centroids_path).astype(np.float32)
    if centroids.ndim != 2:
        raise ValueError(f"Expected 2D centroids in {centroids_path}, got {centroids.shape}")
    k = detect_k(unit_cluster_dir, centroids, summary)
    if k != centroids.shape[0]:
        raise ValueError(f"K={k} but centroids have {centroids.shape[0]} rows")

    debias = load_json(debias_path)
    mu = np.asarray(debias.get("mu"), dtype=np.float32).reshape(1, -1)
    pcs = np.asarray(debias.get("pcs", []), dtype=np.float32)
    if centroids.shape[1] != mu.shape[1]:
        raise ValueError(
            f"Centroid dim={centroids.shape[1]} does not match debias dim={mu.shape[1]}"
        )

    model_name = args.model_name or detect_model_name(unit_cluster_dir, summary)
    go_aspect = str(
        args.go_aspect
        or summary.get("config", {}).get("go_aspect")
        or "MF"
    ).upper()

    term_names = load_term_names(args.go_term_names)
    enrichment_path = enrichment_path_for_split(
        unit_cluster_dir, args.source_split, args.enrichment_file
    )
    functions = load_refined_functions(
        enrichment_path,
        max_qval=args.max_qval,
        top_terms_per_unit_cluster=args.top_terms_per_unit_cluster,
        min_in_unit_cluster=args.min_in_unit_cluster,
        min_precision=args.min_precision,
        min_odds_ratio=args.min_odds_ratio,
        include_nonsignificant=args.include_nonsignificant,
        term_names=term_names,
        source_split=args.source_split,
    )
    support = summarize_assignments(unit_cluster_dir, k)

    cleanup_stale_artifact_files(out_dir)
    shutil.copy2(centroids_path, out_dir / "centroids.npy")
    shutil.copy2(debias_path, out_dir / "debias_transform.json")

    functions.to_csv(out_dir / "unit_cluster_functions.csv", index=False)
    support.to_csv(out_dir / "unit_cluster_support.csv", index=False)
    unit_cluster_summary = write_unit_cluster_summary(
        out_dir / "unit_cluster_summary.csv",
        k=k,
        functions=functions,
        support=support,
    )

    unit_cluster_records = build_unit_cluster_records(k=k, functions=functions, support=support)
    write_json(
        out_dir / "unit_clusters.json",
        {
            "format": FORMAT_VERSION,
            "model_name": model_name,
            "K": k,
            "embedding_dim": int(centroids.shape[1]),
            "go_aspect": go_aspect,
            "source_split": args.source_split,
            "unit_clusters": unit_cluster_records,
        },
    )
    write_jsonl(out_dir / "unit_clusters.jsonl", unit_cluster_records)

    np.savez_compressed(
        out_dir / "unit_cluster_artifact.npz",
        centroids=centroids.astype(np.float32),
        debias_mu=mu.astype(np.float32),
        debias_pcs=pcs.astype(np.float32),
    )

    config = {
        "format": FORMAT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_result": f"{model_name}/K{k}",
        "source_enrichment_file": enrichment_path.name,
        "model_name": model_name,
        "K": k,
        "embedding_dim": int(centroids.shape[1]),
        "assignment_metric": "cosine",
        "preprocessing": {
            "name": "l2_normalize_subtract_train_mean_remove_train_pcs_l2_normalize",
            "debias_transform_file": "debias_transform.json",
            "remove_pcs": int(debias.get("remove_pcs", len(pcs))),
        },
        "function_assignment": {
            "source_split": args.source_split,
            "go_aspect": go_aspect,
            "max_qval": args.max_qval,
            "top_terms_per_unit_cluster": args.top_terms_per_unit_cluster,
            "min_in_unit_cluster": args.min_in_unit_cluster,
            "min_precision": args.min_precision,
            "min_odds_ratio": args.min_odds_ratio,
            "include_nonsignificant": bool(args.include_nonsignificant),
        },
        "files": {
            "centroids": "centroids.npy",
            "debias_transform": "debias_transform.json",
            "functions_csv": "unit_cluster_functions.csv",
            "support_csv": "unit_cluster_support.csv",
            "summary_csv": "unit_cluster_summary.csv",
            "unit_clusters_json": "unit_clusters.json",
            "runtime": RUNTIME_MODULE,
            "npz_bundle": "unit_cluster_artifact.npz",
        },
        "stats": {
            "n_unit_clusters": k,
            "n_unit_clusters_with_functions": int((unit_cluster_summary["n_function_terms"] > 0).sum()),
            "n_function_rows": int(len(functions)),
        },
    }
    write_json(out_dir / "config.json", config)
    write_runtime_module(out_dir / RUNTIME_MODULE)
    write_model_card(
        out_dir / "README.md",
        model_name=model_name,
        k=k,
        embedding_dim=int(centroids.shape[1]),
        go_aspect=go_aspect,
        source_split=args.source_split,
        max_qval=args.max_qval,
        top_terms_per_unit_cluster=args.top_terms_per_unit_cluster,
        source_result=f"{model_name}/K{k}",
    )

    print(f"Wrote artifact: {out_dir}")
    print(f"  centroids: {centroids.shape}")
    print(f"  function rows: {len(functions)}")
    print(f"  unit clusters with functions: {config['stats']['n_unit_clusters_with_functions']}/{k}")
    return 0


def infer_assignments(args: argparse.Namespace) -> int:
    _, centroids, debias, unit_cluster_map = load_artifact(args.artifact_dir.resolve())
    embeddings = load_embeddings(args.embeddings.resolve(), args.embedding_key)
    unit_cluster_ids, sims = assign_to_centroids(
        embeddings,
        centroids,
        debias=debias,
        already_transformed=args.already_transformed,
        batch_size=args.batch_size,
    )

    rows: List[Dict[str, Any]] = []
    for i, (unit_cluster_id, sim) in enumerate(zip(unit_cluster_ids, sims)):
        terms = terms_for_unit_cluster(
            unit_cluster_map,
            int(unit_cluster_id),
            top_n=args.top_n,
            max_qval=args.max_qval,
        )
        rows.append(
            {
                "row_index": i,
                "unit_cluster_id": int(unit_cluster_id),
                "assign_sim": float(sim),
                "go_terms": term_string(terms),
                "function_terms_json": json.dumps(terms, sort_keys=True),
            }
        )

    out = pd.DataFrame(rows)
    if args.metadata is not None:
        meta = pd.read_csv(args.metadata)
        if len(meta) != len(out):
            raise ValueError(
                f"Metadata rows={len(meta)} do not match embeddings rows={len(out)}"
            )
        out = pd.concat([meta.reset_index(drop=True), out], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote assignments: {args.out} ({len(out)} rows)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine/export PUFFIN unit cluster GO assignments and run centroid inference.",
    )
    subparsers = parser.add_subparsers(dest="command")

    export = subparsers.add_parser(
        "export",
        help="Create a Hugging Face-ready unit-cluster/function artifact.",
    )
    export.add_argument("--unit-cluster-dir", type=Path, default=DEFAULT_UNIT_CLUSTER_DIR)
    export.add_argument("--prototype-dir", dest="unit_cluster_dir", type=Path, help=argparse.SUPPRESS)
    export.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/puffin-unit-cluster-functions"),
    )
    export.add_argument("--source-split", choices=SPLITS, default="train")
    export.add_argument("--enrichment-file", type=Path, default=None)
    export.add_argument("--model-name", default=None)
    export.add_argument("--go-aspect", default=None, choices=["MF", "BP", "CC", "mf", "bp", "cc"])
    export.add_argument("--go-term-names", type=Path, default=None)
    export.add_argument("--max-qval", type=float, default=0.05)
    export.add_argument("--top-terms-per-unit-cluster", dest="top_terms_per_unit_cluster", type=int, default=10)
    export.add_argument("--top-terms-per-proto", dest="top_terms_per_unit_cluster", type=int, help=argparse.SUPPRESS)
    export.add_argument("--min-in-unit-cluster", dest="min_in_unit_cluster", type=int, default=1)
    export.add_argument("--min-in-proto", dest="min_in_unit_cluster", type=int, help=argparse.SUPPRESS)
    export.add_argument("--min-precision", type=float, default=0.0)
    export.add_argument("--min-odds-ratio", type=float, default=1.0)
    export.add_argument(
        "--include-nonsignificant",
        action="store_true",
        help="Keep top enriched terms even when qval is above --max-qval.",
    )
    export.set_defaults(func=export_artifact)

    infer = subparsers.add_parser(
        "infer",
        help="Assign unit embeddings to exported unit clusters and functions.",
    )
    infer.add_argument("--artifact-dir", type=Path, required=True)
    infer.add_argument("--embeddings", type=Path, required=True)
    infer.add_argument("--embedding-key", default="embeddings")
    infer.add_argument("--metadata", type=Path, default=None)
    infer.add_argument("--out", type=Path, required=True)
    infer.add_argument("--top-n", type=int, default=5)
    infer.add_argument("--max-qval", type=float, default=None)
    infer.add_argument("--batch-size", type=int, default=8192)
    infer.add_argument(
        "--already-transformed",
        action="store_true",
        help="Skip debias preprocessing; use only L2 normalization before centroid assignment.",
    )
    infer.set_defaults(func=infer_assignments)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] not in {"export", "infer", "-h", "--help"}:
        args_in = ["export"] + args_in
    parser = build_parser()
    args = parser.parse_args(args_in)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
