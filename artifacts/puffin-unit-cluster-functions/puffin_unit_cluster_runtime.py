#!/usr/bin/env python3
"""Standalone runtime loader for PUFFIN unit cluster function artifacts."""

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
