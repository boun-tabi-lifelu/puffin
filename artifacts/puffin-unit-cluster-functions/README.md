---
library_name: numpy
tags:
- protein
- gene-ontology
- clustering
- puffin
- subclustgo
---

# puffin_K64_v4 K1024 Unit Cluster Functions

This artifact maps PUFFIN unit embeddings to global unit clusters and
unit-cluster-level enriched GO terms.

Source result: `puffin_K64_v4/K1024`

`train_centroids.npy` from the source run contains centroid vectors only. GO
enrichment is stored separately here in `unit_cluster_functions.csv` and
`unit_clusters.json`.

## Files

- `centroids.npy`: `(K, H)` normalized unit cluster centroids, `K=1024`, `H=512`.
- `debias_transform.json`: train-fitted preprocessing used before centroid assignment.
- `unit_cluster_functions.csv`: one row per retained unit-cluster/GO-term assignment.
- `unit_cluster_summary.csv`: one row per unit cluster with support and top GO terms.
- `unit_clusters.json`: runtime-friendly unit cluster function map.
- `puffin_unit_cluster_runtime.py`: standalone NumPy loader for inference.
- `unit_cluster_artifact.npz`: compressed bundle with centroids, debias mean, and debias PCs.

## Function Assignment Filter

- Source split: `train`
- GO aspect: `MF`
- Max q-value: `0.05`
- Top terms per unit cluster: `10`

## Inference

```python
import numpy as np
from puffin_unit_cluster_runtime import UnitClusterAssigner

assigner = UnitClusterAssigner.from_dir(".")
unit_embeddings = np.load("unit_embeddings.npy")
assignments = assigner.assign(unit_embeddings, top_n=5, max_qval=0.05)
```

New embeddings should be raw unit embeddings from the same PUFFIN representation
used to fit these unit clusters. The runtime applies the train-fitted debias
transform before cosine nearest-centroid assignment.
