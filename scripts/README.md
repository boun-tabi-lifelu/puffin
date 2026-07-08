# Scripts

## Unit Cluster-Function Associations

Unit cluster-function associations are exported from the GO enrichment tables created during global unit cluster learning. The export step packages train-fitted unit-cluster centroids, the debias transform, support summaries, and retained GO-term associations into a directory that can be used during inference.

To export the associations into an inference-ready artifact:

```bash
python scripts/refine_unit_cluster_functions.py export \
  --prototype-dir ismb26/unit_clusters/puffin_K64/K1024 \
  --out-dir artifacts/puffin_K64_K1024_unit_functions \
  --source-split train \
  --go-aspect MF \
  --max-qval 0.05 \
  --top-terms-per-proto 10
```

`--prototype-dir` and output files such as `prototype_functions.csv` are legacy names in the current scripts; they refer to unit-cluster centroids and unit cluster-function associations.

The export step filters the enrichment table by q-value, support, precision, odds ratio, and rank, then writes:

* `centroids.npy` and `debias_transform.json` for assigning new unit embeddings
* `prototype_functions.csv`: one row per retained unit cluster-GO association
* `prototype_summary.csv`: one row per unit cluster with support and top GO terms
* `prototypes.json` and `prototypes.jsonl`: runtime-friendly unit cluster-function maps
* `puffin_unit_cluster_runtime.py`: standalone loader for assigning new unit embeddings to clusters and attached GO terms

At inference time, new unit embeddings are transformed with the train-fitted debias transform, assigned to the nearest unit-cluster centroid by cosine similarity, and annotated with the retained GO terms for that cluster.
