# Scripts

## Unit Cluster-Function Associations

Unit cluster-function associations are exported from the GO enrichment tables created during global unit cluster learning. The export step packages train-fitted unit-cluster centroids, the debias transform, support summaries, and retained GO-term associations into a directory that can be used during inference.

To export the associations into an inference-ready artifact:

```bash
python scripts/refine_unit_cluster_functions.py export \
  --unit-cluster-dir ismb26/unit_clusters/puffin_K64/K1024 \
  --out-dir artifacts/puffin-unit-cluster-functions \
  --source-split train \
  --go-aspect MF \
  --max-qval 0.05 \
  --top-terms-per-unit-cluster 10
```

`--prototype-dir` and `--top-terms-per-proto` are still accepted as legacy aliases, but new commands should use the unit-cluster option names.

The export step filters the enrichment table by q-value, support, precision, odds ratio, and rank, then writes:

* `centroids.npy` and `debias_transform.json` for assigning new unit embeddings
* `unit_cluster_functions.csv`: one row per retained unit cluster-GO association
* `unit_cluster_summary.csv`: one row per unit cluster with support and top GO terms
* `unit_clusters.json` and `unit_clusters.jsonl`: runtime-friendly unit cluster-function maps
* `puffin_unit_cluster_runtime.py`: standalone loader for assigning new unit embeddings to clusters and attached GO terms

## Single-PDB Inference With Functions

Pass the exported artifact to `scripts/infer_puffin_units.py` to assign each active PUFFIN unit to its nearest global unit cluster and attach that cluster's retained GO terms:

```bash
python scripts/infer_puffin_units.py path/to/protein.pdb \
  --chain A \
  --output-dir units/single_pdb/ \
  --unit-cluster-artifact artifacts/puffin-unit-cluster-functions \
  --unit-cluster-top-n 5 \
  --unit-cluster-max-qval 0.05
```

At inference time, each active unit embedding is transformed with the train-fitted debias transform, assigned to the nearest unit-cluster centroid by cosine similarity, and annotated with the retained GO terms for that cluster.

This writes the standard PUFFIN unit outputs plus `<pdb>_puffin_unit_cluster_functions.csv`, which contains one row per active PUFFIN unit with `unit_cluster_id`, `unit_cluster_similarity`, `go_terms`, `top_go_term`, and `function_terms_json`. The residue and unit metadata CSVs also include the unit-cluster/function columns when an artifact is provided.
