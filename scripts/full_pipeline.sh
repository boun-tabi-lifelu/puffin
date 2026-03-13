#!/usr/bin/env bash
set -euo pipefail

# =========================
# 0) Common settings
# =========================
export HYDRA_FULL_ERROR=1

DATA_DIR="${DATA_DIR:-data/GeneOntology}"
ESM_PATH="${ESM_PATH:-/cta/share/users/esm/ESM-1b}"
RESULTS_DIR="${RESULTS_DIR:-results}"

TEST_LIST="${DATA_DIR}/nrPDB-GO_test.txt"
INTERPRO_JSON="${DATA_DIR}/test_interpro.json"

GO_ASPECT="${GO_ASPECT:-MF}"
MIN_ASSIGNED="${MIN_ASSIGNED:-5}"
KNN_K="${KNN_K:-20}"
KNN_SEARCH_K="${KNN_SEARCH_K:-220}"
ANNOT_TYPES="${ANNOT_TYPES:-active_site,binding_site,conserved_site,domain,repeats}"

# GPUs (override via env)
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_CLUSTER="${GPU_CLUSTER:-1}"
GPU_ESM="${GPU_ESM:-2}"

# =========================
# 1) Shared Hydra overrides
# =========================
MODEL=(
  encoder=protygus
  encoder.gnn_type=GAT
  encoder.hidden_dim=512
  encoder.num_clusters=64
  encoder.num_res_gnn_layers=2
  encoder.num_seg_gnn_layers=2
  encoder.use_seg_res_cross_attn=false
  encoder.proj_layer=true
  encoder.esm_embed_dim=512
  encoder.input_feat_dim=512
  encoder.fuse_lm_method=sum
  objective_type=dual
)

EVAL_ARGS=(
  --annotation_dir "${DATA_DIR}"
  --min_assigned "${MIN_ASSIGNED}"
  --knn_k "${KNN_K}"
  --knn_search_k "${KNN_SEARCH_K}"
  --exclude_same_protein
  --knn_backend auto
  --device cuda
  --go_aspect "${GO_ASPECT}"
  --silhouette
  --silhouette_metric cosine
  --silhouette_sample_max 20000
  --unit_eval
  --interpro_json "${INTERPRO_JSON}"
  --unit_eval_K 64
  --annotation_types "${ANNOT_TYPES}"
)

# =========================
# 2) Helper: run eval + prototypes
# =========================
run_eval_and_protos () {
  local cluster_dir="$1"
  local prefix="$2"
  local eval_dir="$3"
  local proto_dir="$4"
  local proto_k="$5"

  python src/cluster_eval.py \
    --cluster_dir "${cluster_dir}" \
    --output_dir "${eval_dir}" \
    --prefix "${prefix}" \
    "${EVAL_ARGS[@]}"

  python src/global_prototypes.py \
    --cluster_dir "${cluster_dir}" \
    --prefix "${prefix}" \
    --annotation_dir "${DATA_DIR}" \
    --go_aspect "${GO_ASPECT}" \
    --min_assigned "${MIN_ASSIGNED}" \
    --max_per_protein 50 \
    --method spherical_kmeans \
    --k "${proto_k}" \
    --remove_pcs 2 \
    --eval_top_terms 10 \
    --stability_runs 10 \
    --stability_frac_proteins 0.8 \
    --proto_knn_k 20 \
    --proto_min_go_terms 1 \
    --proto_exclude_shared_proteins \
    --out_dir "${proto_dir}"
}

# =========================
# 3) Protygus (supervised dual)
# =========================
CKPT_MODEL="${CKPT_MODEL:-/cta/users/guludogan/SubClustGO/models/checkpoints/epoch_009-v1.ckpt}"
OUT_MODEL="${RESULTS_DIR}/protygus_test_output"
PREFIX_MODEL="protygus_test"

CUDA_VISIBLE_DEVICES="${GPU_CLUSTER}" python src/cluster.py \
  name=protygus_test_run \
  "${MODEL[@]}" \
  encoder.esm_model_path="${ESM_PATH}" \
  ckpt_path="${CKPT_MODEL}" \
  cluster.input_file="${TEST_LIST}" \
  cluster.split=test \
  cluster.output_file="${PREFIX_MODEL}" \
  output_dir="${OUT_MODEL}"

run_eval_and_protos \
  "${OUT_MODEL}" \
  "${PREFIX_MODEL}" \
  "${RESULTS_DIR}/protygus_test_eval" \
  "${RESULTS_DIR}/protygus_global_proto_k128" \
  128

# =========================
# 4) Mincut (unsupervised)
# =========================
CKPT_MINCUT="${CKPT_MINCUT:-/cta/users/guludogan/SubClustGO/models/protygus_mincut/checkpoints/epoch_009.ckpt}"
OUT_MINCUT="${RESULTS_DIR}/mincut_output"
PREFIX_MINCUT="mincut_test"

CUDA_VISIBLE_DEVICES="${GPU_CLUSTER}" python src/cluster.py \
  name=protygus_mincut_run \
  "${MODEL[@]}" \
  encoder.esm_model_path="${ESM_PATH}" \
  ckpt_path="${CKPT_MINCUT}" \
  cluster.input_file="${TEST_LIST}" \
  cluster.split=test \
  cluster.output_file="${PREFIX_MINCUT}" \
  output_dir="${OUT_MINCUT}"

run_eval_and_protos \
  "${OUT_MINCUT}" \
  "${PREFIX_MINCUT}" \
  "${RESULTS_DIR}/mincut_test_eval" \
  "${RESULTS_DIR}/mincut_global_proto_k256" \
  256

# =========================
# 5) Baselines: Louvain + ESM KMeans
# =========================
OUT_LOU="${RESULTS_DIR}/louvain_test_output"
PREFIX_LOU="louvain_test"

CUDA_VISIBLE_DEVICES="${GPU_CLUSTER}" python src/cluster.py \
  cluster.model=baseline_model \
  cluster.input_file="${TEST_LIST}" \
  cluster.split=test \
  cluster.output_file="${PREFIX_LOU}" \
  output_dir="${OUT_LOU}" \
  baseline_model.algorithm=louvain \
  ckpt_path=""

python src/cluster_eval.py \
  --cluster_dir "${OUT_LOU}" \
  --output_dir "${RESULTS_DIR}/louvain_test_eval" \
  --prefix "${PREFIX_LOU}" \
  "${EVAL_ARGS[@]}"

OUT_ESM="${RESULTS_DIR}/esm_kmeans_output"
PREFIX_ESM="esm_test"

CUDA_VISIBLE_DEVICES="${GPU_ESM}" python src/cluster.py \
  cluster.model=baseline_model \
  cluster.input_file="${TEST_LIST}" \
  cluster.split=test \
  cluster.output_file="${PREFIX_ESM}" \
  output_dir="${OUT_ESM}" \
  baseline_model.algorithm=kmeans \
  ckpt_path=""

run_eval_and_protos \
  "${OUT_ESM}" \
  "${PREFIX_ESM}" \
  "${RESULTS_DIR}/esm_kmeans_eval" \
  "${RESULTS_DIR}/esm_kmeans_global_proto_k256" \
  256

echo "✅ Done."
