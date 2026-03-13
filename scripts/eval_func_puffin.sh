#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="ismb26/models"
OUT_DIR="ismb26/results/func_eval"
mkdir -p "$OUT_DIR"

# Same split files as clustering (edit if needed)
TRAIN_FILE="data/GeneOntology/nrPDB-GO_train.txt"
VAL_FILE="data/GeneOntology/nrPDB-GO_val.txt"
TEST_FILE="data/GeneOntology/nrPDB-GO_test.txt"

# GPU assignment (round-robin)
GPUS=(0 1 2)
gpu_i=0

for d in "${BASE_DIR}"/puffin_*; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"

  # infer K from folder name e.g., puffin_K64_v2 -> 64
  if [[ "$name" =~ _K([0-9]+) ]]; then
    K="${BASH_REMATCH[1]}"
  else
    echo "[WARN] Skipping $name (cannot infer K from folder name)"
    continue
  fi

  # choose checkpoint

  CKPT="$(ls -t "$d"/epoch_*.ckpt 2>/dev/null | head -n 1 || true)"
  if [[ -z "$CKPT" ]]; then
    echo "[WARN] Skipping $name (no ckpt found)"
    continue
  fi
  

  GPU="${GPUS[$gpu_i]}"
  gpu_i=$(( (gpu_i + 1) % ${#GPUS[@]} ))


  LOG="${OUT_DIR}/${name}.log"
  echo "=== EVAL $name | K=$K | GPU=$GPU ==="
  echo "ckpt: $CKPT"
  echo "log : $LOG"

  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="$GPU" python src/eval.py \
    name="$name" \
    encoder=puffin \
    encoder.gnn_type=GAT \
    encoder.hidden_dim=512 \
    encoder.num_clusters="$K" \
    encoder.num_res_gnn_layers=2 \
    encoder.num_seg_gnn_layers=2 \
    encoder.use_seg_res_cross_attn=false \
    encoder.proj_layer=true \
    encoder.esm_embed_dim=512 \
    encoder.input_feat_dim=512 \
    encoder.fuse_lm_method=sum \
    objective_type=dual \
    ckpt_path="$CKPT" \
    >> "$LOG" 2>&1
done
