#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="ismb26/models"
ESM_PATH="/cta/share/users/esm/ESM-1b"

TRAIN_FILE="data/GeneOntology/nrPDB-GO_train.txt"
VAL_FILE="data/GeneOntology/nrPDB-GO_valid.txt"
TEST_FILE="data/GeneOntology/nrPDB-GO_test.txt"

OUT_ROOT="ismb26/segments"

# Round-robin GPUs
GPUS=(0 1 2)
gpu_i=0

for d in "${BASE_DIR}"/*; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"

  # infer K from folder name *_K<number>*
  if [[ "$name" =~ _K([0-9]+) ]]; then
    K="${BASH_REMATCH[1]}"
  else
    echo "Skipping $name (cannot infer K from name)"
    continue
  fi

  # pick checkpoint
  CKPT="$(ls -t "$d"/epoch_*.ckpt 2>/dev/null | head -n 1 || true)"
  if [[ -z "$CKPT" ]]; then
    echo "Skipping $name (no ckpt found)"
    continue
  fi
  

  GPU="${GPUS[$gpu_i]}"
  gpu_i=$(( (gpu_i + 1) % ${#GPUS[@]} ))

  for split in train valid test; do
    case "$split" in
      train) INFILE="$TRAIN_FILE" ;;
      valid)   INFILE="$VAL_FILE" ;;
      test)  INFILE="$TEST_FILE" ;;
    esac

    OUTDIR="${OUT_ROOT}/${name}/${split}"
    mkdir -p "$OUTDIR"

    echo "=== $name | K=$K | split=$split | GPU=$GPU ==="

    CUDA_VISIBLE_DEVICES="$GPU" python src/cluster.py \
      name="cluster_${name}" \
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
      encoder.esm_model_path="$ESM_PATH" \
      ckpt_path="$CKPT" \
      cluster.input_file="$INFILE" \
      cluster.split="$split" \
      cluster.output_file="$split" \
      output_dir="$OUTDIR"
  done
done
