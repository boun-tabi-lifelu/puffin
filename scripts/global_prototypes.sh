#!/usr/bin/env bash
# scripts/global_prototypes.sh
# ------------------------------------------------------------------------------
# Auto-discover model dirs under segments/ and run global prototypes:
# - fit on train
# - assign valid/test
# - sweep K: 128,256,512,1024,2048,4096
#
# Usage:
#   bash scripts/global_prototypes.sh \
#     --segments_root segments \
#     --annotation_dir data/GeneOntology \
#     --out_root results/global_prototypes \
#     --go_aspect MF \
#     --k_list 128,256,512,1024,2048,4096
# ------------------------------------------------------------------------------

set -euo pipefail

SEGMENTS_ROOT="ismb26/segments"
ANNOTATION_DIR="data/GeneOntology"
OUT_ROOT="ismb26/prototypes"

GO_FILE="nrPDB-GO_annot.tsv"
GO_ASPECT="MF"

K_LIST="128,256,512,1024,2048,4096"
MIN_ASSIGNED="3"
MAX_ASSIGNED=""                    # blank => None
REMOVE_PCS="2"
SEED="0"

MAX_PER_PROTEIN_TRAIN="50"
MAX_PER_PROTEIN_EVAL="0"          # 0=all (can be heavy). Consider 50-200 if too slow.

PROTO_KNN_K="20"
PROTO_MIN_GO_TERMS="1"
PROTO_EXCLUDE_SHARED="false"      # true|false

ENRICH_TOP_TERMS="10"
ENRICH_MIN_PROTS_PER_PROTO="10"
ENRICH_MIN_TERM_PROTS="10"
ENRICH_QVAL="0.05"

STAB_RUNS="8"
STAB_FRAC="0.8"

DEVICE="cuda"
MODEL_GLOB="*"
SKIP_REGEX="^(\.git|__pycache__|val)$"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --segments_root) SEGMENTS_ROOT="$2"; shift 2 ;;
    --annotation_dir) ANNOTATION_DIR="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --go_file) GO_FILE="$2"; shift 2 ;;
    --go_aspect) GO_ASPECT="$2"; shift 2 ;;
    --k_list) K_LIST="$2"; shift 2 ;;
    --min_assigned) MIN_ASSIGNED="$2"; shift 2 ;;
    --max_assigned) MAX_ASSIGNED="$2"; shift 2 ;;
    --remove_pcs) REMOVE_PCS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max_per_protein_train) MAX_PER_PROTEIN_TRAIN="$2"; shift 2 ;;
    --max_per_protein_eval) MAX_PER_PROTEIN_EVAL="$2"; shift 2 ;;
    --proto_knn_k) PROTO_KNN_K="$2"; shift 2 ;;
    --proto_min_go_terms) PROTO_MIN_GO_TERMS="$2"; shift 2 ;;
    --proto_exclude_shared) PROTO_EXCLUDE_SHARED="true"; shift 1 ;;
    --enrich_top_terms) ENRICH_TOP_TERMS="$2"; shift 2 ;;
    --enrich_min_prots_per_proto) ENRICH_MIN_PROTS_PER_PROTO="$2"; shift 2 ;;
    --enrich_min_term_prots) ENRICH_MIN_TERM_PROTS="$2"; shift 2 ;;
    --enrich_qval) ENRICH_QVAL="$2"; shift 2 ;;
    --stability_runs) STAB_RUNS="$2"; shift 2 ;;
    --stability_frac) STAB_FRAC="$2"; shift 2 ;;
    --model_glob) MODEL_GLOB="$2"; shift 2 ;;
    --skip_regex) SKIP_REGEX="$2"; shift 2 ;;
    -h|--help) sed -n '1,200p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$SEGMENTS_ROOT" ]]; then
  echo "[ERROR] segments_root not found: $SEGMENTS_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ANNOTATION_DIR" ]]; then
  echo "[ERROR] annotation_dir not found: $ANNOTATION_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# Discover models (immediate children containing train/)
models=()
while IFS= read -r -d '' d; do
  name="$(basename "$d")"
  if [[ "$name" =~ $SKIP_REGEX ]]; then
    continue
  fi
  if [[ -d "$d/train" ]]; then
    models+=("$name")
  fi
done < <(find "$SEGMENTS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$MODEL_GLOB" -print0 | sort -z)

if [[ "${#models[@]}" -eq 0 ]]; then
  echo "[ERROR] No models found under $SEGMENTS_ROOT (glob=$MODEL_GLOB)" >&2
  exit 1
fi

echo "============================================================"
echo "Global prototypes (train-fit, val/test-assign) for ALL MODELS"
echo "  segments_root : $SEGMENTS_ROOT"
echo "  out_root      : $OUT_ROOT"
echo "  models        : ${#models[@]}"
echo "  K_list        : $K_LIST"
echo "  go_aspect     : $GO_ASPECT"
echo "============================================================"

fail=0
for m in "${models[@]}"; do
  echo ""
  echo "############################################################"
  echo "MODEL: $m"
  echo "############################################################"

  cmd=(
    python src/global_prototypes_fit.py
    --segments_root "$SEGMENTS_ROOT"
    --model_name "$m"
    --out_root "$OUT_ROOT"
    --annotation_dir "$ANNOTATION_DIR"
    --go_file "$GO_FILE"
    --go_aspect "$GO_ASPECT"
    --k_list "$K_LIST"
    --min_assigned "$MIN_ASSIGNED"
    --remove_pcs "$REMOVE_PCS"
    --seed "$SEED"
    --max_per_protein_train "$MAX_PER_PROTEIN_TRAIN"
    --max_per_protein_eval "$MAX_PER_PROTEIN_EVAL"
    --proto_knn_k "$PROTO_KNN_K"
    --proto_min_go_terms "$PROTO_MIN_GO_TERMS"
    --enrich_top_terms "$ENRICH_TOP_TERMS"
    --enrich_min_proteins_per_proto "$ENRICH_MIN_PROTS_PER_PROTO"
    --enrich_min_term_proteins "$ENRICH_MIN_TERM_PROTS"
    --enrich_qval "$ENRICH_QVAL"
    --stability_runs "$STAB_RUNS"
    --stability_frac_proteins "$STAB_FRAC"
    --device "$DEVICE"
  )

  if [[ -n "$MAX_ASSIGNED" ]]; then
    cmd+=( --max_assigned "$MAX_ASSIGNED" )
  fi
  if [[ "$PROTO_EXCLUDE_SHARED" == "true" ]]; then
    cmd+=( --proto_exclude_shared_proteins )
  fi

  log_path="${OUT_ROOT}/${m}/run_global_prototypes.log"
  echo "[INFO] Running:"
  printf '  %q' "${cmd[@]}"
  echo ""
  if ! "${cmd[@]}" 2>&1 | tee "$log_path" ; then
    echo "[ERROR] FAILED model=$m (see $log_path)" >&2
    fail=$((fail + 1))
  else
    echo "[OK] model=$m -> ${OUT_ROOT}/${m}"
  fi
done

echo ""
echo "============================================================"
echo "DONE. failures=$fail"
echo "============================================================"
if [[ "$fail" -gt 0 ]]; then
  exit 2
fi
