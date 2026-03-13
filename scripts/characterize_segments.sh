#!/usr/bin/env bash
# scripts/characterize_segments.sh
# ------------------------------------------------------------------------------
# Auto-discover model directories under SEGMENTS_ROOT (e.g., segments/mincut_K128,
# segments/protygus_*, ...) and run characterization for each model + split.
#
# Assumes each model has:
#   <MODEL_DIR>/{train,valid,test}/
#   and inside each split:
#     <split>_residue_assignments.csv
#
# Runs with:
#   - structural metrics (requires --structure_dir)
#   - size-matched random baseline (--random_baseline)
#
# Outputs:
#   <OUT_ROOT>/<model_name>/<split>/characterization/...
#
# Usage:
#   bash scripts/run_segment_characterization_all_models.sh \
#     --segments_root segments \
#     --structure_dir /path/to/structures \
#     --out_root segment_reports \
#     --prefer_cif \
#     --contact_cutoff 10.0 \
#     --random_seeds 0,1,2,3,4
# ------------------------------------------------------------------------------

set -euo pipefail

# -----------------------
# Defaults
# -----------------------
SEGMENTS_ROOT="ismb26/segments"
STRUCTURE_DIR="data/pdb_chain"               
OUT_ROOT="ismb26/results/segment_reports"
STRUCTURE_FMT="auto"               # auto|pdb|cif
PREFER_CIF="true"                 # true|false
CONTACT_CUTOFF="10.0"
PAD_LABEL="-1"
RANDOM_SEEDS="0,1,2,3,4"
SPLITS=("train" "valid" "test")

# discovery controls
MODEL_GLOB="*"                     # restrict with e.g. "mincut_*" or "protygus_*"
SKIP_MODELS_REGEX="^(\.git|__pycache__|val)$"  # names to skip if accidentally present
MAX_MODELS=""                      # set to an int to cap discovery


# -----------------------
# Args
# -----------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --segments_root) SEGMENTS_ROOT="$2"; shift 2 ;;
    --structure_dir) STRUCTURE_DIR="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --structure_fmt) STRUCTURE_FMT="$2"; shift 2 ;;
    --prefer_cif) PREFER_CIF="true"; shift 1 ;;
    --contact_cutoff) CONTACT_CUTOFF="$2"; shift 2 ;;
    --pad_label) PAD_LABEL="$2"; shift 2 ;;
    --random_seeds) RANDOM_SEEDS="$2"; shift 2 ;;
    --splits)
      IFS=',' read -r -a SPLITS <<< "$2"
      shift 2
      ;;
    --model_glob) MODEL_GLOB="$2"; shift 2 ;;
    --skip_models_regex) SKIP_MODELS_REGEX="$2"; shift 2 ;;
    --max_models) MAX_MODELS="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,160p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# -----------------------
# Checks
# -----------------------
if [[ ! -d "$SEGMENTS_ROOT" ]]; then
  echo "[ERROR] segments_root not found: $SEGMENTS_ROOT" >&2
  exit 1
fi

if [[ -z "$STRUCTURE_DIR" ]]; then
  echo "[ERROR] --structure_dir is required for structural metrics." >&2
  exit 1
fi

if [[ ! -d "$STRUCTURE_DIR" ]]; then
  echo "[ERROR] structure_dir not found: $STRUCTURE_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# -----------------------
# Discover models
# -----------------------
# Models are immediate children of SEGMENTS_ROOT that contain at least one split dir.
models=()
while IFS= read -r -d '' d; do
  name="$(basename "$d")"
  if [[ "$name" =~ $SKIP_MODELS_REGEX ]]; then
    continue
  fi
  # require at least one expected split folder
  has_split="false"
  for s in "${SPLITS[@]}"; do
    if [[ -d "${d}/${s}" ]]; then
      has_split="true"
      break
    fi
  done
  if [[ "$has_split" == "true" ]]; then
    models+=("$d")
  fi
done < <(find "$SEGMENTS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$MODEL_GLOB" -print0 | sort -z)

if [[ "${#models[@]}" -eq 0 ]]; then
  echo "[ERROR] No model directories found under: $SEGMENTS_ROOT (glob=$MODEL_GLOB)" >&2
  exit 1
fi

if [[ -n "$MAX_MODELS" ]]; then
  models=("${models[@]:0:${MAX_MODELS}}")
fi

echo "============================================================"
echo "Segment characterization: ALL MODELS"
echo "  segments_root : $SEGMENTS_ROOT"
echo "  model_glob    : $MODEL_GLOB"
echo "  n_models      : ${#models[@]}"
echo "  structure_dir : $STRUCTURE_DIR"
echo "  out_root      : $OUT_ROOT"
echo "  structure_fmt : $STRUCTURE_FMT"
echo "  prefer_cif    : $PREFER_CIF"
echo "  contact_cutoff: $CONTACT_CUTOFF"
echo "  pad_label     : $PAD_LABEL"
echo "  random_seeds  : $RANDOM_SEEDS"
echo "  splits        : ${SPLITS[*]}"
echo "============================================================"
echo ""

fail_count=0
skip_count=0
ok_count=0

# -----------------------
# Run
# -----------------------
for model_dir in "${models[@]}"; do
  model_name="$(basename "$model_dir")"
  echo ""
  echo "############################################################"
  echo "MODEL: $model_name"
  echo "  dir: $model_dir"
  echo "############################################################"

  for split in "${SPLITS[@]}"; do
    run_dir="${model_dir}/${split}"
    prefix="${split}"
    out_dir="${OUT_ROOT}/${model_name}/${split}"

    if [[ ! -d "$run_dir" ]]; then
      echo "[WARN] Missing split dir: $run_dir (skipping split)"
      skip_count=$((skip_count + 1))
      continue
    fi

    ra="${run_dir}/${prefix}_residue_assignments.csv"
    if [[ ! -f "$ra" ]]; then
      echo "[WARN] Missing residue assignments: $ra (skipping split)"
      skip_count=$((skip_count + 1))
      continue
    fi

    mkdir -p "$out_dir"

    cmd=(
      python src/segment_characterize.py
      --cluster_dir "$run_dir"
      --output_dir "$out_dir"
      --prefix "$prefix"
      --pad_label "$PAD_LABEL"
      --structure_dir "$STRUCTURE_DIR"
      --structure_fmt "$STRUCTURE_FMT"
      --contact_cutoff "$CONTACT_CUTOFF"
      --random_baseline
      --random_seeds "$RANDOM_SEEDS"
    )
    if [[ "$PREFER_CIF" == "true" ]]; then
      cmd+=( --prefer_cif )
    fi

    echo ""
    echo "-----------------------------"
    echo "Split: ${split}"
    echo "  run_dir: ${run_dir}"
    echo "  out_dir: ${out_dir}"
    echo "-----------------------------"
    echo "[INFO] Running:"
    printf '  %q' "${cmd[@]}"
    echo ""

    log_path="${out_dir}/characterize_${model_name}_${split}.log"
    if ! "${cmd[@]}" 2>&1 | tee "$log_path" ; then
      echo "[ERROR] FAILED: model=${model_name} split=${split} (see $log_path)" >&2
      fail_count=$((fail_count + 1))
      continue
    fi

    must1="${out_dir}/${prefix}_summary.json"
    must2="${out_dir}/${prefix}_segments.csv"
    must3="${out_dir}/plots/${prefix}_hist_segment_size.pdf"
    if [[ -f "$must1" && -f "$must2" && -f "$must3" ]]; then
      echo "[OK] Wrote: $must1"
      ok_count=$((ok_count + 1))
    else
      echo "[WARN] Completed but expected outputs missing for model=${model_name} split=${split}." >&2
    fi
  done
done

echo ""
echo "============================================================"
echo "DONE"
echo "  ok    : $ok_count"
echo "  skipped: $skip_count"
echo "  failed: $fail_count"
echo "  out_root: $OUT_ROOT"
echo "============================================================"

if [[ "$fail_count" -gt 0 ]]; then
  exit 2
fi
