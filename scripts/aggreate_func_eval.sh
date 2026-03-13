#!/usr/bin/env bash
set -euo pipefail

ROOT="ismb26"
OUT_TABLE="puffin_best_table.tsv"
OUT_APR_AUPR="puffin_best_apr_aupr.tsv"

# Reset outputs
: > "$OUT_TABLE"
echo -e "file\tmetric\tvalue" > "$OUT_APR_AUPR"

# We will write the table header only once (with "file" prepended)
header_written=0

# Process in a stable order
find "$ROOT" -type f -name "evaluation_best_*.tsv" | sort | while read -r f; do
  base="$(basename "$f")"

  # Handle APR/AUPR separately (format: key newline value)
  if [[ "$base" == "evaluation_best_apr.tsv" || "$base" == "evaluation_best_aupr.tsv" ]]; then
    metric="$(awk 'NF{print; exit}' "$f")"
    value="$(awk 'NF{v=$0} END{print v}' "$f")"
    echo -e "${f}\t${metric}\t${value}" >> "$OUT_APR_AUPR"
    continue
  fi

  # For table-style TSVs:
  #  - first non-empty line is header
  #  - second non-empty line is the data row
  # We prepend file path as the first column.
  if [[ $header_written -eq 0 ]]; then
    hdr="$(awk 'NF{print; exit}' "$f")"
    echo -e "file\t${hdr}" >> "$OUT_TABLE"
    header_written=1
  fi

  row="$(awk 'NF{c++; if(c==2){print; exit}}' "$f")"
  if [[ -n "$row" ]]; then
    echo -e "${f}\t${row}" >> "$OUT_TABLE"
  else
    # If somehow file has only one non-empty line, still emit something
    echo -e "${f}\t" >> "$OUT_TABLE"
  fi
done

echo "✅ Wrote:"
echo "  - $OUT_TABLE"
echo "  - $OUT_APR_AUPR"
