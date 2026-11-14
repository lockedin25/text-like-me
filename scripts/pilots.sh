#!/usr/bin/env bash
# Run on an already provisioned GPU, after signed data has been uploaded and verified.
# This script never creates a VM or stores a GPU-provider API key.
set -euo pipefail
: "${WORK_DIR:?Set WORK_DIR to the private work folder on this GPU}"
: "${MODEL_REVISION:?Set MODEL_REVISION to the revision in datasets/build.json}"
run_pilot() {
  local name="$1" variant="$2" lr="$3" accum="$4"
  python -m recipe.train --data "$WORK_DIR/datasets/$variant" --datasets "$WORK_DIR/datasets" \
    --signoff "$WORK_DIR/signoff.json" --revision "$MODEL_REVISION" --out "$WORK_DIR/runs/$name" \
    --lr "$lr" --accum "$accum" --epochs 1 --schedule-epochs 2
}
run_pilot p1 everything 5e-5 16
run_pilot p2 everything 1e-4 16
# Compare p1/p2 before choosing the LR. The original selected value was 1e-4;
# override after examining your own pilots, not the author's outcome.
PILOT_LR="${PILOT_LR:-1e-4}"
run_pilot p3 everything "$PILOT_LR" 8
run_pilot p4 english "$PILOT_LR" 16
