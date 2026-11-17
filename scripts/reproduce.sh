#!/usr/bin/env bash
# Rebuild every number in the README from scratch:
#   1. the release run and three ablations (same config, features switched off),
#   2. the int8 model file, the loss curve, the held-out evaluation,
#   3. the browser parity fixture.
# Usage: bash scripts/reproduce.sh [python]   (python defaults to "python")
set -euo pipefail
PY="${1:-python}"
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

"$PY" -m lyra train --config configs/kieu.json --out runs/kieu-v1
for name in no-slots no-phonology plain; do
  "$PY" -m lyra train --config "configs/ablations/$name.json" --out "runs/ablation-$name"
done

"$PY" -m lyra export --checkpoint runs/kieu-v1/best.npz --tokenizer runs/kieu-v1/tokenizer.json \
  --out models/lyra-kieu-q8.bin
"$PY" -m lyra plot --log runs/kieu-v1/log.csv --out docs/assets/loss-curve.svg \
  --title "Training Lyra on Truyện Kiều" \
  --subtitle "Cross-entropy per token, 1.14M parameters, 6 CPU threads (Intel i7-14700K)"
"$PY" -m lyra evaluate > runs/kieu-v1/evaluation.json
"$PY" scripts/summarize_runs.py runs/kieu-v1 runs/ablation-no-slots runs/ablation-no-phonology \
  runs/ablation-plain > runs/summary.md
"$PY" scripts/make_parity_fixture.py
