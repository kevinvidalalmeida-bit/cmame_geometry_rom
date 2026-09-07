#!/usr/bin/env bash
# Run the production Sobol--Ritz factorized compiler on G00--G09.
# Usage: bash scripts/cmame_interpretable_pipeline/run_sobol_ritz_factorized_all.sh RUN_NAME
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_name="${1:-sobol_ritz_factorized_all_20260906}"
out_root="$project_root/results/cmame_method/interpretable_vf05_25_ar5_20"

source "$project_root/scripts/cmame_interpretable_pipeline/activate_env.sh"

python "$project_root/scripts/cmame_interpretable_pipeline/03_run_pipeline.py" \
  --config "$project_root/scripts/cmame_interpretable_pipeline/campaign_config.json" \
  --out-root "$out_root" \
  --stage sobol_pod \
  --run-name "$run_name" \
  --geometry-ids 0 1 2 3 4 5 6 7 8 9
