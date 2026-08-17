#!/usr/bin/env bash
set -euo pipefail

# Independent rerun after the stale quality_resource lineage was stopped.
# Every stage has a new output root; the Python runner refuses non-empty roots.
ROOT="/mnt/sda/rhyang/slm"
RUNNER="$ROOT/neural_instance_culling/benchmark/run_pvs_hierarchical_survival_threshold_aligned_v2.py"
BASE="$ROOT/neural_instance_culling/model/out/pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2"
SCAN="${BASE}_scan12_20260814_rerun"
REFINE="${BASE}_refine24_20260814_rerun"
FORMAL="${BASE}_formal80_20260814_rerun"

printf '[%s] v2 rerun pipeline started; testRead=false; loss lineage=threshold_aligned_utility\n' "$(date -Is)"

conda run -n slm_pvs python "$RUNNER" scan12 \
  --output-root "$SCAN" --log-root "$SCAN/logs" --gpu-ids 0 1 2 3
python3 - "$SCAN/scan_member_summary.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))["members"]
assert len(rows) == 16, len(rows)
assert all(int(row.get("returnCode", 1)) == 0 for row in rows)
assert all(row["job"]["variantSpec"]["lossVariant"] == "threshold_aligned_utility" for row in rows)
print("scan contract: 16/16 successful; threshold_aligned_utility")
PY

conda run -n slm_pvs python "$RUNNER" refine24 \
  --scan-root "$SCAN" --output-root "$REFINE" --log-root "$REFINE/logs" --gpu-ids 0 1 2 3
python3 - "$REFINE/refine_member_summary.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))["members"]
assert len(rows) == 12, len(rows)
assert all(int(row.get("returnCode", 1)) == 0 for row in rows)
assert all(row["job"]["variantSpec"]["lossVariant"] == "threshold_aligned_utility" for row in rows)
assert path.with_name("frozen_config.json").is_file()
print("refine contract: 12/12 successful; frozen config present")
PY

conda run -n slm_pvs python "$RUNNER" formal80 \
  --refine-root "$REFINE" --output-root "$FORMAL" --log-root "$FORMAL/logs" \
  --variants full without_hierarchical_relation shuffled_relation_source without_viewcell_integration without_threshold_aligned_utility \
  --seeds 20260801 20260802 20260803 --gpu-ids 0 1 2 3
python3 - "$FORMAL/formal_member_summary.json" <<'PY'
import json, sys
expected = {
    "full": "threshold_aligned_utility",
    "without_hierarchical_relation": "threshold_aligned_utility",
    "shuffled_relation_source": "threshold_aligned_utility",
    "without_viewcell_integration": "threshold_aligned_utility",
    "without_threshold_aligned_utility": "rvl",
}
rows = json.load(open(sys.argv[1], encoding="utf-8"))["members"]
assert len(rows) == 15, len(rows)
assert all(int(row.get("returnCode", 1)) == 0 for row in rows)
for row in rows:
    job = row["job"]
    assert job["variantSpec"]["lossVariant"] == expected[job["variant"]], job
print("formal contract: 15/15 successful; 12 threshold_aligned_utility + 3 rvl")
PY

printf '[%s] v2 rerun pipeline completed\n' "$(date -Is)"
