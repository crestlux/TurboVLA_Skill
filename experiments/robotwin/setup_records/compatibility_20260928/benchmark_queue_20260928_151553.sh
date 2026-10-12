#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/jack/TurboVLA
export CUDA_VISIBLE_DEVICES=0,1 ROBOTWIN_TEST_NUM=10 ROBOTWIN_INSTRUCTION_SAMPLE_NUM=100
export ROBOTWIN_BASE_PORT=7600
export EVALUATION_TMUX_SESSION=turbovla_release150k_20260928_151553
exec > >(tee -a /home/jack/TurboVLA/experiments/robotwin/setup_records/compatibility_20260928/benchmark_queue.log) 2>&1
on_queue_exit() {
    local rc=$?
    trap - EXIT
    python3 - "$rc" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path('/home/jack/TurboVLA/experiments/robotwin/setup_records/compatibility_20260928/benchmark_queue.json'); record=json.loads(p.read_text())
rc=int(sys.argv[1])
if rc:
    record.update(stage='failed',exit_code=rc,updated_at=datetime.datetime.now().astimezone().isoformat())
    p.write_text(json.dumps(record,indent=2)+'\n')
PY
    exit "$rc"
}
trap on_queue_exit EXIT
export RUN_ID=turbovla_release150000_postnorm_demo_clean_eval10_20260928_151553
export ROBOTWIN_EVAL_MODE=demo_clean
bash scripts/robotwin/evaluate_latest_released_model.sh
export RUN_ID=turbovla_release150000_postnorm_demo_randomized_eval10_20260928_151553
export ROBOTWIN_EVAL_MODE=demo_randomized
bash scripts/robotwin/evaluate_latest_released_model.sh
python3 - <<'PY'
import csv,datetime,json
from pathlib import Path
record_path=Path('/home/jack/TurboVLA/experiments/robotwin/setup_records/compatibility_20260928/benchmark_queue.json')
record=json.loads(record_path.read_text()); results=[]
for plan in record['plans']:
    root=Path('/data/jack/robotwin2_turbovla/evaluation')/(plan['run_id']+'_seed0')
    summary=json.loads((root/'summary.json').read_text())
    rows=list(csv.DictReader((root/'task_results.csv').open()))
    assert len(rows)==50 and summary['trials']==500 and summary['mode']==plan['mode']
    successes=sum(int(row['successes']) for row in rows)
    assert abs(summary['mean_success_pct']-successes/5)<1e-9
    results.append(dict(plan,successes=successes,trials=500,success_pct=successes/5,results_dir=str(root)))
combined=dict(status='completed',completed_at=datetime.datetime.now().astimezone().isoformat(),
              results=results,mean_success_pct=sum(r['success_pct'] for r in results)/2,
              note=record['note'])
(record_path.parent/'latest150k_eval10_summary.json').write_text(json.dumps(combined,indent=2)+'\n')
record.update(stage='completed',completed_at=combined['completed_at'])
record_path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(combined,indent=2))
PY
