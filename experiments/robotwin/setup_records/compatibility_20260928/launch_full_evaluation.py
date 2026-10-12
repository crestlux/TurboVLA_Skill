"""Launch the validated 150k EMA, Clean then Randomized, 50 tasks x 10 trials each."""
import datetime
import json
import re
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert REPO == Path('/home/jack/TurboVLA')
assert json.loads((HERE / 'probe_results.json').read_text())['status'] == 'passed'
assert json.loads((HERE / 'randomized_preflight.json').read_text())['status'] == 'passed'
record_path = HERE / 'benchmark_queue.json'
if record_path.exists():
    raise FileExistsError('The full evaluation queue has already been prepared; inspect its state before relaunching.')

now = datetime.datetime.now().astimezone()
stamp = now.strftime('%Y%m%d_%H%M%S')
session = f'turbovla_release150k_{stamp}'
plans = [dict(mode=mode, run_id=f'turbovla_release150000_postnorm_{mode}_eval10_{stamp}',
              tasks=50, trials_per_task=10, instruction_sample_num=100)
         for mode in ['demo_clean', 'demo_randomized']]
script = HERE / f'benchmark_queue_{stamp}.sh'
record = dict(stage='queued', created_at=now.isoformat(), tmux_session=session,
              queue_script=str(script), model_revision='9cdb576ed9c56039b9c1149142ed654e876bad76',
              weights='EMA', optimizer_steps=150000, planned_trials=1000, plans=plans,
              note='Reduced evaluation: 10 rather than the official 100 episodes per task and variant. '
                   'The official instruction sampling count of 100 is retained.')
record_path.write_text(json.dumps(record, indent=2) + '\n')
lines = ['#!/usr/bin/env bash', 'set -Eeuo pipefail', f'cd {shlex.quote(str(REPO))}',
         'export CUDA_VISIBLE_DEVICES=0,1 ROBOTWIN_TEST_NUM=10 ROBOTWIN_INSTRUCTION_SAMPLE_NUM=100',
         'export ROBOTWIN_BASE_PORT=7600', f'export EVALUATION_TMUX_SESSION={shlex.quote(session)}',
         f'exec > >(tee -a {shlex.quote(str(HERE / "benchmark_queue.log"))}) 2>&1']
lines += ['on_queue_exit() {', '    local rc=$?', '    trap - EXIT',
          '    python3 - "$rc" <<\'PY\'', 'import datetime,json,sys', 'from pathlib import Path',
          f'p=Path({str(record_path)!r}); record=json.loads(p.read_text())',
          'rc=int(sys.argv[1])', 'if rc:',
          "    record.update(stage='failed',exit_code=rc,updated_at=datetime.datetime.now().astimezone().isoformat())",
          "    p.write_text(json.dumps(record,indent=2)+'\\n')", 'PY',
          '    exit "$rc"', '}', 'trap on_queue_exit EXIT']
for plan in plans:
    lines += [f'export RUN_ID={shlex.quote(plan["run_id"])}',
              f'export ROBOTWIN_EVAL_MODE={shlex.quote(plan["mode"])}',
              'bash scripts/robotwin/evaluate_latest_released_model.sh']
lines += ["python3 - <<'PY'", 'import csv,datetime,json', 'from pathlib import Path',
          f'record_path=Path({str(record_path)!r})',
          "record=json.loads(record_path.read_text()); results=[]",
          "for plan in record['plans']:",
          "    root=Path('/data/jack/robotwin2_turbovla/evaluation')/(plan['run_id']+'_seed0')",
          "    summary=json.loads((root/'summary.json').read_text())",
          "    rows=list(csv.DictReader((root/'task_results.csv').open()))",
          "    assert len(rows)==50 and summary['trials']==500 and summary['mode']==plan['mode']",
          "    successes=sum(int(row['successes']) for row in rows)",
          "    assert abs(summary['mean_success_pct']-successes/5)<1e-9",
          "    results.append(dict(plan,successes=successes,trials=500,success_pct=successes/5,results_dir=str(root)))",
          "combined=dict(status='completed',completed_at=datetime.datetime.now().astimezone().isoformat(),",
          "              results=results,mean_success_pct=sum(r['success_pct'] for r in results)/2,",
          "              note=record['note'])",
          "(record_path.parent/'latest150k_eval10_summary.json').write_text(json.dumps(combined,indent=2)+'\\n')",
          "record.update(stage='completed',completed_at=combined['completed_at'])",
          "record_path.write_text(json.dumps(record,indent=2)+'\\n')",
          "print(json.dumps(combined,indent=2))", 'PY']
script.write_text('\n'.join(lines) + '\n')
for index, code in enumerate(re.findall(r"python3[^\n]*<<'PY'\n(.*?)\nPY", script.read_text(), re.S)):
    compile(code, f'{script}:embedded_python_{index}', 'exec')
subprocess.run(['bash', '-n', str(script)], check=True)
record['stage'] = 'running'
record_path.write_text(json.dumps(record, indent=2) + '\n')
try:
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash', str(script)], check=True)
except Exception:
    record['stage'] = 'launch_failed'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    raise
print(json.dumps(record, indent=2))
