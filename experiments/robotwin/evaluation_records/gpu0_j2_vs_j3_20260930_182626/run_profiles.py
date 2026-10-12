"""Run the fixed 2-versus-3 worker experiment sequentially on GPU 0."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = root.parents[3]
path = root / 'plan.json'
plan = json.loads(path.read_text())
if plan['status'] != 'prepared':
    raise SystemExit('This experiment has already been started; do not overwrite it.')


def save():
    plan['updated_at'] = datetime.datetime.now().astimezone().isoformat()
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(plan, indent=2)+'\n')
    temporary.replace(path)


plan['status'] = 'running'
save()
for stage in plan['stages']:
    plan['current_stage'] = stage['name']
    save()
    print('Starting '+stage['name']+' at '+plan['updated_at'], flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['gpu'],
               ROBOTWIN_JOBS_PER_GPU=str(stage['jobs']),
               ROBOTWIN_TEST_NUM=str(plan['trials_per_task']),
               ROBOTWIN_INSTRUCTION_SAMPLE_NUM=str(plan['instruction_pool']),
               ROBOTWIN_PYTHON_RANDOM_SEED=str(plan['python_random_seed']),
               PYTHONHASHSEED='0', ROBOTWIN_BASE_PORT='8200',
               ROBOTWIN_EVAL_MODE=stage['mode'], RUN_ID=stage['run_id'])
    command = [sys.executable, str(repo/'scripts/robotwin/profile_evaluation.py'),
               '--output', str(root/stage['name']), '--gpus', plan['gpu'],
               '--interval', '1', '--timeout-seconds', '2700', '--',
               'bash', 'scripts/robotwin/evaluate_latest_released_model.sh', *plan['tasks']]
    with (root/'queue.log').open('a') as output:
        rc = subprocess.call(command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
    if rc:
        plan.update(status='failed', failed_stage=stage['name'], exit_code=rc)
        save()
        raise SystemExit(rc)
    resource = json.loads((root/stage['name']/'summary.json').read_text())
    plan['completed_stages'].append(stage['name'])
    save()
    print(json.dumps(dict(stage=stage['name'], seconds=resource['wall_seconds'],
                          peak_vram_mib=resource['gpu'][0]['peak_used_mib'],
                          cpu_mean_cores=resource['process_cpu_mean_pct']/100)), flush=True)
plan.update(status='collecting', current_stage=None)
save()
rc = subprocess.call([sys.executable, str(root/'collect_results.py')], cwd=repo)
if rc:
    plan = json.loads(path.read_text())
    plan.update(status='collection_failed', exit_code=rc)
    save()
raise SystemExit(rc)
