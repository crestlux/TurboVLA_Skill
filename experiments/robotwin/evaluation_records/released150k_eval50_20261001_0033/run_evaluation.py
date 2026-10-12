"""Durable Clean -> Randomized evaluation queue; stop on any failed stage."""
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
STORE = Path('/data/jack/robotwin2_turbovla')


def now():
    return dt.datetime.now().astimezone().isoformat()


def write(name, data):
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    # Exclusive creation prevents accidental duplicate starts of this queue.
    with (ROOT / 'started.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(), started_at=now()), handle)
    plan = json.loads((ROOT / 'plan.json').read_text())
    plan.update(status='running', started_at=now(), pid=os.getpid())
    write('plan.json', plan)
    results = dict(status='running', model=plan['model'], protocol=plan['protocol'],
                   started_at=plan['started_at'], variants=[])
    write('evaluation_results.json', results)
    try:
        for stage in plan['stages']:
            stage.update(status='running', started_at=now())
            plan['current_stage'] = stage['mode']
            write('plan.json', plan)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', ROBOTWIN_JOBS_PER_GPU='3',
                       ROBOTWIN_TEST_NUM='50', ROBOTWIN_INSTRUCTION_SAMPLE_NUM='100',
                       ROBOTWIN_PYTHON_RANDOM_SEED='20260930', PYTHONHASHSEED='0',
                       ROBOTWIN_BASE_PORT='8400', ROBOTWIN_EVAL_MODE=stage['mode'],
                       RUN_ID=stage['run_id'], EVALUATION_TMUX_SESSION=plan['tmux_session'])
            command = [sys.executable, str(REPO/'scripts/robotwin/profile_evaluation.py'),
                       '--output', str(ROOT/stage['mode']), '--gpus', '0,1',
                       '--interval', '10', '--timeout-seconds', '259200', '--',
                       'bash', 'scripts/robotwin/evaluate_latest_released_model.sh', 'all']
            print(f"{now()} Starting {stage['mode']}", flush=True)
            rc = subprocess.call(command, cwd=REPO, env=env)
            stage.update(exit_code=rc, finished_at=now())
            if rc:
                stage['status'] = 'failed'
                raise RuntimeError(f"{stage['mode']} exited with {rc}; subsequent stages not started")
            log_dir = Path(stage['evaluation_dir'])
            rows = list(csv.DictReader((log_dir/'task_results.csv').open()))
            if len(rows) != 50 or len({r['task'] for r in rows}) != 50:
                raise RuntimeError('Incomplete or duplicate task results')
            if any(int(r['trials']) != 50 for r in rows):
                raise RuntimeError('Incomplete task trial count')
            successes = sum(int(r['successes']) for r in rows)
            trials = sum(int(r['trials']) for r in rows)
            stage['status'] = 'completed'
            results['variants'].append(dict(mode=stage['mode'], successes=successes,
                trials=trials, success_rate_pct=100*successes/trials, tasks=rows,
                evaluation_dir=str(log_dir), resources=str(ROOT/stage['mode']/'summary.json')))
            write('evaluation_results.json', results)
            write('plan.json', plan)
        successes = sum(v['successes'] for v in results['variants'])
        trials = sum(v['trials'] for v in results['variants'])
        assert trials == 5000
        results.update(status='completed', finished_at=now(),
                       overall=dict(successes=successes, trials=trials,
                                    success_rate_pct=100*successes/trials))
        plan.update(status='completed', current_stage=None, finished_at=now())
    except BaseException as error:
        plan.update(status='failed', error=repr(error), finished_at=now())
        results.update(status='failed', error=repr(error), finished_at=now())
        raise
    finally:
        write('evaluation_results.json', results)
        write('plan.json', plan)


if __name__ == '__main__':
    main()
