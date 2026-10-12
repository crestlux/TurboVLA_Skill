"""Audit complete compatibility probes and compare their recorded episode inputs."""
import csv
import datetime
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORE = Path('/data/jack/robotwin2_turbovla')
SIMULATOR = Path('/home/jack/practice/RoboTwin-TurboVLA-bf44be5')
BASELINE = 'turbovla_released55k_eval10_20260927_2237'
QUEUE = json.loads((HERE / 'probe_queue.json').read_text())
TASKS = QUEUE['plans'][0]['tasks']


def read_trials(log):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', log.read_text(errors='replace'))
    rates = re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>[^\n]+current seed:\s*(\d+)', text)
    inputs = [json.loads(line.split('EVAL_EPISODE ', 1)[1])
              for line in text.splitlines() if line.startswith('EVAL_EPISODE ')]
    trials = []
    previous = 0
    for cumulative, count, seed in rates:
        cumulative, count, seed = int(cumulative), int(count), int(seed)
        assert count == len(trials) + 1
        assert cumulative - previous in (0, 1)
        item = dict(seed=seed, success=bool(cumulative - previous), instruction=None)
        if inputs:
            entry = inputs[count - 1]
            assert entry['seed'] == seed
            item['instruction'] = entry['instruction']
        trials.append(item)
        previous = cumulative
    return text, trials


def baseline():
    root = STORE / 'evaluation' / f'{BASELINE}_seed0'
    rows = {}
    for task in TASKS:
        _, trials = read_trials(next(root.glob(f'{task}_demo_clean_*_eval.log')))
        rows[task] = trials[:5]
    return dict(run_id=BASELINE, tasks=rows,
                successes=sum(sum(t['success'] for t in row) for row in rows.values()),
                trials=20, note='First five trials of the same four tasks; historical instructions were not logged.')


def audit(plan):
    run = plan['run_id']
    pipeline = STORE / 'pipelines' / run
    status = json.loads((pipeline / 'status.json').read_text())
    assert status['stage'] == 'completed', f'{run}: not yet complete'
    assert int((pipeline / 'exit_code.txt').read_text()) == 0
    root = Path(status['evaluation_dir'])
    mode = plan.get('mode', 'demo_clean')
    assert status.get('eval_mode', 'demo_clean') == mode
    summary = json.loads((root / 'summary.json').read_text())
    csv_rows = {row['task']: row for row in csv.DictReader((root / 'task_results.csv').open())}
    rows = {}
    video_count = 0
    for task in plan['tasks']:
        log = next(root.glob(f'{task}_{mode}_*_eval.log'))
        text, trials = read_trials(log)
        assert len(trials) == plan['trials_per_task']
        assert all(t['instruction'] for t in trials)
        assert sum(t['success'] for t in trials) == int(csv_rows[task]['successes'])
        result = SIMULATOR / re.search(r'Data has been saved to (.+)', text)[1]
        assert float(result.read_text().strip().splitlines()[-1]) == sum(t['success'] for t in trials) / len(trials)
        videos = list(result.parent.glob('episode*.mp4'))
        assert len(videos) == len(trials) and all(p.stat().st_size > 0 for p in videos)
        video_count += len(videos)
        server = log.with_name(log.name.replace('_eval.log', '_server.log')).read_text()
        assert plan['checkpoint'] in server
        assert 'feature_source=last_hidden_state' in server
        assert f"compatibility_profile={status['compatibility_profile']}" in server
        rows[task] = trials
    successes = sum(sum(t['success'] for t in row) for row in rows.values())
    total = sum(map(len, rows.values()))
    assert total == summary['trials']
    assert abs(summary['mean_success_pct'] - 100 * successes / total) < 1e-9
    return dict(run_id=run, tasks=rows, successes=successes, trials=total,
                success_pct=100 * successes / total, audited_nonempty_videos=video_count,
                started_at=status['started_at'], completed_at=status['updated_at'])


def compare(a, b):
    rows = {}
    for task in TASKS:
        ta, tb = a['tasks'][task], b['tasks'][task]
        by_seed_a = {t['seed']: t for t in ta}
        by_seed_b = {t['seed']: t for t in tb}
        common = sorted(by_seed_a.keys() & by_seed_b.keys())
        same_inputs = [s for s in common if by_seed_a[s]['instruction'] is not None
                       and by_seed_a[s]['instruction'] == by_seed_b[s]['instruction']]
        rows[task] = dict(seed_sequence_equal=[t['seed'] for t in ta] == [t['seed'] for t in tb],
                          common_seeds=common, same_seed_and_instruction=same_inputs,
                          same_input_trials=len(same_inputs),
                          same_input_successes_a=sum(by_seed_a[s]['success'] for s in same_inputs),
                          same_input_successes_b=sum(by_seed_b[s]['success'] for s in same_inputs))
    return dict(a=a['run_id'], b=b['run_id'], tasks=rows)


if __name__ == '__main__':
    old = baseline()
    fixed, latest = [audit(plan) for plan in QUEUE['plans']]
    report = dict(checked_at=datetime.datetime.now().astimezone().isoformat(),
                  status='passed', baseline=old, fixed55k=fixed, latest150k=latest,
                  comparisons=[compare(old, fixed), compare(fixed, latest)],
                  limitation='Four selected Clean tasks, five trials each; not the 50-task benchmark. '
                             'Planner seed acceptance may differ; exact paired comparisons require both seed and instruction equality.')
    output = HERE / 'probe_results.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {p: v[p] for p in ('run_id', 'successes', 'trials')}
                      for k, v in report.items() if k in ('baseline', 'fixed55k', 'latest150k')}, indent=2))
    print(output)
