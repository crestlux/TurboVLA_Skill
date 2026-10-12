"""Audit the completed 150k release evaluation without modifying its outputs."""
import ast
import concurrent.futures
import csv
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

from audit_probes import audit, read_trials

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STORE = Path('/data/jack/robotwin2_turbovla')
SIMULATOR = Path('/home/jack/practice/RoboTwin-TurboVLA-bf44be5')
OUTPUT = HERE.parents[1] / 'evaluation_records/released150k_20260929'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def video_metadata(path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=codec_name,width,height,nb_frames,duration', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True, timeout=30)
    stream = json.loads(result.stdout)['streams'][0]
    assert int(stream['nb_frames']) > 0 and float(stream['duration']) > 0, str(path)
    return dict(path=str(path), bytes=path.stat().st_size, **stream)


def main():
    queue = json.loads((HERE / 'benchmark_queue.json').read_text())
    combined = json.loads((HERE / 'latest150k_eval10_summary.json').read_text())
    validation = json.loads((HERE / 'fixed150k_transformers4576.json').read_text())
    assert queue['stage'] == combined['status'] == 'completed'
    checkpoint = Path(validation['checkpoint'])
    for path, key in [(checkpoint, 'checkpoint_sha256'),
                      (checkpoint.parent / 'config.yaml', 'config_sha256'),
                      (checkpoint.parent / 'dataset_statistics.json', 'statistics_sha256')]:
        assert sha(path) == validation[key], str(path)
    tree = ast.parse((REPO / 'experiments/robotwin/data_registry/data_config.py').read_text())
    tasks = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == '_CLEAN50_TASKS'
                         for target in node.targets))
    assert len(tasks) == len(set(tasks)) == 50
    results, videos = [], []
    for plan in queue['plans']:
        pipeline = STORE / 'pipelines' / plan['run_id']
        status = json.loads((pipeline / 'status.json').read_text())
        assert status['weights'] == 'EMA' and status['checkpoint_optimizer_steps'] == 150000
        assert status['checkpoint'] == str(checkpoint)
        assert status['feature_source'] == 'last_hidden_state'
        assert status['compatibility_profile'] == 'robotwin'
        assert status['eval_seed'] == 0 and status['instruction_sample_num'] == 100
        assert status['simulator_commit'] == 'bf44be51cf5717a5595ce59447f2cf5263d2aa95'
        root = Path(status['evaluation_dir'])
        with (root / 'task_results.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(set(row['task'] for row in rows)) == 50
        assert set(row['task'] for row in rows) == set(tasks)
        checked = audit(dict(plan, tasks=tasks, checkpoint=str(checkpoint)))
        provenance = pipeline / 'provenance'
        manifest = json.loads((provenance / 'SHA256.json').read_text())
        for name, expected in manifest.items():
            assert sha(provenance / name) == expected, name
            if name.startswith('source/'):
                assert sha(REPO / name[len('source/'):]) == expected, name
        for name, key in [('config.yaml', 'config_sha256'),
                          ('dataset_statistics.json', 'statistics_sha256')]:
            assert sha(provenance / name) == validation[key]
        packages = (provenance / 'policy-packages.txt').read_text().splitlines()
        assert 'transformers==4.57.6' in packages and 'torch==2.6.0+cu124' in packages
        failures_at_limit = 0
        tracebacks = []
        for task in tasks:
            paths = list(root.glob(f'{task}_{plan["mode"]}_*_eval.log'))
            assert len(paths) == 1
            text, trials = read_trials(paths[0])
            starts = list(re.finditer(r'^EVAL_EPISODE ', text, re.M))
            ends = list(re.finditer(r'Success rate:\s*\d+/\d+', text))
            assert len(starts) == len(ends) == len(trials) == 10
            for start, end, trial in zip(starts, ends, trials):
                section = text[start.start():end.start()]
                assert 'Traceback (most recent call last):' not in section
                outcome = re.findall(r'^(Success!|Fail!)$', section, re.M)
                assert outcome == ['Success!' if trial['success'] else 'Fail!']
                if not trial['success']:
                    steps = re.findall(r'step:\s*(\d+) / (\d+)', section)
                    assert steps and steps[-1][0] == steps[-1][1], task
                    failures_at_limit += 1
            count = text.count('Traceback (most recent call last):')
            assert text.count('error occurs !') == count
            if count:
                tracebacks.append(dict(task=task, count=count,
                                       phase='caught expert/scene validation before graded trial'))
            result_file = SIMULATOR / re.search(r'Data has been saved to (.+)', text)[1]
            task_videos = list(result_file.parent.glob('episode*.mp4'))
            assert {p.name for p in task_videos} == {f'episode{i}.mp4' for i in range(10)}
            videos.extend(task_videos)
        assert failures_at_limit == checked['trials'] - checked['successes']
        expected = next(item for item in combined['results'] if item['run_id'] == plan['run_id'])
        assert checked['successes'] == expected['successes'] and checked['trials'] == 500
        elapsed = (datetime.datetime.fromisoformat(checked['completed_at'])
                   - datetime.datetime.fromisoformat(checked['started_at'])).total_seconds()
        checked.update(mode=plan['mode'], elapsed_seconds=elapsed,
                       failures_at_step_limit=failures_at_limit,
                       caught_expert_tracebacks=tracebacks,
                       archived_files_verified=len(manifest),
                       zero_success_tasks=sum(not any(t['success'] for t in v) for v in checked['tasks'].values()),
                       perfect_tasks=sum(all(t['success'] for t in v) for v in checked['tasks'].values()))
        results.append(checked)
        print(f'{plan["mode"]}: {checked["successes"]}/500, all logs, CSV, simulator scores, '
              f'checkpoint records and {len(manifest)} archived files verified.', flush=True)
    assert len(videos) == len(set(videos)) == 1000
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        streams = list(executor.map(video_metadata, videos))
    total = sum(item['successes'] for item in results)
    assert abs(combined['mean_success_pct'] - total / 10) < 1e-9
    report = dict(status='passed', checked_at=datetime.datetime.now().astimezone().isoformat(),
                  successes=total, trials=1000, mean_success_pct=total / 10,
                  model_revision=queue['model_revision'], checkpoint=validation['checkpoint'],
                  checkpoint_sha256=validation['checkpoint_sha256'], results=results,
                  videos_checked=len(streams), video_bytes=sum(s['bytes'] for s in streams),
                  limitation=queue['note'],
                  interpretation='All 50 tasks in both variants completed. This is a 10-episode-per-task '
                                 'evaluation, not the official 100-episode-per-task protocol.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'completion_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUTPUT / 'video_stream_audit.json').write_text(json.dumps(streams, indent=2) + '\n')
    comparisons = []
    for task in tasks:
        row = dict(task=task)
        for variant, result in zip(['clean', 'randomized'], results):
            row[variant + '_successes'] = sum(t['success'] for t in result['tasks'][task])
        row['total_successes'] = row['clean_successes'] + row['randomized_successes']
        row['total_trials'] = 20
        comparisons.append(row)
    with (OUTPUT / 'task_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps({key: report[key] for key in ['status', 'successes', 'trials',
                                                  'mean_success_pct', 'videos_checked']}, indent=2))
    print(OUTPUT)


if __name__ == '__main__':
    main()
