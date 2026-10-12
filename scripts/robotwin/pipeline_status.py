#!/usr/bin/env python3
"""Read a train/evaluate pipeline's status and measured training progress."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--store', type=Path, default=Path('/data/jack/robotwin2_turbovla'))
    args = parser.parse_args()
    root = args.store / 'pipelines' / args.run_id
    status = json.loads((root / 'status.json').read_text())
    target_steps = int(status.get('target_optimizer_steps', 0))
    status['checked_at'] = datetime.datetime.now().astimezone().isoformat()
    pid = int((root / 'pid').read_text())
    try:
        script_name = status.get('pipeline_script', 'train_then_evaluate.sh').encode()
        status['pipeline_process_alive'] = script_name in Path(f'/proc/{pid}/cmdline').read_bytes()
    except (FileNotFoundError, PermissionError):
        status['pipeline_process_alive'] = None
        status['process_check_note'] = 'Process exited or is not visible in this PID namespace; check on the training host.'
    log = root / 'train.log'
    if log.exists() and target_steps > 0:
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 1024 * 1024))
            text = handle.read().decode('utf-8', errors='replace')
        samples = {}
        for step, total, elapsed in re.findall(r'(\d+)/(\d+) \[([\d:]+)<', text):
            if int(total) != target_steps:
                continue
            seconds = 0
            for component in elapsed.split(':'):
                seconds = seconds * 60 + int(component)
            samples[int(step)] = seconds
        if samples:
            recent = sorted(samples.items())[-101:]
            step, elapsed = recent[-1]
            status['optimizer_steps'] = step
            status['training_progress_pct'] = round(100 * step / target_steps, 3)
            status['training_elapsed_seconds'] = elapsed
            status['training_log_updated_at'] = datetime.datetime.fromtimestamp(log.stat().st_mtime).astimezone().isoformat()
            if len(recent) > 1:
                first_step, first_elapsed = recent[0]
                seconds_per_step = (elapsed - first_elapsed) / (step - first_step)
                status['measured_window_updates'] = step - first_step
                status['seconds_per_optimizer_update'] = round(seconds_per_step, 3)
                if status['stage'] == 'training':
                    remaining = (target_steps - step) * seconds_per_step
                    status['estimated_training_hours_remaining'] = round(remaining / 3600, 2)
                    status['estimated_training_finish'] = datetime.datetime.fromtimestamp(log.stat().st_mtime + remaining).astimezone().isoformat(timespec='seconds')
    summary = Path(status['evaluation_dir']) / 'summary.json'
    if summary.exists():
        status['evaluation_summary'] = json.loads(summary.read_text())
    else:
        results = []
        mode = status.get('eval_mode', 'demo_clean')
        for path in sorted(Path(status['evaluation_dir']).glob(f'*_{mode}_*_eval.log')):
            with path.open('rb') as handle:
                handle.seek(max(0, path.stat().st_size - 65536))
                text = handle.read().decode('utf-8', errors='replace')
            text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
            counts = re.findall(r'Success rate:\s*(\d+)\s*/\s*(\d+)', text)
            if counts:
                successes, trials = map(int, counts[-1])
                results.append(dict(task=path.name.split(f'_{mode}_')[0],
                                    successes=successes, trials=trials,
                                    complete=('Data has been saved to' in text
                                              and trials == status['eval_trials_per_task'])))
        if results:
            status['partial_evaluation'] = dict(
                note='Progress only; this is not the final all-task success rate.',
                completed_tasks=sum(item['complete'] for item in results),
                finished_trials=sum(item['trials'] for item in results),
                planned_trials=status['eval_tasks'] * status['eval_trials_per_task'],
                tasks=results)
    comparison = Path(status['evaluation_dir']) / 'comparison_to_pilot5k.json'
    if comparison.exists():
        status['comparison_to_pilot5k'] = json.loads(comparison.read_text())
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
