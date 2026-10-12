"""Compare completed evaluations, including the first common number of trials."""
import argparse
import csv
import json
import re
from pathlib import Path


def read_tasks(root):
    rows = list(csv.DictReader((root / 'task_results.csv').open()))
    tasks = {}
    for row in rows:
        task = row['task']
        if task in tasks:
            raise ValueError(f'Duplicate task: {task}')
        paths = list(root.glob(f'{task}_demo_clean_*_eval.log'))
        if len(paths) != 1:
            raise ValueError(f'{task}: expected one evaluation log')
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', paths[0].read_text(errors='replace'))
        counts = [tuple(map(int, item)) for item in re.findall(
            r'Success rate:\s*(\d+)\s*/\s*(\d+).*?current seed:\s*(\d+)', text)]
        trials, successes = int(row['trials']), int(row['successes'])
        if ('Data has been saved to' not in text or len(counts) != trials
                or [n for s, n, seed in counts] != list(range(1, trials + 1))
                or counts[-1][0] != successes):
            raise ValueError(f'{task}: incomplete or inconsistent evaluation')
        if not 0 <= successes <= trials or float(row['success_rate_pct']) != 100 * successes / trials:
            raise ValueError(f'{task}: invalid success rate')
        tasks[task] = counts
    if not tasks:
        raise ValueError('No task results')
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_dir', type=Path)
    parser.add_argument('candidate_dir', type=Path)
    parser.add_argument('--common-trials', type=int, default=10)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    baseline, candidate = read_tasks(args.baseline_dir), read_tasks(args.candidate_dir)
    if set(baseline) != set(candidate) or args.common_trials < 1:
        raise ValueError('Task lists differ or common-trials is invalid')
    rows = []
    for task in baseline:
        a, b = baseline[task], candidate[task]
        n = args.common_trials
        if min(len(a), len(b)) < n:
            raise ValueError(f'{task}: fewer than {n} trials')
        sa, sb = a[n - 1][0], b[n - 1][0]
        rows.append(dict(task=task, common_trials=n,
                         baseline_successes=sa, candidate_successes=sb,
                         baseline_success_pct=100 * sa / n,
                         candidate_success_pct=100 * sb / n,
                         difference_percentage_points=100 * (sb - sa) / n,
                         same_episode_seeds=[x[2] for x in a[:n]] == [x[2] for x in b[:n]],
                         candidate_full_successes=b[-1][0], candidate_full_trials=b[-1][1]))
    total = len(rows) * args.common_trials
    a_rate = 100 * sum(row['baseline_successes'] for row in rows) / total
    b_rate = 100 * sum(row['candidate_successes'] for row in rows) / total
    summary = dict(baseline_dir=str(args.baseline_dir.resolve()),
                   candidate_dir=str(args.candidate_dir.resolve()), tasks=len(rows),
                   common_trials_per_task=args.common_trials, common_trials_total=total,
                   baseline_success_pct=a_rate, candidate_success_pct=b_rate,
                   difference_percentage_points=round(b_rate - a_rate, 8),
                   tasks_with_identical_episode_seeds=sum(row['same_episode_seeds'] for row in rows),
                   seed_note='Episode seeds were compared. Instruction strings are not logged, so exact instruction identity was not checked.',
                   candidate_full_trials=sum(row['candidate_full_trials'] for row in rows),
                   candidate_full_mean_success_pct=sum(
                       100 * row['candidate_full_successes'] / row['candidate_full_trials'] for row in rows
                   ) / len(rows))
    output = args.output_dir or args.candidate_dir
    output.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output / 'comparison_to_pilot5k.csv', output / 'comparison_to_pilot5k.json'
    if csv_path.exists() or json_path.exists():
        raise FileExistsError('Comparison output already exists')
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
