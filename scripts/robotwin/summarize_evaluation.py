#!/usr/bin/env python3
"""Summarize only complete RoboTwin evaluations (one final result per task)."""
import argparse
import ast
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--mode", choices=["demo_clean", "demo_randomized"], default="demo_clean")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    tree = ast.parse((repo / "experiments/robotwin/data_registry/data_config.py").read_text())
    all_tasks = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_CLEAN50_TASKS"
                             for t in node.targets))
    tasks = args.tasks or all_tasks
    if args.trials < 1 or len(set(tasks)) != len(tasks) or not set(tasks) <= set(all_tasks):
        raise ValueError("Invalid trials or task list")
    rows = []
    for task in tasks:
        logs = list(args.log_dir.glob(f"{task}_{args.mode}_*_eval.log"))
        if len(logs) != 1:
            raise RuntimeError(f"{task}: expected one log, found {logs}")
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", logs[0].read_text(errors="replace"))
        results = re.findall(r"Success rate:\s*(\d+)\s*/\s*(\d+)", text)
        if not results or "Data has been saved to" not in text:
            raise RuntimeError(f"{task}: missing final result or completion marker")
        successes, trials = map(int, results[-1])
        if trials != args.trials or not 0 <= successes <= trials:
            raise RuntimeError(f"{task}: incomplete/invalid result {successes}/{trials}")
        rows.append(dict(task=task, successes=successes, trials=trials,
                         success_rate_pct=100 * successes / trials))
    with (args.log_dir / "task_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(mode=args.mode, tasks=len(rows), trials=sum(row["trials"] for row in rows),
                   mean_success_pct=sum(row["success_rate_pct"] for row in rows) / len(rows))
    (args.log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
