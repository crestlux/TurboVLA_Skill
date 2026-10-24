#!/usr/bin/env python3
"""Read an existing SFM parallel run without restarting any training process."""
import argparse
import json
from pathlib import Path
import sys
import time


def display(root):
    path=root/'status.json'
    if not path.exists():
        print(f'Waiting for {path}');return
    status=json.loads(path.read_text())
    print(f"{root.name} | {status['status']} | {status.get('mode','')} | {status.get('elapsed_seconds',0)/60:.1f} min")
    print(f"{'variant':32s} {'GPU':>3s} {'status':>16s} {'step':>8s} {'%':>7s} {'loss':>10s} {'obs/s':>8s}")
    for job in status.get('jobs',[]):
        metric=job.get('last_metric') or {}
        print(f"{job['variant']:32s} {job['gpu']:3d} {job['status']:>16s} {job.get('step',0):8d} "
              f"{job.get('progress_percent',0):7.3f} {metric.get('loss',float('nan')):10.5f} "
              f"{metric.get('samples_per_second',0):8.2f}")
    print('Peak whole-GPU GiB:',status.get('peak_gpu_used_gib',{}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=5)
    args=parser.parse_args()
    while True:
        if args.watch and sys.stdout.isatty(): print('\033[2J\033[H',end='')
        display(args.run_dir)
        if not args.watch: break
        time.sleep(args.interval)


if __name__=='__main__': main()
