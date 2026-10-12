"""Change only a specified training process tree's CPU affinity, with restoration data."""
import argparse
import datetime
import json
import os
from pathlib import Path


def descendants(root):
    result = {root}
    changed = True
    while changed:
        changed = False
        for path in Path('/proc').glob('[0-9]*/status'):
            try:
                parent = int(next(line.split()[1] for line in path.read_text().splitlines()
                                  if line.startswith('PPid:')))
                pid = int(path.parent.name)
            except (OSError, StopIteration):
                continue
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def start_time(tid):
    return Path(f'/proc/{tid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def restore(record):
    count = 0
    for thread in record['threads']:
        try:
            if start_time(thread['tid']) == thread['start_time']:
                os.sched_setaffinity(thread['tid'], thread['original_cpus'])
                count += 1
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
    # Threads created during the probe inherit the temporary mask. When a
    # process originally had one uniform mask, restore its new threads too.
    by_pid = {}
    for thread in record['threads']:
        by_pid.setdefault(thread['pid'], []).append(thread)
    for pid, threads in by_pid.items():
        masks = {tuple(thread['original_cpus']) for thread in threads}
        main = next((thread for thread in threads if thread['tid'] == pid), None)
        if len(masks) != 1 or main is None:
            continue
        try:
            if start_time(pid) != main['start_time']:
                continue
        except FileNotFoundError:
            continue
        target = set(next(iter(masks)))
        for path in Path(f'/proc/{pid}/task').glob('[0-9]*'):
            try:
                tid = int(path.name)
                if os.sched_getaffinity(tid) != target:
                    os.sched_setaffinity(tid, target)
                    count += 1
            except (ProcessLookupError, FileNotFoundError):
                pass
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['apply', 'restore'])
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--pids', type=int, nargs=2)
    args = parser.parse_args()
    if args.action == 'restore':
        record = json.loads(args.record.read_text())
        count = restore(record)
        record.update(status='restored', restored_at=datetime.datetime.now().astimezone().isoformat(),
                      restored_threads=count)
        args.record.write_text(json.dumps(record, indent=2) + '\n')
        print('Restored', count, 'thread affinities.')
        return
    if args.record.exists() or not args.pids or not args.run_id:
        raise ValueError('Apply requires two PIDs, run-id, and a new record path.')
    # This workstation has GPU0 on NUMA0 and GPU1 on NUMA1; verify sysfs at apply time.
    cpu_lists = [Path(f'/sys/bus/pci/devices/{pci}/local_cpulist').read_text().strip()
                 for pci in ['0000:18:00.0', '0000:af:00.0']]
    record = dict(status='prepared', applied_at=datetime.datetime.now().astimezone().isoformat(),
                  run_id=args.run_id, root_pids=args.pids, cpu_lists=cpu_lists, threads=[])
    for rank, pid in enumerate(args.pids):
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if args.run_id not in command or 'train_robotwin_clean_act_pi05_recipe.py' not in command:
            raise ValueError(f'PID {pid} is not the requested training rank.')
        cpus = set()
        for group in cpu_lists[rank].split(','):
            ends = list(map(int, group.split('-')))
            cpus.update(range(ends[0], ends[-1] + 1))
        for child in descendants(pid):
            for path in Path(f'/proc/{child}/task').glob('[0-9]*'):
                tid = int(path.name)
                try:
                    original = sorted(os.sched_getaffinity(tid))
                    target = sorted(cpus.intersection(original))
                    if not target:
                        raise ValueError(f'No allowed local CPU for thread {tid}')
                    record['threads'].append(dict(tid=tid, pid=child, rank=rank,
                                                  start_time=start_time(tid),
                                                  original_cpus=original, target_cpus=target))
                except ProcessLookupError:
                    pass
                except FileNotFoundError:
                    pass
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + '\n')
    try:
        for thread in record['threads']:
            try:
                if start_time(thread['tid']) == thread['start_time']:
                    os.sched_setaffinity(thread['tid'], thread['target_cpus'])
            except (ProcessLookupError, FileNotFoundError):
                # Video decoders create and retire short-lived threads continuously.
                continue
        record['status'] = 'applied'
    except Exception:
        restore(record)
        record['status'] = 'rolled_back_after_error'
        raise
    finally:
        args.record.write_text(json.dumps(record, indent=2) + '\n')
    print('Applied GPU-local CPU affinity to', len(record['threads']), 'threads:', cpu_lists)


if __name__ == '__main__':
    main()
