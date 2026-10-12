#!/usr/bin/env python3
"""Run a bounded evaluation command and sample GPU and process-tree resources.

GPU memory includes the renderer, CUDA, and model allocations reported by NVML.
CPU 100% means one logical CPU. RSS is the sum across the command's session,
so shared pages can be counted more than once. Peaks are sampled, not exact.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


class EvaluationLogError(RuntimeError):
    """Rendering or GPU allocation errors invalidate the evaluation."""


class LogGuard:
    pattern = re.compile(r'OIDN Error:|cannot create buffer|out of memory|OutOfMemoryError|'
                         r'ErrorOutOfDeviceMemory|VK_ERROR_OUT_OF_DEVICE_MEMORY', re.I)

    def __init__(self, directory):
        self.directory = directory
        self.positions = {}
        self.tails = {}

    def check(self):
        if self.directory is None:
            return
        for path in sorted(self.directory.glob('*.log')):
            with path.open('rb') as handle:
                handle.seek(self.positions.get(path, 0))
                while block := handle.read(1024 * 1024):
                    text = self.tails.get(path, '') + block.decode(errors='replace')
                    match = self.pattern.search(text)
                    if match:
                        line = text[text.rfind('\n', 0, match.start())+1:].split('\n', 1)[0]
                        raise EvaluationLogError(f'{path}: {line[:1000]}')
                    self.tails[path] = text[-256:]
                self.positions[path] = handle.tell()


def timestamp():
    return datetime.datetime.now().astimezone().isoformat()


def system_cpu():
    return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))


def session_processes(session):
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            row = path.read_text().rsplit(')', 1)[1].split()
            if int(row[3]) != session:
                continue
            pid = int(path.parent.name)
            result[pid] = dict(ticks=int(row[11])+int(row[12]), start=int(row[19]),
                               rss_bytes=int(row[21])*os.sysconf('SC_PAGE_SIZE'),
                               threads=int(row[17]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def gpu_state(gpus):
    output = subprocess.check_output([
        'nvidia-smi', '--id='+gpus,
        '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [dict(index=int(r[0]), uuid=r[1].strip(), total_mib=int(r[2]),
                 used_mib=int(r[3]), free_mib=int(r[4]), utilization_pct=int(r[5]),
                 memory_utilization_pct=int(r[6])) for r in csv.reader(output.splitlines())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    parser.add_argument('--abort-on-render-error-dir', type=Path,
                        help='Watch evaluation/server logs and abort on rendering or GPU allocation errors')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.interval <= 0 or args.timeout_seconds <= 0:
        parser.error('Need a command and positive interval/timeout')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at=timestamp(), command=command, gpus=args.gpus,
                    abort_on_render_error_dir=str(args.abort_on_render_error_dir) if args.abort_on_render_error_dir else None,
                    interval_seconds=args.interval, timeout_seconds=args.timeout_seconds,
                    notes=['GPU peaks are sampled at the stated interval.',
                           'Process CPU 100% is one logical CPU; RSS can double-count shared pages.'],
                    environment={key: value for key, value in os.environ.items()
                                 if key.startswith('ROBOTWIN_') or key in ['CUDA_VISIBLE_DEVICES', 'RUN_ID']})
    baseline_gpu = gpu_state(args.gpus)
    manifest['baseline_gpu'] = baseline_gpu
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    samples = []
    exit_reason = 'exited'
    error = None
    guard = LogGuard(args.abort_on_render_error_dir)
    with (args.output/'command.log').open('w') as log, (args.output/'resources.jsonl').open('w', buffering=1) as raw:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (args.output/'pid').write_text(str(proc.pid)+'\n')
        start = previous_time = time.monotonic()
        previous_cpu = system_cpu()
        previous_processes = {}
        try:
            while True:
                now = time.monotonic()
                current_cpu = system_cpu()
                processes = session_processes(proc.pid)
                elapsed = max(now-previous_time, 1e-6)
                ticks = 0
                for pid, item in processes.items():
                    previous = previous_processes.get(pid)
                    ticks += max(0, item['ticks']-(previous['ticks'] if previous and previous['start']==item['start'] else 0))
                delta = [a-b for a,b in zip(current_cpu, previous_cpu)]
                total = max(sum(delta), 1)
                sample = dict(timestamp=timestamp(), elapsed_seconds=now-start,
                              system_cpu_busy_pct=100*(sum(delta)-delta[3]-delta[4])/total,
                              system_cpu_iowait_pct=100*delta[4]/total,
                              process_cpu_pct=100*ticks/os.sysconf('SC_CLK_TCK')/elapsed,
                              process_rss_mib=sum(p['rss_bytes'] for p in processes.values())/1048576,
                              process_count=len(processes), thread_count=sum(p['threads'] for p in processes.values()),
                              pids=sorted(processes), gpu=gpu_state(args.gpus))
                raw.write(json.dumps(sample)+'\n')
                samples.append(sample)
                previous_cpu, previous_processes, previous_time = current_cpu, processes, now
                guard.check()
                if proc.poll() is not None:
                    break
                if now-start > args.timeout_seconds:
                    exit_reason = 'timeout'
                    raise TimeoutError('Evaluation exceeded profile timeout')
                time.sleep(args.interval)
        except BaseException as exc:
            error = str(exc)
            exit_reason = 'timeout' if isinstance(exc, TimeoutError) else type(exc).__name__
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if not isinstance(exc, (TimeoutError, KeyboardInterrupt, EvaluationLogError)):
                raise
        wall = time.monotonic()-start
        rc = proc.wait()
    measurements = samples[1:] or samples
    summary = dict(manifest, finished_at=timestamp(), exit_code=rc, exit_reason=exit_reason,
                   error=error,
                   status='completed' if rc==0 and exit_reason=='exited' else 'failed',
                   wall_seconds=wall, sample_count=len(samples),
                   process_cpu_mean_pct=sum(s['process_cpu_pct'] for s in measurements)/len(measurements),
                   process_cpu_peak_pct=max(s['process_cpu_pct'] for s in measurements),
                   process_rss_peak_mib=max(s['process_rss_mib'] for s in samples),
                   system_cpu_mean_pct=sum(s['system_cpu_busy_pct'] for s in measurements)/len(measurements),
                   gpu=[])
    for baseline in baseline_gpu:
        rows=[g for sample in samples for g in sample['gpu'] if g['index']==baseline['index']]
        summary['gpu'].append(dict(index=baseline['index'], uuid=baseline['uuid'],
                                   baseline_used_mib=baseline['used_mib'],
                                   peak_used_mib=max(g['used_mib'] for g in rows),
                                   minimum_free_mib=min(g['free_mib'] for g in rows),
                                   mean_utilization_pct=sum(g['utilization_pct'] for g in rows)/len(rows),
                                   peak_utilization_pct=max(g['utilization_pct'] for g in rows)))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(0 if summary['status']=='completed' else (rc if rc>0 else 1))


if __name__=='__main__':
    main()
