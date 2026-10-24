#!/usr/bin/env python3
"""Persistently run eight independent Jack ablations sequentially on GPU 1."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.real_robot.launch_go import source_hashes
from turbovla.training.real_robot import write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def supervise(run):
    manifest = json.loads((run / 'schedule.json').read_text())
    gpu = manifest['gpu_uuid']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu
    assert source_hashes(ROOT) == manifest['source_sha256']
    assert json.loads((run / 'preparation_status.json').read_text())['passed']
    status = dict(phase='starting', pid=os.getpid(), gpu_index=1, gpu_uuid=gpu, started_utc=now(), completed=[], failed=[])

    def update(phase, **extra):
        status.update(phase=phase, updated_utc=now(), **extra)
        write_json(run / 'queue_status.json', status)
        print(json.dumps(status), flush=True)

    def execute(arguments, log):
        with log.open('a', buffering=1) as stream:
            subprocess.run([sys.executable, *arguments], cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    # One cooperative lock for this physical GPU, shared across Jack queues.
    with (run.parent / '.jack_gpu1.lock').open('a') as lock:
        update('waiting_for_queue_lock')
        fcntl.flock(lock, fcntl.LOCK_EX)
        for job in manifest['jobs']:
            name, mode, output, stage = job['name'], job['mode'], Path(job['output']), Path(job['stage'])
            try:
                prior = json.loads((output / 'job_status.json').read_text())
                if prior['phase'] == 'completed':
                    status['completed'].append(name)
                    continue
                if (output / 'config.json').exists():
                    raise RuntimeError('Existing production run requires explicit checkpoint resume; refusing to overwrite')
                while True:
                    query = subprocess.check_output(['nvidia-smi', '-i', gpu,
                            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                    index, uuid, memory, utilization = [v.strip() for v in query.strip().split(',')]
                    assert int(index) == 1 and uuid == gpu
                    if int(memory) <= 512 and int(utilization) <= 5:
                        break
                    update('waiting_for_gpu', current_job=name, gpu_memory_mib=int(memory))
                    time.sleep(15)
                update('preflight', current_job=name)
                write_json(output / 'job_status.json', dict(phase='preflight', name=name, gpu_uuid=gpu))
                execute(['scripts/real_robot/preflight_go.py', '--directory', str(stage/'preflight'), '--mode', mode], output/'preflight.log')
                verified = json.loads((stage/'preflight'/f'verified_{mode}.json').read_text())
                assert verified['passed'] and verified['training_config'] == json.loads((output/'recipe.json').read_text())
                benchmark = json.loads((stage/'preflight'/f'benchmark_{mode}.json').read_text())
                assert any(v['batch_size'] == 64 and v['status'] == 'passed' and v['peak_reserved_gib'] <= 22 for v in benchmark['results'])
                for path, digest in verified['source_sha256'].items():
                    assert manifest['source_sha256'][path] == digest
                # Smoke weights are never used to initialize production training.
                for path in (stage/'preflight'/f'smoke_{mode}').glob('*.pt'):
                    path.unlink()
                update('training_and_evaluation', current_job=name)
                execute(['scripts/real_robot/launch_go.py', '--worker', mode, '--run', str(stage)], output/'worker.log')
                assert json.loads((output/'job_status.json').read_text())['phase'] == 'completed'
                status['completed'].append(name)
                update('job_completed', current_job=name)
            except Exception as error:
                failure = dict(name=name, error=f'{type(error).__name__}: {error}', finished_utc=now())
                status['failed'].append(failure)
                write_json(output/'job_status.json', dict(phase='failed', **failure))
                update('job_failed', current_job=name)
                # Independent jobs continue; failures are retained for inspection.
        update('completed' if not status['failed'] else 'finished_with_failures', current_job=None, finished_utc=now())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    supervise(parser.parse_args().run.resolve())
