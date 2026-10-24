#!/usr/bin/env python3
"""Run independent SFM experiments on idle GPUs, or measure a finite smoke group."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from turbovla.training.sfm_utils import write_json

METHODS=('ft_state','ft_gru','ft_gru_adaln')
STATES=('xy_keep','xy_remove','xy_vxy_remove')


def gpu_inventory():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    return {int(row[0]):dict(index=int(row[0]),uuid=row[1].strip(),name=row[2].strip(),
                            total_mib=int(row[3]),used_mib=int(row[4]),utilization=int(row[5]))
            for row in csv.reader(io.StringIO(text))}


def assert_idle(gpus):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    selected={gpu['uuid'] for gpu in gpus}
    occupied=[row for row in csv.reader(io.StringIO(text)) if row and row[0].strip() in selected]
    if occupied:
        raise RuntimeError(f'Selected GPUs have existing compute processes; preserve them: {occupied}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir',type=Path,default=ROOT/'experiments/default_sfm/training_configs')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='0,1,2')
    parser.add_argument('--jobs-per-gpu',type=int,required=True)
    parser.add_argument('--microbatch',type=int,required=True)
    parser.add_argument('--variants',help='Comma-separated subset; default covers the balanced nine-way matrix')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train',action='store_true')
    mode.add_argument('--smoke-steps',type=int)
    parser.add_argument('--profile-warmup',type=int,default=3)
    parser.add_argument('--resume',action='store_true',help='Resume an existing formal group from each latest.pt')
    parser.add_argument('--queue-smoke',action='store_true',help='Smoke all variants through the queue, without a simultaneous-start barrier')
    args=parser.parse_args()
    profile=args.smoke_steps is not None
    if args.queue_smoke and not profile:
        raise ValueError('--queue-smoke requires --smoke-steps')
    if not profile and args.microbatch!=64:
        raise ValueError('Confirmed formal training requires batch 64 without accumulation')
    mode_name='smoke_queue' if args.queue_smoke else 'smoke_profile' if profile else 'formal_training'
    if args.resume and profile:
        raise ValueError('Group resume is for formal training; smoke resume uses train.py')
    if not 1<=args.jobs_per_gpu<=3 or args.microbatch not in (1,2,4,8,16,32,64):
        raise ValueError('Use 1..3 jobs/GPU and a microbatch divisor of 64')
    if profile and args.smoke_steps<=args.profile_warmup:
        raise ValueError('Smoke must include measured steps after warmup')
    inventory=gpu_inventory()
    selected=[inventory[int(index)] for index in args.gpus.split(',')]
    if len({gpu['index'] for gpu in selected})!=len(selected):
        raise ValueError('Duplicate GPU')
    assert_idle(selected)
    variants=args.variants.split(',') if args.variants else [f'{method}_{state}' for state in STATES for method in METHODS]
    if profile and not args.queue_smoke and args.variants is None:
        variants=variants[:len(selected)*args.jobs_per_gpu]
    if len(set(variants))!=len(variants) or (profile and not args.queue_smoke and len(variants)>len(selected)*args.jobs_per_gpu):
        raise ValueError('Profile requires distinct variants fitting in one concurrent wave')
    for variant in variants:
        config=json.loads((args.config_dir/f'{variant}.json').read_text())
        if config['variant']!=variant or config['train']['effective_batch_size']!=64:
            raise ValueError('Configuration/variant mismatch')
    output=args.output.resolve()
    if args.resume:
        previous=json.loads((output/'launch.json').read_text())
        if previous['mode']!='formal_training' or {job['variant'] for job in previous['jobs']}!=set(variants):
            raise ValueError('Resume requires the same formal experiment group')
    output.mkdir(parents=True,exist_ok=args.resume)
    barrier=output/'measure_start.json'
    jobs=[]
    for index,variant in enumerate(variants):
        # Cyclic assignment gives each GPU different FT/state combinations.
        gpu=selected[(index%3+index//3)%len(selected)]
        jobs.append(dict(variant=variant,gpu=gpu['index'],gpu_uuid=gpu['uuid'],status='queued',
                         output=str(output/variant),pid=None,exit_code=None))
    if args.resume:
        for job in jobs:
            directory=Path(job['output'])
            completed=directory/'result.json'
            result=json.loads(completed.read_text()) if completed.exists() else {}
            if result.get('status')=='completed':
                job.update(status='completed',step=result['optimizer_steps'],
                           progress_percent=100.,last_metric=result.get('last_metric'),exit_code=0)
            elif not directory.exists():
                continue  # A queued variant that had never started needs a fresh run.
            else:
                checkpoint=directory/'latest.pt'
                if not checkpoint.exists():
                    raise FileNotFoundError(f'No committed checkpoint for {job["variant"]}; preserve this group and start that variant in a new output directory')
                job['resume_checkpoint']=str(checkpoint)
        history=output/'group_history'/str(time.time_ns())
        history.mkdir(parents=True)
        for filename in ('launch.json','status.json','result.json'):
            if (output/filename).exists(): shutil.copyfile(output/filename,history/filename)
    write_json(output/'launch.json',dict(mode=mode_name,
        created_at_utc=datetime.now(timezone.utc).isoformat(),jobs_per_gpu=args.jobs_per_gpu,
        microbatch_size=args.microbatch,effective_batch_size=64,gpus=selected,jobs=jobs,
        command=sys.argv,resume=args.resume,smoke_steps=args.smoke_steps,profile_warmup=args.profile_warmup))
    active={};logs={};peak={gpu['index']:gpu['used_mib'] for gpu in selected}
    stopped=False;failed=False;stop_at=None

    def stop(signum,frame):
        nonlocal stopped
        stopped=True

    original={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    started=time.monotonic();last_print=-100.
    try:
        with (output/'gpu_metrics.jsonl').open('a' if args.resume else 'w',buffering=1) as metrics:
            while any(job['status']=='queued' for job in jobs) or active:
                if stopped or failed:
                    if stop_at is None:
                        stop_at=time.monotonic()
                        for process in active.values(): process.send_signal(signal.SIGTERM)
                        for job in jobs:
                            if job['status']=='queued': job['status']='not_started'
                    elif time.monotonic()-stop_at>180:
                        for process in active.values():
                            if process.poll() is None: os.killpg(process.pid,signal.SIGKILL)
                else:
                    for job in jobs:
                        if job['status']!='queued': continue
                        occupied=sum(other['gpu']==job['gpu'] and other['variant'] in active for other in jobs)
                        if occupied>=args.jobs_per_gpu: continue
                        log=(output/(job['variant']+'.log')).open('a' if args.resume else 'w')
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=job['gpu_uuid'],PYTHONUNBUFFERED='1',
                                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
                        command=[sys.executable,str(ROOT/'scripts/default_sfm/train.py'),'--config',
                                 str((args.config_dir/f"{job['variant']}.json").resolve()),'--output',job['output'],
                                 '--microbatch',str(args.microbatch)]
                        if profile:
                            command+=['--smoke-steps',str(args.smoke_steps),'--profile-warmup',str(args.profile_warmup)]
                            if not args.queue_smoke: command+=['--barrier',str(barrier)]
                        else: command+=['--train']
                        if job.get('resume_checkpoint'): command+=['--resume',job['resume_checkpoint']]
                        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        active[job['variant']]=process;logs[job['variant']]=log
                        job.update(status='starting',pid=process.pid)
                        print(json.dumps({'launched':job['variant'],'gpu':job['gpu'],'pid':process.pid}),flush=True)
                for job in jobs:
                    if job['variant'] not in active: continue
                    process=active[job['variant']]
                    status_file=Path(job['output'])/'status.json'
                    if status_file.exists():
                        status=json.loads(status_file.read_text())
                        job.update(status=status['status'],step=status['optimizer_steps'],
                                   progress_percent=status['progress_percent'],last_metric=status.get('last_metric',job.get('last_metric')))
                    code=process.poll()
                    if code is not None:
                        job['exit_code']=code
                        job['status']='smoke_complete' if profile and code==0 else 'completed' if code==0 else 'failed'
                        if stopped and code==0: job['status']='interrupted'
                        if code!=0: failed=True
                        logs.pop(job['variant']).close();del active[job['variant']]
                if profile and not args.queue_smoke and not barrier.exists() and all((Path(job['output'])/'ready.json').exists() for job in jobs):
                    write_json(barrier,dict(start_unix=time.time()))
                    print('All workers ready; synchronized measurement started.',flush=True)
                current=gpu_inventory()
                for gpu in selected:
                    peak[gpu['index']]=max(peak[gpu['index']],current[gpu['index']]['used_mib'])
                metrics.write(json.dumps({'time_unix':time.time(),'gpus':[current[g['index']] for g in selected]})+'\n')
                state='failed' if failed else 'interrupted' if stopped else 'running' if active or any(j['status']=='queued' for j in jobs) else 'complete'
                write_json(output/'status.json',dict(status=state,elapsed_seconds=time.monotonic()-started,
                    mode=mode_name,jobs=jobs,
                    peak_gpu_used_gib={str(key):value/1024 for key,value in peak.items()}))
                if time.monotonic()-last_print>=10:
                    print(' | '.join(f"{job['variant']} GPU{job['gpu']} {job['status']} step={job.get('step',0)}" for job in jobs),flush=True)
                    last_print=time.monotonic()
                if active: time.sleep(2)
        result=dict(status='failed' if failed else 'interrupted' if stopped else 'passed',
            mode=mode_name,jobs_per_gpu=args.jobs_per_gpu,
            microbatch_size=args.microbatch,effective_batch_size=64,jobs=jobs,
            peak_gpu_used_gib={str(key):value/1024 for key,value in peak.items()},elapsed_seconds=time.monotonic()-started)
        if profile and not failed and not stopped:
            records=[json.loads((Path(job['output'])/'result.json').read_text()) for job in jobs]
            begin=min(row['profile']['first_measured_unix'] for row in records)
            end=max(row['profile']['last_measured_unix'] for row in records)
            samples=sum(row['profile']['measured_samples'] for row in records)
            result.update(measured_samples=samples,measurement_wall_seconds=end-begin,
                          aggregate_samples_per_second=samples/(end-begin),
                          measurement_scope=('queue smoke includes subsequent process startup gaps; not a steady throughput benchmark' if args.queue_smoke else
                                             'earliest post-warmup update start to latest update completion; includes between-update data waits'),
                          per_job=[dict(variant=row['variant'],profile=row['profile']) for row in records])
        write_json(output/'result.json',result)
        print(json.dumps({key:value for key,value in result.items() if key not in ('jobs','per_job')},indent=2),flush=True)
        if failed or stopped: raise SystemExit(1)
    finally:
        for process in active.values():
            if process.poll() is None: process.send_signal(signal.SIGTERM)
        for log in logs.values(): log.close()
        for sig,handler in original.items(): signal.signal(sig,handler)


if __name__=='__main__': main()
