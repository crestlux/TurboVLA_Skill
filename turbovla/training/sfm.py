"""Single-GPU SFM training, effective-batch accumulation and strict resume.

Independent experiments are queued on GPUs. The confirmed formal recipe uses
batch 64 without accumulation; smaller microbatches are allowed only in smoke tests.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from turbovla.data.default_sfm import DefaultSFMDataset
from turbovla.models import TurboVLAConfig, build_turbovla
from .sfm_utils import (EMA, TaskBalancedSampler, build_optimizer, capture_rng, file_sha256,
                       forward_batch, masked_l1_sum, move_batch, normalize_images, parameter_hashes,
                       restore_rng, seed_everything, set_learning_rate, write_json)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'turbovla_sfm_checkpoint_v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_config(config):
    config = copy.deepcopy(config)
    if config['schema'] != 'turbovla_sfm_training_v1':
        raise ValueError('Not an SFM training configuration')
    train = config['train']
    if train['effective_batch_size'] != 64 or train['sampling'] != 'uniform_task_then_uniform_frame_with_replacement':
        raise ValueError('Require confirmed effective batch 64 and task-uniform sampling')
    micro = train['microbatch_size']
    if micro < 1 or 64 % micro:
        raise ValueError('Microbatch must be a positive divisor of 64')
    if not 0 < train['warmup_samples'] < train['max_samples']:
        raise ValueError('Invalid sample-based warmup/budget')
    if train['max_samples'] != 80 * config['data']['expected_frames']:
        raise ValueError('Require the confirmed 80-dataset-pass sample budget')
    if min(train['num_workers'], train['cpu_threads']) < 0 or train['cpu_threads'] < 1:
        raise ValueError('Invalid CPU resource setting')
    if config['wandb']['mode'] != 'online':
        raise ValueError('Formal training requires W&B online')
    if config['initialization']['policy_checkpoint'] is not None:
        raise ValueError('The confirmed policy is newly initialized, with pretrained vision/text backbones')
    model = TurboVLAConfig.from_mapping(config['model'])
    if (model.action.action_dim, model.action.horizon, model.vision.image_size, model.vision.num_views) != (7,5,224,2):
        raise ValueError('Unexpected action/image contract')
    if model.vision.frozen or not model.text.frozen or not model.text.force_eval_when_frozen:
        raise ValueError('Fine-tune DINO and freeze/eval BERT as confirmed')
    if model.action.initialization_seed != train['seed']:
        raise ValueError('Common initialization streams must use the training seed')
    return config


def open_dataset(config):
    dataset = DefaultSFMDataset(resolve(config['data']['root']), config['variant'])
    data = config['data']
    if (len(dataset), dataset.manifest['episodes']) != (data['expected_frames'],data['expected_episodes']):
        raise ValueError('Dataset cohort changed')
    for key in ('normalization_sha256','data_contract_sha256'):
        if dataset.manifest[key] != data[key]:
            raise ValueError(f'Dataset {key} changed')
    spec = dataset.variant
    model = config['model']
    ft = model['ft_history']
    if model['action']['state_dim'] != spec['model_state_dim']:
        raise ValueError('State width differs from selected dataset view')
    if ft['enabled'] != (spec['ft_method'] != 'state_embedding') or ft['adaln'] != (spec['ft_method'] == 'gru_embedding_adaln'):
        raise ValueError('Model FT method differs from selected dataset view')
    if not dataset.contract['ft_normalized_in_dataset'] or not dataset.contract['model_must_not_normalize_ft_again']:
        raise ValueError('Expected already normalized FT')
    return dataset


def make_model(config, dataset):
    model_config = copy.deepcopy(config['model'])
    for name in ('text','vision'):
        model_config[name]['model_name_or_path'] = str(resolve(model_config[name]['model_name_or_path']).resolve())
    model = build_turbovla(model_config)
    lengths = {name: len(model.text_encoder.tokenizer(text)['input_ids'])
               for name,text in dataset.contract['instruction_mapping'].items()}
    if max(lengths.values()) > model.config.text.padding_length:
        raise ValueError(f'Task instruction would be truncated: {lengths}')
    return model, lengths


class LimitedBatches:
    """Finite iterator so short checks drain workers without discarded prefetch."""
    def __init__(self, sampler, steps):
        self.sampler, self.steps = sampler, steps

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.steps)

    def __len__(self):
        return self.steps


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def start_wandb(config, output, resume_id=None, job_type='train'):
    import wandb
    settings = config['wandb']
    logged_config = copy.deepcopy(config)
    if resume_id:
        logged_config['resume_parent_run_id'] = resume_id
    # A fresh tracking segment preserves any pre-crash history beyond the last
    # checkpoint. Global optimizer step/sample counters continue in the metrics.
    run = wandb.init(project=settings['project'], entity=settings.get('entity'),
                     group=settings['group'], name=output.name, job_type=job_type,
                     config=logged_config, dir=str(output), mode='online',
                     settings=wandb.Settings(init_timeout=60))
    if run is None or run.settings.mode != 'online':
        raise RuntimeError('W&B online initialization failed; no offline fallback')
    write_json(output/'wandb_run.json', dict(id=run.id, entity=run.entity, project=run.project,
                                           url=run.url, mode='online', parent_run_id=resume_id, initialized_at_utc=utc()))
    return run


def finish_wandb(run, output, final_step, state):
    import wandb
    path = f'{run.entity}/{run.project}/{run.id}'
    run.summary.update({'final_step':final_step,'final_state':state})
    run.finish(exit_code=0 if state in ('completed','interrupted','validated') else 1)
    error = None
    for attempt in range(5):
        try:
            remote = wandb.Api(timeout=30).run(path)
            if remote.summary.get('final_step') != final_step or remote.summary.get('final_state') != state:
                raise RuntimeError('Remote W&B summary does not yet match the finished run')
            write_json(output/'wandb_sync.json',dict(status='verified',run=path,final_step=final_step,
                                                   final_state=state,checked_at_utc=utc()))
            return
        except Exception as caught:
            error = caught
            if attempt < 4:
                time.sleep(2)
    raise RuntimeError(f'W&B final sync verification failed: {error}')


def train(config, output, *, smoke_steps=None, device='cuda:0', resume=None, save_smoke=False,
          profile_warmup=2, barrier=None):
    config = validate_config(config)
    output = Path(output).resolve()
    smoke = smoke_steps is not None
    if smoke and smoke_steps < 1:
        raise ValueError('Smoke steps must be positive')
    if output.exists() and not resume:
        raise FileExistsError('Select a new output directory or use --resume')
    output.mkdir(parents=True, exist_ok=True)
    settings = config['train']
    if not smoke and settings['microbatch_size'] != 64:
        raise ValueError('Confirmed formal SFM training uses batch 64 without gradient accumulation')
    torch.set_num_threads(settings['cpu_threads'])
    torch.set_num_interop_threads(1)
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('SFM GPU preparation/training requires CUDA')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The confirmed recipe requires BF16 support')
    seed_everything(settings['seed'])
    dataset = open_dataset(config)
    sampler = TaskBalancedSampler(dataset.shards,64,settings['seed'],settings['max_samples'])
    signature = dict(manifest_sha256=file_sha256(dataset.root/'manifest.json'),
                     normalization_sha256=dataset.manifest['normalization_sha256'],
                     data_contract_sha256=dataset.manifest['data_contract_sha256'],
                     variant=config['variant'], frames=len(dataset), episodes=dataset.manifest['episodes'])
    model, lengths = make_model(config,dataset)
    initial_hashes = parameter_hashes(model)
    model = model.to(device)
    optimizer = build_optimizer(model,settings)
    ema = EMA(model,settings['ema_decay'],settings['ema_reference_batch'])
    generator = torch.Generator().manual_seed(settings['seed']+100000)
    step, prior_seconds, previous_wandb_id, resume_sources = 0,0.,None,None
    if resume:
        # Only checkpoints produced by this trusted local runner.
        payload = torch.load(resume,map_location='cpu',weights_only=False)
        if payload['schema'] != SCHEMA or payload['config'] != config or payload['data_signature'] != signature:
            raise ValueError('Checkpoint training configuration or dataset contract changed')
        if payload['smoke_test'] != smoke:
            raise ValueError('A smoke checkpoint cannot initialize/resume formal training')
        model.load_state_dict(payload['raw'],strict=True)
        ema.shadow = {key:value.to(device) for key,value in payload['ema'].items()}
        optimizer.load_state_dict(payload['optimizer'])
        sampler.load_state_dict(payload['sampler'])
        generator.set_state(payload['loader_generator'])
        restore_rng(payload['rng'],device)
        step,prior_seconds = payload['step'],payload['elapsed_seconds']
        previous_wandb_id = payload['wandb_run_id']
        resume_sources = payload['source_sha256']
        initial_hashes = payload['initial_parameter_sha256']
        del payload
        metrics=output/'metrics.jsonl'
        if metrics.exists():
            lines=metrics.read_text().splitlines()
            retained=[line for line in lines if json.loads(line)['step']<=step]
            if retained!=lines:
                shutil.copyfile(metrics,output/f'metrics_before_resume_{time.time_ns()}.jsonl')
                metrics.write_text('\n'.join(retained)+'\n')
    elif (output/'config.json').exists():
        raise FileExistsError('Refusing to overwrite an existing run')
    remaining_steps = math.ceil((sampler.budget-sampler.seen)/64)
    steps_this_run = min(remaining_steps,smoke_steps) if smoke else remaining_steps
    if steps_this_run < 1:
        raise ValueError('No training steps remain')
    loader_kwargs = dict(batch_sampler=LimitedBatches(sampler,steps_this_run),num_workers=settings['num_workers'],
                         pin_memory=True,generator=generator)
    if settings['num_workers']:
        loader_kwargs.update(multiprocessing_context='spawn',prefetch_factor=2)
    loader = DataLoader(dataset,**loader_kwargs)
    run = None
    stopped = False
    started = time.perf_counter()
    initial_step = step
    measured_samples, measured_seconds, first_measured_at, last_measured_at = 0,0.,None,None
    profile = []
    iterator = None
    last_record = None
    write_json(output/'config.json',config)
    write_json(output/'data_signature.json',signature)
    write_json(output/'initialization.json',dict(parameter_sha256=initial_hashes,instruction_token_lengths=lengths,
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    preprocessing = output/'preprocessing'
    preprocessing.mkdir(exist_ok=True)
    for filename in ('normalization.json','data_contract.json','inference_config.json','manifest.json'):
        shutil.copyfile(dataset.root/filename,preprocessing/filename)
    source_files=sorted((ROOT/'turbovla').rglob('*.py'))+[ROOT/'scripts/default_sfm/train.py']
    current_sources={str(p.relative_to(ROOT)):file_sha256(p) for p in source_files}
    if resume_sources is not None and current_sources != resume_sources:
        raise ValueError('Training source changed since the checkpoint')
    for source in source_files:
        target=output/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copyfile(source,target)
        elif not target.exists() or file_sha256(target)!=file_sha256(source):
            raise ValueError(f'Training source changed on resume: {source.relative_to(ROOT)}')
    provenance = dict(started_at_utc=utc(),smoke_test=smoke,formal_training_started=not smoke,
        dataset_root=str(dataset.root.resolve()),pid=os.getpid(),device=str(device),
        gpu_name=torch.cuda.get_device_name(device),cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=current_sources)
    if (output/'run.json').exists():
        shutil.copyfile(output/'run.json',output/f'run_before_resume_{time.time_ns()}.json')
    write_json(output/'run.json',provenance)

    def status(state, **extra):
        elapsed = prior_seconds+time.perf_counter()-started
        remaining = sampler.budget-sampler.seen
        value = dict(status=state,updated_at_utc=utc(),pid=os.getpid(),variant=config['variant'],smoke_test=smoke,
            formal_training_started=not smoke,optimizer_steps=step,samples_seen=sampler.seen,target_samples=sampler.budget,
            progress_percent=100*sampler.seen/sampler.budget,equivalent_dataset_passes=sampler.seen/len(dataset),
            elapsed_seconds=elapsed,estimated_remaining_seconds=remaining*elapsed/sampler.seen if sampler.seen else None,
            effective_batch_size=64,microbatch_size=settings['microbatch_size'],**extra)
        if last_record is not None:
            value.setdefault('last_metric',last_record)
        write_json(output/'status.json',value)
        return value

    def save():
        payload = dict(schema=SCHEMA,config=config,data_signature=signature,smoke_test=smoke,step=step,
            raw=cpu_tree(model.state_dict()),ema=cpu_tree(ema.shadow),optimizer=cpu_tree(optimizer.state_dict()),
            sampler=sampler.state_dict(),rng=capture_rng(device),loader_generator=generator.get_state(),
            elapsed_seconds=prior_seconds+time.perf_counter()-started,wandb_run_id=run.id if run else None,
            initial_parameter_sha256=initial_hashes,source_sha256=current_sources)
        path=output/f'checkpoint_{step:07d}.pt';temporary=path.with_suffix('.pt.tmp')
        torch.save(payload,temporary);temporary.replace(path)
        latest=output/'latest.pt';temporary_link=output/'latest.pt.tmp'
        temporary_link.unlink(missing_ok=True)
        os.link(path,temporary_link);temporary_link.replace(latest)
        for old in sorted(output.glob('checkpoint_*.pt'))[:-settings['keep_checkpoints']]:
            old.unlink()
        write_json(output/'checkpoint.json',dict(path=path.name,step=step,samples_seen=sampler.seen,
                                               sha256=file_sha256(path),smoke_test=smoke))
        return path

    def request_stop(signum, frame):
        nonlocal stopped
        stopped=True

    original_handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        if not smoke:
            run=start_wandb(config,output,previous_wandb_id)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        iterator=iter(loader)
        status('smoke_running' if smoke else 'running')
        with (output/'metrics.jsonl').open('a',buffering=1) as stream:
            for batch in iterator:
                local_step=step-initial_step
                if smoke and barrier and local_step==profile_warmup:
                    write_json(output/'ready.json',dict(ready_at_utc=utc(),pid=os.getpid()))
                    deadline=time.monotonic()+300
                    while not Path(barrier).exists():
                        if time.monotonic()>deadline or stopped:
                            raise RuntimeError('Benchmark barrier timed out or was interrupted')
                        time.sleep(.1)
                torch.cuda.synchronize(device)
                update_started=time.perf_counter()
                update_wall=time.time()
                count=len(batch['state'])
                denominator=int(batch['action_mask'].sum())*7
                if denominator<1:
                    raise ValueError('No valid action targets')
                set_learning_rate(optimizer,settings,sampler.seen,count)
                optimizer.zero_grad(set_to_none=True)
                loss_sum=torch.zeros((),device=device)
                for start in range(0,count,settings['microbatch_size']):
                    micro=move_batch(batch,device,start,start+settings['microbatch_size'])
                    pixels=normalize_images(micro['images'],config['images'],augment=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=settings['amp']):
                        predicted=forward_batch(model,micro,pixels)
                        part=masked_l1_sum(predicted,micro['actions'],micro['action_mask'])
                    if not torch.isfinite(part):
                        raise FloatingPointError('Nonfinite loss; optimizer not updated')
                    (part/denominator).backward()
                    loss_sum+=part.detach()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['max_grad_norm'],error_if_nonfinite=True)
                if local_step==0:
                    groups={'vision':'vision_encoder.backbone.','language_projection':'text_encoder.text_projection.',
                            'ft_gru':'action_head.ft_history_encoder.gru.','adaln':'action_head.decoder.adaln_modulations.'}
                    gradients={name:sum(float(p.grad.detach().float().square().sum()) for key,p in model.named_parameters()
                                       if key.startswith(prefix) and p.grad is not None)**.5 for name,prefix in groups.items()}
                    required=['vision','language_projection']
                    if config['model']['ft_history']['enabled']: required.append('ft_gru')
                    if config['model']['ft_history']['adaln']: required.append('adaln')
                    if any(gradients[name]<=0 for name in required):
                        raise ValueError(f'Required module did not receive gradients: {gradients}')
                    if any(p.grad is not None for p in model.text_encoder.bert.parameters()):
                        raise ValueError('Frozen BERT received gradients')
                    write_json(output/'gradient_validation.json',dict(status='passed',norms=gradients,bert_frozen=True))
                optimizer.step();ema.update(model,count);sampler.consume(count);step+=1
                torch.cuda.synchronize(device)
                seconds=time.perf_counter()-update_started
                record=dict(step=step,samples_seen=sampler.seen,loss=float(loss_sum/denominator),gradient_norm=float(grad),
                    update_seconds=seconds,samples_per_second=count/seconds,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                    learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
                last_record=record
                if local_step>=profile_warmup:
                    measured_samples+=count;measured_seconds+=seconds
                    first_measured_at=update_wall if first_measured_at is None else first_measured_at
                    last_measured_at=time.time()
                    if smoke: profile.append(record)
                status('smoke_running' if smoke else 'running',last_metric=record)
                if step==1 or step%settings['log_interval']==0 or step-initial_step==steps_this_run or stopped:
                    stream.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                    if run:
                        run.log({'train/'+key:value for key,value in record.items() if key!='learning_rates'} |
                                {'lr/'+key:value for key,value in record['learning_rates'].items()},step=step)
                if (not smoke and (step%settings['save_interval']==0 or sampler.seen==sampler.budget or stopped)):
                    save()
                if stopped:
                    break
        if smoke and save_smoke:
            save()
        final_state='interrupted' if stopped else ('smoke_complete' if smoke else 'completed')
        result=status(final_state,profile=dict(warmup_steps=profile_warmup,measured_samples=measured_samples,
            measured_update_seconds=measured_seconds,first_measured_unix=first_measured_at,last_measured_unix=last_measured_at,
            update_samples_per_second=measured_samples/measured_seconds if measured_seconds else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,steps=profile))
        if run:
            finish_wandb(run,output,step,final_state)
        write_json(output/'result.json',result)
        return result
    except BaseException as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        if run:
            try: finish_wandb(run,output,step,'failed')
            except Exception as sync_error:
                write_json(output/'wandb_sync.json',dict(status='failed',error=str(sync_error)))
        raise
    finally:
        if iterator is not None and hasattr(iterator,'_shutdown_workers'):
            iterator._shutdown_workers()
        dataset.close()
        for sig,handler in original_handlers.items(): signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train',action='store_true')
    mode.add_argument('--smoke-steps',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--microbatch',type=int)
    parser.add_argument('--save-smoke-checkpoint',action='store_true')
    parser.add_argument('--profile-warmup',type=int,default=2)
    parser.add_argument('--barrier',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.microbatch is not None: config['train']['microbatch_size']=args.microbatch
    if args.profile_warmup<0 or (args.barrier and args.smoke_steps is None):
        raise ValueError('Invalid benchmark-only settings')
    train(config,args.output,smoke_steps=args.smoke_steps,device=args.device,resume=args.resume,
          save_smoke=args.save_smoke_checkpoint,profile_warmup=args.profile_warmup,barrier=args.barrier)


if __name__=='__main__':
    main()
