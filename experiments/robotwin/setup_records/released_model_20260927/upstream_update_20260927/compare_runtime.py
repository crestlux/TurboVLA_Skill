"""Bounded offline audit; never modifies or connects to active policy servers."""
from __future__ import annotations

import datetime
import argparse
import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path('/home/jack/TurboVLA')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/starvla_runtime')]

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from starVLA.model.framework.base_framework import baseframework

torch.set_num_threads(2)
torch.manual_seed(42)
CHECKPOINT = ROOT / 'pretrained/converted/turbovla_robotwin_cb530054_55k_ema/steps_55000_ema_model.safetensors'
DATA = Path('/home/jack/datasets/RoboTwin/Clean')
ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
TASKS = ['adjust_bottle', 'beat_block_hammer', 'click_bell', 'dump_bin_bigbin', 'grab_roller', 'lift_pot']


def normalize(values, stats):
    low, high = np.asarray(stats['min']), np.asarray(stats['max'])
    result = np.array(values, dtype=np.float32, copy=True)
    result[..., :12] = 2 * (result[..., :12] - low[:12]) / (high[:12] - low[:12]) - 1
    result[..., 12:] = (result[..., 12:] > 0.49).astype(np.float32)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-layernorm', action='store_true')
    args = parser.parse_args()
    stem = 'post_layernorm_comparison' if args.post_layernorm else 'runtime_output_comparison'
    output = HERE / f'{stem}.json'
    if output.exists():
        raise FileExistsError(output)
    stats = json.loads((CHECKPOINT.parent / 'dataset_statistics.json').read_text())['new_embodiment']
    examples, targets, identities = [], [], []
    for task in TASKS:
        folder = DATA / task
        episode = json.loads((folder / 'meta/episodes.jsonl').read_text().splitlines()[0])
        table = pq.read_table(folder / 'data/chunk-000/episode_000000.parquet', columns=['observation.state', 'action'])
        states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)[:, ORDER]
        actions = normalize(np.asarray(table['action'].to_pylist(), dtype=np.float32)[:, ORDER], stats['action'])
        indices = [max(1, episode['length'] // 4), episode['length'] // 2]
        frames = {index: [] for index in indices}
        for camera in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
            path = folder / f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4'
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                for index, frame in enumerate(container.decode(stream)):
                    if index in frames:
                        frames[index].append(cv2.resize(frame.to_ndarray(format='rgb24'), (224, 224), interpolation=cv2.INTER_AREA))
                    if index >= max(indices):
                        break
                stream.codec_context.close(strict=False)
        for index in indices:
            assert len(frames[index]) == 3
            examples.append({'image': frames[index], 'lang': episode['tasks'][0],
                             'state': normalize(states[index], stats['state'])[None]})
            targets.append(actions[np.minimum(np.arange(index, index + 50), len(actions) - 1)])
            identities.append({'task': task, 'episode': 0, 'frame': index})
    print(f'Prepared {len(examples)} real dataset observations with deployment ordering/normalization.', flush=True)
    model = baseframework.from_pretrained(str(CHECKPOINT)).eval()
    package = types.ModuleType('updated_turbovla_audit')
    package.__path__ = [str(HERE / 'official_source/turbovla')]
    sys.modules[package.__name__] = package
    updated_module = importlib.import_module('updated_turbovla_audit.models.turbovla')
    updated_config = importlib.import_module('updated_turbovla_audit.models.configuration')
    config = model.model.config.to_dict()
    for key in ('backbone_type', 'resnet_weights', 'resnet_output_stride', 'load_pretrained'):
        config['vision'].pop(key, None)
    config['compatibility_profile'] = 'robotwin'
    reference = updated_module.TurboVLA(updated_config.TurboVLAConfig.from_mapping(config)).eval()
    reference.load_state_dict(model.model.state_dict(), strict=True)
    model.to(device='cuda', dtype=torch.bfloat16)
    reference.to(device='cuda', dtype=torch.bfloat16)
    feature_checks = []
    if args.post_layernorm:
        def select_post_layernorm(module, inputs, outputs):
            if not feature_checks:
                before, after = outputs.hidden_states[-1], outputs.last_hidden_state
                feature_checks.append(dict(
                    hidden_state_dtype=str(before.dtype), last_hidden_state_dtype=str(after.dtype),
                    max_abs_difference=float((before.float()-after.float()).abs().max()),
                    mean_abs_difference=float((before.float()-after.float()).abs().mean()),
                    last_hidden_state_equals_final_layernorm=torch.equal(module.norm(before), after),
                ))
            # Scoped to these isolated model instances; running servers and files are untouched.
            outputs.hidden_states = (*outputs.hidden_states[:-1], outputs.last_hidden_state)
            return outputs
        model.model.vision_encoder.backbone.register_forward_hook(select_post_layernorm)
        reference.vision_encoder.backbone.register_forward_hook(select_post_layernorm)
    torch.cuda.reset_peak_memory_stats()
    records, old_outputs, new_outputs = [], [], []
    with torch.inference_mode():
        for identity, example, target in zip(identities, examples, targets):
            instructions, samples, states = model._model_inputs([example])
            old = model.model(instructions, samples, states).float().cpu().numpy()[0]
            new = reference(instructions, samples, states).float().cpu().numpy()[0]
            assert np.isfinite(old).all() and np.isfinite(new).all()
            delta = np.abs(old-new)
            records.append(dict(**identity, max_abs_output_difference=float(delta.max()),
                                mean_abs_output_difference=float(delta.mean()),
                                old_normalized_action_mae=float(np.abs(old-target).mean()),
                                new_normalized_action_mae=float(np.abs(new-target).mean()),
                                gripper_threshold_disagreements=int(((old[:,12:] > .49) != (new[:,12:] > .49)).sum())))
            old_outputs.append(old)
            new_outputs.append(new)
    report = dict(checked_at=datetime.datetime.now().astimezone().isoformat(),
                  old_official_commit='b29ab1420baa5c663ec935df513f2012430beb67',
                  updated_official_commit='6727c875666f8d5dda8d8cca0043da200738fe73',
                  checkpoint=str(CHECKPOINT), samples=len(records), precision='bfloat16',
                  feature_source='last_hidden_state (post LayerNorm)' if args.post_layernorm else 'hidden_states[-1] (installed Transformers semantics)',
                  feature_checks=feature_checks,
                  normalization='Model order: 12 joints, then 2 binary grippers; min-max joints; threshold 0.49 grippers.',
                  limitation='Small in-distribution demonstration sample, not a rollout, unseen-task test, or calibrated success-rate estimate. MAE is a diagnostic only.',
                  max_abs_output_difference=max(r['max_abs_output_difference'] for r in records),
                  mean_abs_output_difference=float(np.mean([r['mean_abs_output_difference'] for r in records])),
                  old_normalized_action_mae=float(np.mean([r['old_normalized_action_mae'] for r in records])),
                  new_normalized_action_mae=float(np.mean([r['new_normalized_action_mae'] for r in records])),
                  gripper_threshold_disagreements=sum(r['gripper_threshold_disagreements'] for r in records),
                  gripper_values_compared=len(records)*50*2,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), records=records)
    if args.post_layernorm:
        baseline = np.load(HERE/'runtime_output_comparison.npz')['old']
        fixed = np.asarray(old_outputs)
        delta = np.abs(baseline-fixed)
        report['feature_fix_impact'] = dict(
            max_abs_output_difference=float(delta.max()),
            mean_abs_output_difference=float(delta.mean()),
            baseline_normalized_action_mae=float(np.abs(baseline-np.asarray(targets)).mean()),
            post_layernorm_normalized_action_mae=float(np.abs(fixed-np.asarray(targets)).mean()),
            gripper_threshold_disagreements=int(((baseline[...,12:]>.49)!=(fixed[...,12:]>.49)).sum()),
        )
    output.write_text(json.dumps(report, indent=2)+'\n')
    np.savez(HERE/f'{stem}.npz', old=np.asarray(old_outputs), new=np.asarray(new_outputs), targets=np.asarray(targets))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2), flush=True)


if __name__ == '__main__':
    main()
