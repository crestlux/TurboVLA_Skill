#!/usr/bin/env python3
"""Reload local SFM raw/EMA weights and predict recorded observations; no robot I/O."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import default_collate

from turbovla.data.sfm_preprocessing import normalize
from turbovla.training.sfm import SCHEMA, cpu_tree, make_model, open_dataset, validate_config
from turbovla.training.sfm_utils import (TaskBalancedSampler, build_optimizer, capture_rng, file_sha256,
    forward_batch, move_batch, normalize_images, restore_rng, write_json)
from compare_checkpoints import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', default='0,138984')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Load only a trusted local checkpoint produced by this repository.
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload['schema'] != SCHEMA:
        raise ValueError('Not an SFM checkpoint')
    config = validate_config(payload['config'])
    torch.set_num_threads(config['train']['cpu_threads'])
    dataset = open_dataset(config)
    signature = payload['data_signature']
    if file_sha256(dataset.root / 'manifest.json') != signature['manifest_sha256']:
        raise ValueError('Checkpoint and dataset manifest differ')
    bundle = args.checkpoint.parent / 'preprocessing'
    for filename, key in [('normalization.json', 'normalization_sha256'),
                          ('data_contract.json', 'data_contract_sha256')]:
        if file_sha256(bundle / filename) != signature[key]:
            raise ValueError(f'Checkpoint preprocessing bundle differs: {filename}')
    stats = json.loads((bundle / 'normalization.json').read_text())
    device = torch.device(args.device)
    indices = [int(index) for index in args.indices.split(',')]
    batch = move_batch(default_collate([dataset[index] for index in indices]), device)
    pixels = normalize_images(batch['images'], config['images'], augment=False)
    model, _ = make_model(config, dataset)
    model.to(device).eval()
    results = {}
    restored = dict(tensor_count=0, tensor_elements=0, mismatch_count=0, first_mismatches=[])
    optimizer = build_optimizer(model, config['train'])
    optimizer.load_state_dict(payload['optimizer'])
    compare(payload['optimizer'], cpu_tree(optimizer.state_dict()), 'optimizer', restored)
    sampler = TaskBalancedSampler(dataset.shards, 64, config['train']['seed'], config['train']['max_samples'])
    sampler.load_state_dict(payload['sampler'])
    compare(payload['sampler'], sampler.state_dict(), 'sampler', restored)
    restore_rng(payload['rng'], device)
    compare(payload['rng'], capture_rng(device), 'rng', restored)
    with torch.inference_mode():
        for weights in ('raw', 'ema'):
            model.load_state_dict(payload[weights], strict=True)
            compare(payload[weights], cpu_tree(model.state_dict()), weights, restored)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda' and config['train']['amp']):
                prediction = forward_batch(model, batch, pixels).float().cpu().numpy()
                repeated = forward_batch(model, batch, pixels).float().cpu().numpy()
            if prediction.shape != (len(indices), 5, 7) or not np.isfinite(prediction).all():
                raise ValueError('Invalid checkpoint prediction')
            np.testing.assert_array_equal(prediction, repeated)
            physical = prediction.astype(np.float64)
            physical[..., :6] = normalize(physical[..., :6], stats['action_pose'], inverse=True)
            results[weights] = dict(shape=list(prediction.shape), finite=True, repeat_exact=True,
                normalized_abs_max=float(np.abs(prediction).max()),
                inverse_normalization_finite=bool(np.isfinite(physical).all()))
    if restored['mismatch_count']:
        raise ValueError(f'Checkpoint contents did not restore exactly: {restored}')
    dataset.close()
    result = dict(status='passed', checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=file_sha256(args.checkpoint), variant=config['variant'],
        checkpoint_step=payload['step'], smoke_checkpoint=payload['smoke_test'],
        observation_indices=indices, action_units=['m', 'm', 'm', 'rad', 'rad', 'rad', 'held_gripper'],
        model_config_source='checkpoint', normalization_source='checkpoint preprocessing bundle',
        robot_commands_sent=False, evaluation_success_measured=False,
        checkpoint_contents_restored_exactly=restored, results=results)
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}, indent=2))


if __name__ == '__main__':
    main()
