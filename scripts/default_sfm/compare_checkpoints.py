#!/usr/bin/env python3
"""Check uninterrupted versus resumed local smoke checkpoints, without training."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.training.sfm_utils import file_sha256, write_json


def compare(left, right, path, report):
    if isinstance(left, torch.Tensor):
        report['tensor_count'] += 1
        report['tensor_elements'] += left.numel()
        equal = isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        equal = np.array_equal(left, right)
    elif isinstance(left, dict):
        equal = left.keys() == right.keys()
        if equal:
            for key in left:
                compare(left[key], right[key], f'{path}/{key}', report)
            return
    elif isinstance(left, (tuple, list)):
        equal = type(left) is type(right) and len(left) == len(right)
        if equal:
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f'{path}/{index}', report)
            return
    else:
        equal = left == right
    if not equal:
        report['mismatch_count'] += 1
        max_difference = None
        if isinstance(left, torch.Tensor) and left.shape == right.shape:
            max_difference = float((left.double() - right.double()).abs().max())
            field = path.split('/')[0]
            report.setdefault('field_max_abs_difference', {})[field] = max(
                report.get('field_max_abs_difference', {}).get(field, 0.), max_difference)
        if len(report['first_mismatches']) < 20:
            detail = dict(path=path)
            if max_difference is not None:
                detail['max_abs_difference'] = max_difference
            report['first_mismatches'].append(detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uninterrupted', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    left = torch.load(args.uninterrupted, map_location='cpu', weights_only=False)
    right = torch.load(args.resumed, map_location='cpu', weights_only=False)
    if not left['smoke_test'] or not right['smoke_test']:
        raise ValueError('This verification tool is for disposable smoke checkpoints')
    fields = ['schema', 'config', 'data_signature', 'step', 'raw', 'ema', 'optimizer',
              'sampler', 'rng', 'source_sha256', 'initial_parameter_sha256']
    report = dict(tensor_count=0, tensor_elements=0, mismatch_count=0, first_mismatches=[])
    for field in fields:
        compare(left[field], right[field], field, report)
    result = dict(status='passed' if not report['mismatch_count'] else 'failed',
        comparison='bitwise exact for raw weights, EMA, AdamW, sampler and model/augmentation RNG',
        fields=fields, step=left['step'], samples_seen=left['sampler']['seen'],
        uninterrupted_sha256=file_sha256(args.uninterrupted), resumed_sha256=file_sha256(args.resumed),
        loader_generator_equal=torch.equal(left['loader_generator'], right['loader_generator']),
        loader_generator_note='A new DataLoader iterator consumes a worker base seed on resume. '
            'Workers only read deterministic HDF5; sampling and augmentation RNG are separate and checked.',
        formal_training_started=False, **report)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if report['mismatch_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
