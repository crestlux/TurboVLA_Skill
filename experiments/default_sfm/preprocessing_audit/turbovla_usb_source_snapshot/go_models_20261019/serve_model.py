#!/usr/bin/env python3
"""Select a packaged checkpoint, GPU, weights, and port for the trainer host."""
import argparse
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    catalog = json.loads((root / 'models.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_id', nargs='?', choices=catalog['models'])
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--gpu', choices=('0', '1'), default='0')
    parser.add_argument('--weights', choices=('raw', 'ema'), default='raw')
    parser.add_argument('--port', type=int, default=catalog['default_port'])
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--precision', choices=('auto', 'float32', 'bfloat16'), default='bfloat16')
    args, extra = parser.parse_known_args()
    if args.list:
        for name, model in catalog['models'].items():
            print(f"{name:25s} step={model['step']:6d}  {model['description']}")
        return
    if not args.model_id:
        parser.error('Choose a model_id, or use --list')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in 1..65535')
    model = catalog['models'][args.model_id]
    if args.weights not in model['weights_available']:
        parser.error('These weights are absent from the packaged checkpoint')
    project = Path(catalog['server_project_root'])
    python = catalog['server_python']
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=catalog['gpu_uuids'][args.gpu],
                       USE_TF='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    command = [python, str(project / 'scripts/real_robot/serve.py'),
               '--checkpoint', model['checkpoint'],
               '--preprocess-bundle', str(root / model['bundle']),
               '--weights', args.weights, '--device', 'cuda', '--precision', args.precision,
               '--host', args.host, '--port', str(args.port), *extra]
    print(f"Starting {args.model_id}, step {model['step']}, {args.weights}, "
          f"GPU {args.gpu}, port {args.port}", flush=True)
    os.chdir(project)
    os.execve(python, command, environment)


if __name__ == '__main__':
    main()
