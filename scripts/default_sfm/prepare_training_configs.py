#!/usr/bin/env python3
"""Materialize the nine user-confirmed SFM recipes; does not start training."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.training.sfm_utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / 'experiments/default_sfm/training_configs')
    args = parser.parse_args()
    if args.microbatch not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("Microbatch must divide the confirmed effective batch 64")
    dataset_root = ROOT / 'data/default_sfm_preprocessed'
    manifest = json.loads((dataset_root / 'manifest.json').read_text())
    contract = json.loads((dataset_root / 'data_contract.json').read_text())
    reference = ROOT / 'experiments/default_sfm/preprocessing_audit/turbovla_training_source_snapshot/runs/jack/faflow_delta_gru500hz/config.json'
    original = json.loads(reference.read_text())
    model = copy.deepcopy(original['model'])
    model.pop('use_language')  # This implementation always uses the task instruction.
    model['text']['model_name_or_path'] = 'pretrained/backbones/bert-base-uncased'
    model['vision']['model_name_or_path'] = 'pretrained/backbones/dinov3-vitb16'
    model['action']['initialization_seed'] = 42
    model['ft_history'] = dict(enabled=False, history_length=50, input_dim=6, hidden_dim=128,
                               num_layers=1, input_normalization='pre_normalized', adaln=False,
                               adaln_initialization='zero')
    train = copy.deepcopy(original['train'])
    train.pop('batch_size')
    train.update(effective_batch_size=64, microbatch_size=args.microbatch,
                 max_samples=80 * manifest['frames'], equivalent_dataset_passes=80,
                 sampling='uniform_task_then_uniform_frame_with_replacement',
                 epsilon=1e-8, cpu_threads=4, num_workers=2)
    for variant, spec in contract['variants'].items():
        config = dict(schema='turbovla_sfm_training_v1', variant=variant,
            data=dict(root='data/default_sfm_preprocessed', expected_frames=138985, expected_episodes=450,
                      normalization_sha256=manifest['normalization_sha256'],
                      data_contract_sha256=manifest['data_contract_sha256']),
            model=copy.deepcopy(model), images=copy.deepcopy(original['images']), train=copy.deepcopy(train),
            wandb=dict(mode='online', project='TurboVLA_Skill', entity=None, group='default_sfm_9way_seed42'),
            initialization=dict(policy_checkpoint=None, pretrained_vision=True, pretrained_text=True,
                                common_parameter_initialization='shared seeded streams across all nine variants'),
            confirmation=dict(recipe='user confirmed existing USB settings', sampling='user confirmed task uniform',
                              effective_batch=64, adaln_initialization='zero', full_training_started=False))
        config['model']['action']['state_dim'] = spec['model_state_dim']
        config['model']['ft_history']['enabled'] = spec['ft_method'] != 'state_embedding'
        config['model']['ft_history']['adaln'] = spec['ft_method'] == 'gru_embedding_adaln'
        write_json(args.output / f'{variant}.json', config)
    write_json(args.output / 'index.json', dict(variants=sorted(contract['variants']), effective_batch_size=64,
        microbatch_size=args.microbatch, max_samples=train['max_samples'],
        optimizer_steps=(train['max_samples'] + 63) // 64, sampling=train['sampling'],
        note='Microbatch is a resource setting. Full training has not started.'))
    print(f"Prepared {len(contract['variants'])} configs in {args.output}; no training started")


if __name__ == '__main__':
    main()
