#!/usr/bin/env python3
"""Prepare eight Jack dataset/mode/FT-history runs without modifying source data."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.real_robot.enable_ft_history import enable_ft_history
from scripts.real_robot.launch_go import source_hashes
from turbovla.data.real_robot import RealRobotDataset, inverse_normalized_actions
from turbovla.evaluation.real_trajectories import read_geometry, score
from turbovla.training.real_robot import configure, write_json

ORIGINAL = Path('/data/jack/jack_exp_demo_preprocessed')
FAFLOW = Path('/data/jack/jack_exp_demo_faflow_preprocessed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt_faflow(run, mode):
    """Select active right arm; restore raw FT verified against original samples.

    Preserve native H=5, normalized state/actions, frames and original RGB pixels.
    Images use read-only external links. Geometry is added only for replay scoring.
    """
    source = FAFLOW / f'{mode}_500hz_held'
    target = run / 'prepared' / 'faflow' / mode
    target.mkdir(parents=True)
    stats = json.loads((source / 'normalization.json').read_text())
    original = ORIGINAL / mode
    normal = json.loads((original / 'normalization.json').read_text())
    source_manifest = json.loads((source / 'manifest.json').read_text())
    assert source_manifest['complete'] and len(source_manifest['shards']) == 30
    lookup = {}
    for path in sorted(original.glob('*.hdf5')):
        with h5py.File(path, 'r') as f:
            lookup[f.attrs['source_file']] = path
    normal['schema']['action_shape'] = [5, 7]
    normal['schema']['action']['length'] = 5
    normal['schema']['shapes'] = {k: v for k, v in normal['schema']['shapes'].items() if k != 'state_history'}
    normal['schema']['observations'] = [v for v in normal['schema']['observations'] if v['target'] != 'state_history']
    normal['observations']['state'] = deepcopy(stats['state'])
    normal['observations']['ft_history'] = {**deepcopy(stats['ft']), 'method': 'none', 'offset': [0.] * 6, 'scale': [1.] * 6}
    actions = deepcopy(stats['action_pose'][mode])
    for key, tail in [('offset', 0.), ('scale', 1.), ('min', -1.), ('max', 1.), ('mean', 0.), ('std', 1.)]:
        actions[key] = list(actions[key]) + [tail]
    normal['actions'] = actions
    normal['fit_scope'] = stats['fit_scope'] + '; adapter selects right arm, keeps native H=5; raw FT recovered from matching original history'
    entries, audit, grippers = [], [], []
    common_contract = None
    for entry in source_manifest['shards']:
        path = source / entry['file']
        output = target / (path.stem + '.hdf5')
        with h5py.File(path, 'r') as f, h5py.File(lookup[f.attrs['source_file']], 'r') as base:
            indices = f['source_indices'][:]
            base_indices = base['source_indices'][:]
            positions = np.searchsorted(base_indices, indices)
            assert np.array_equal(base_indices[positions], indices)
            length = len(indices)
            state, action, mask = f['observations/state'][:], f['actions'][:], f['action_mask'][:]
            assert state.shape == (length, 38) and action.shape == (length, 5, 14)
            assert np.all(state[:, :19] == 0) and np.all(action[..., :7] == 0)
            raw_ft = base['observations/ft_history'][:][positions]
            expected_ft = ((raw_ft.astype(np.float64) - np.array(stats['ft']['offset'])) / np.array(stats['ft']['scale'])).astype(np.float32)
            np.testing.assert_allclose(expected_ft, f['observations/ft_history'][:], atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(f['timestamps'][:], base['timestamps'][:][positions])
            contract = json.loads(f['metadata/action_contract'][()])
            contract['gripper'] = 'held desired state: -1 closed, +1 open; receipt target preferred'
            contract['gripper_relabel'] = {'method': 'latched_command', 'close_threshold': -.5, 'open_threshold': .5,
                                          'closed': -1, 'open': 1, 'temporal_shift': 0, 'source_labels': 'FAFLOW held receipt targets'}
            if common_contract is None:
                common_contract = contract
            assert contract == common_contract
            labels = f['gripper_labels'][:]
            np.testing.assert_array_equal(action[:, 0, 13], labels)
            grippers.append(action[..., 13][mask].astype(np.float64))
            with h5py.File(output, 'x') as out:
                out.attrs.update(schema_version=2, image_color='RGB', action_transform=mode,
                                 source_file=f.attrs['source_file'], num_steps=length)
                out.create_dataset('observations/state', data=state[:, 19:])
                out.create_dataset('observations/ft_history', data=raw_ft)
                out.create_dataset('actions', data=action[..., 7:])
                out.create_dataset('action_mask', data=mask)
                out.create_dataset('action_reference_poses', data=base['action_reference_poses'][:][positions])
                out.create_dataset('gripper_targets', data=labels)
                for key in ['source_actions', 'source_indices', 'timestamps', 'terminals']:
                    out[key] = h5py.ExternalLink(str(path), '/' + key)
                for key in ['image_head', 'image_wrist_right']:
                    out['observations/' + key] = h5py.ExternalLink(str(path), '/observations/' + key)
                out.create_dataset('metadata/action_contract', data=json.dumps(contract))
                out.create_dataset('metadata/adapter_provenance', data=json.dumps(dict(source=str(path), geometry_source=str(base.filename),
                                   state_slice=[19,38], action_slice=[7,14], ft='original raw samples aligned and verified against FAFLOW standard coefficients')))
            entries.append(dict(output=str(output), frames=length, skipped=False))
            audit.append(dict(source=str(path), source_bytes=path.stat().st_size, source_mtime_ns=path.stat().st_mtime_ns,
                              original=str(base.filename), frames=length, ft_standardization_verified=True,
                              state_action_right_slice_unchanged=True, images='external links to unchanged source datasets'))
    grip = np.concatenate(grippers)
    for key, value in [('mean', grip.mean()), ('std', grip.std()), ('min', grip.min()), ('max', grip.max())]:
        normal['actions'][key][-1] = float(value)
    normal['schema']['action'] = common_contract
    write_json(target / 'normalization.json', normal)
    write_json(target / 'manifest.json', entries)
    for path in target.glob('*.hdf5'):
        with h5py.File(path, 'r+') as f:
            f.create_dataset('metadata/normalization', data=json.dumps(normal))
    write_json(target / 'adapter_audit.json', dict(passed=True, source=str(source),
               source_metadata_sha256={name: sha(source/name) for name in ['manifest.json','normalization.json','preprocessing_contract.json']},
               native_horizon=5, frames=sum(v['frames'] for v in entries), episodes=audit))
    return target


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((ROOT / 'outputs/go_runs/go_turbovla_color_jitter_20261018_025920/delta/recipe.json').read_text())
    manifest = dict(run=str(run), created_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                    gpu_index=1, gpu_uuid='GPU-7e7d7f97-c512-32b6-0a17-bfa6c8d19fe5',
                    ft_source_hz=500, ft_samples=50, action_hz=10,
                    rate_basis='nominal collector rate; original 50 samples preserved without resampling',
                    training_basis='80 epochs per dataset; identical optimizer, seed 42, batch 64, language and image augmentation',
                    horizon_basis='native stored horizons: original=50, faflow=5; comparison includes this preprocessing difference',
                    jobs=[])
    for dataset_name in ['original', 'faflow']:
        for mode in ['delta', 'relative']:
            data_root = ORIGINAL / mode if dataset_name == 'original' else adapt_faflow(run, mode)
            raw_manifest = json.loads((data_root / 'manifest.json').read_text())
            frames = sum(v['frames'] for v in raw_manifest if not v.get('skipped'))
            horizon = 50 if dataset_name == 'original' else 5
            config = deepcopy(baseline)
            config['data'].update(dataset_dir=str(data_root), action_mode=mode, horizon=horizon,
                                  expected_episodes=30, expected_frames=frames, preload_images=False,
                                  gripper_relabel=None)
            config['model']['action']['horizon'] = horizon
            config['train'].update(max_samples=frames*80, keep_checkpoints=2)
            for gru in [False, True]:
                name = f'{dataset_name}_{mode}_' + ('gru500hz' if gru else 'no_ft_history')
                recipe = deepcopy(config)
                recipe['experiment'] = 'jack_exp_demo_turbovla_' + name
                recipe = enable_ft_history(recipe, source_hz=500, target_hz=500) if gru else configure(recipe)
                dataset = RealRobotDataset(recipe['data'])
                assert len(dataset) == frames and len(dataset.paths) == 30
                # Decode every stored target through the same replay geometry used after training.
                if not gru:
                    episodes = read_geometry(dataset.paths)
                    physical = np.concatenate([inverse_normalized_actions(torch.from_numpy(row['actions']), dataset.normalization['actions']).numpy()
                                               for row in dataset.numeric])
                    report = score(physical[None], episodes, dataset.action_contract, run / f'{dataset_name}_{mode}_target_check.npz', [0])
                    assert report['all_valid_chunk_entries']['position_mm']['max'] < .001, report['all_valid_chunk_entries']
                    assert report['all_valid_chunk_entries']['rotation_deg']['max'] < .001, report['all_valid_chunk_entries']
                    assert report['all_valid_chunk_entries']['gripper_absolute_error']['max'] < 1e-5
                    write_json(run / f'{dataset_name}_{mode}_target_check.json', dict(passed=True, report=report))
                out = run / name / mode
                out.mkdir(parents=True)
                write_json(out / 'recipe.json', recipe)
                write_json(out / 'job_status.json', dict(phase='queued', name=name))
                preflight = run / name / 'preflight'
                preflight.mkdir()
                write_json(preflight / f'config_{mode}.json', recipe)
                write_json(preflight / 'integrity_report.json', dict(passed=True, frames=frames, episodes=30,
                           target_geometry_verified=True, dataset_signature=dataset.signature()))
                manifest['jobs'].append(dict(name=name, mode=mode, dataset=dataset_name, gru=gru,
                                             output=str(out), stage=str(out.parent), phase='queued',
                                             frames=frames, epochs=80, samples=frames*80, horizon=horizon))
                dataset.close()
    code = run / 'code'
    for folder in ['turbovla', 'scripts/real_robot']:
        shutil.copytree(ROOT / folder, code / folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest['source_sha256'] = source_hashes(code)
    assert manifest['source_sha256'] == source_hashes(ROOT)
    manifest['tmux_session'] = 'jack-turbovla-gpu1-' + run.name.split('_')[-1]
    write_json(run / 'schedule.json', manifest)
    exports = dict(CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'], USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(code))
    script = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join(f'export {k}={shlex.quote(v)}' for k,v in exports.items()) + '\n'
    script += f'cd {shlex.quote(str(code))}\nexec ' + shlex.join([sys.executable, 'scripts/real_robot/queue_jack_ablation.py', '--run', str(run)])
    script += f' >> {shlex.quote(str(run / "queue.log"))} 2>&1\n'
    (run / 'start_queue.sh').write_text(script)
    write_json(run / 'preparation_status.json', dict(passed=True, jobs=len(manifest['jobs']), phase='ready'))
    print(json.dumps(dict(run=str(run), jobs=len(manifest['jobs']), tmux_session=manifest['tmux_session'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    prepare(parser.parse_args().run.resolve())
