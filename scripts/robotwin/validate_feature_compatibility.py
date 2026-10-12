"""Compare complete release inference across dependency versions and official sources."""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_observations(path, data_root):
    import av
    import cv2
    import pyarrow.parquet as pq

    images, states, actions, languages, identities = [], [], [], [], []
    order = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    tasks = ['adjust_bottle', 'beat_block_hammer', 'click_bell', 'dump_bin_bigbin', 'grab_roller', 'lift_pot']
    for task in tasks:
        folder = data_root / task
        episode = json.loads((folder / 'meta/episodes.jsonl').read_text().splitlines()[0])
        table = pq.read_table(folder / 'data/chunk-000/episode_000000.parquet', columns=['observation.state', 'action'])
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)[:, order]
        action = np.asarray(table['action'].to_pylist(), dtype=np.float32)[:, order]
        indices = [max(1, episode['length'] // 4), episode['length'] // 2]
        frames = {index: [] for index in indices}
        for camera in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
            video = folder / f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4'
            with av.open(str(video)) as container:
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
            images.append(frames[index])
            states.append(state[index])
            actions.append(action[np.minimum(np.arange(index, index + 50), len(action) - 1)])
            languages.append(episode['tasks'][0])
            identities.append(f'{task}/episode_000000/frame_{index}')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=np.asarray(images), states=np.asarray(states), actions=np.asarray(actions),
             languages=np.asarray(languages), identities=np.asarray(identities))


def normalize(values, stats):
    low, high = np.asarray(stats['min']), np.asarray(stats['max'])
    result = np.array(values, dtype=np.float32, copy=True)
    assert np.all(high[:12] != low[:12])
    result[..., :12] = 2 * (result[..., :12] - low[:12]) / (high[:12] - low[:12]) - 1
    result[..., 12:] = (result[..., 12:] > 0.49).astype(np.float32)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/home/jack/datasets/RoboTwin/Clean'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-transformers', required=True)
    parser.add_argument('--official-source', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError(args.output)

    import torch
    import transformers
    from starVLA.model.framework.base_framework import baseframework

    assert transformers.__version__ == args.expected_transformers, transformers.__version__
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if not args.observations.exists():
        prepare_observations(args.observations, args.data_root)
    data = np.load(args.observations, allow_pickle=False)
    stats = json.loads((args.checkpoint.parent / 'dataset_statistics.json').read_text())['new_embodiment']
    model = baseframework.from_pretrained(str(args.checkpoint)).eval()
    official = None
    if args.official_source:
        package = types.ModuleType('official_feature_reference')
        package.__path__ = [str(args.official_source / 'turbovla')]
        sys.modules[package.__name__] = package
        module = importlib.import_module('official_feature_reference.models.turbovla')
        config_module = importlib.import_module('official_feature_reference.models.configuration')
        config = model.model.config.to_dict()
        allowed_config = {f.name for f in dataclasses.fields(config_module.TurboVLAConfig)}
        config = {key: value for key, value in config.items() if key in allowed_config}
        allowed_vision = {f.name for f in dataclasses.fields(config_module.VisionEncoderConfig)}
        config['vision'] = {key: value for key, value in config['vision'].items() if key in allowed_vision}
        official = module.TurboVLA(config_module.TurboVLAConfig.from_mapping(config)).eval()
        official.load_state_dict(model.model.state_dict(), strict=True)
        official.to(device='cuda', dtype=torch.bfloat16)
    model.to(device='cuda', dtype=torch.bfloat16)
    predictions, targets = [], []
    with torch.inference_mode():
        for image, state, action, language in zip(data['images'], data['states'], data['actions'], data['languages']):
            example = dict(image=list(image), state=normalize(state, stats['state'])[None], lang=str(language))
            instructions, samples, states = model._model_inputs([example])
            predicted = model.model(instructions, samples, states)
            assert predicted.shape == (1, 50, 14) and torch.isfinite(predicted).all()
            if official is not None:
                expected = official(instructions, samples, states)
                torch.testing.assert_close(predicted, expected, rtol=0, atol=0)
            predictions.append(predicted.float().cpu().numpy()[0])
            targets.append(normalize(action, stats['action']))
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    comparison = None
    if args.reference:
        reference_metadata = json.loads(args.reference.with_suffix('.json').read_text())
        assert reference_metadata['observations_sha256'] == digest(args.observations)
        reference = np.load(args.reference, allow_pickle=False)
        assert np.array_equal(reference['targets'], targets), 'Statistics or targets differ'
        delta = np.abs(predictions - reference['predictions'])
        comparison = dict(reference=str(args.reference), max_abs_difference=float(delta.max()),
                          bitwise_equal=bool(np.array_equal(predictions, reference['predictions'])))
        np.testing.assert_array_equal(predictions, reference['predictions'])
    report = dict(status='passed', checked_at=datetime.datetime.now().astimezone().isoformat(),
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
                  config_sha256=digest(args.checkpoint.parent / 'config.yaml'),
                  statistics_sha256=digest(args.checkpoint.parent / 'dataset_statistics.json'),
                  observations_sha256=digest(args.observations), transformers_version=transformers.__version__,
                  torch_version=torch.__version__, precision='bfloat16', samples=len(predictions),
                  feature_source=model.model.config.vision.feature_source,
                  compatibility_profile=model.model.config.compatibility_profile,
                  official_source=str(args.official_source) if official else None,
                  official_core_bitwise_equal=True if official else None,
                  normalized_action_mae=float(np.abs(predictions-targets).mean()), comparison=comparison,
                  limitation='12 in-distribution demonstration observations, not a rollout success estimate.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix('.npz'), predictions=predictions, targets=targets)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
