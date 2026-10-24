"""Compare a recorded frame with configured preprocessing and a center crop.

Uses convert.image_sample, the same implementation that writes training HDF5.
No dataset conversion or configuration modification is performed.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from .convert import (StateDecoder, collection_config, crop_start, episodes,
                      image_sample, latest, read_settings, setting_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path, help='Trusted recorded pickle')
    parser.add_argument('--episode', type=int, default=0, help='Zero-based episode')
    parser.add_argument('--frame', type=int, help='Zero-based frame before stationary trimming; default: first retained')
    parser.add_argument('--center-crop-size', type=int, help='Square side in source pixels; default: largest possible square')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.episode < 0 or (args.frame is not None and args.frame < 0):
        parser.error('Episode and frame must be nonnegative')
    if args.center_crop_size is not None and args.center_crop_size <= 0:
        parser.error('Center crop size must be positive')
    cfg, config_path = read_settings(args.config)
    with args.input.open('rb') as stream:
        data = pickle.load(stream)
    selected = next((episode for index, episode in enumerate(episodes(data))
                     if index == args.episode), None)
    if selected is None:
        parser.error('Episode not found')
    rows, metadata, _ = selected
    explicit = cfg['input'].get('collection_config')
    collection, _ = collection_config(args.input.resolve(), metadata,
                                     setting_path(explicit, config_path) if explicit else None)
    start, trimming = crop_start(rows, StateDecoder(collection), cfg.get('crop_initial_stationary', {}))
    if start is None and args.frame is None:
        parser.error('Episode is dropped by stationary trimming; supply --frame to inspect it')
    frame_index = start if args.frame is None else args.frame
    if frame_index >= len(rows):
        parser.error(f'Frame must be below {len(rows)}')
    observation = rows[frame_index]['observations']
    entries = [entry for entry in cfg['observations'] if entry['type'] == 'image']
    if not entries:
        parser.error('Configuration has no image observations')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(config=str(config_path), source=str(args.input.resolve()),
                  episode=args.episode, source_frame=frame_index,
                  stationary_trimming=trimming, images=[])
    fig, axes = plt.subplots(len(entries), 3, figsize=(12, 4 * len(entries)), squeeze=False)
    for entry, row_axes in zip(entries, axes):
        source = entry['source']
        original = latest(observation[source], 3)
        if entry.get('input_color', 'rgb') == 'bgr':
            original = original[..., ::-1]
        height, width = original.shape[:2]
        size = args.center_crop_size or min(height, width)
        if size > min(height, width):
            parser.error(f'{source}: crop size {size} exceeds {height}x{width}')
        crop = [(height - size) // 2, (width - size) // 2, size, size]
        centered_entry = {**entry, 'crop': crop}
        configured = image_sample(observation[source], entry)
        centered = image_sample(observation[source], centered_entry)
        variants = [('original', original), ('configured', configured), ('center_crop', centered)]
        for axis, (name, pixels) in zip(row_axes, variants):
            Image.fromarray(pixels).save(args.out_dir / f'{source}_{name}.png')
            axis.imshow(pixels, interpolation='nearest')
            axis.set_title(f'{name.replace("_", " ")}\n{pixels.shape[1]} x {pixels.shape[0]}', fontsize=12)
            axis.set_xticks([])
            axis.set_yticks([])
        top, left, ch, cw = crop
        row_axes[0].add_patch(Rectangle((left - .5, top - .5), cw, ch, fill=False,
                                       edgecolor='#ffb000', linewidth=2))
        row_axes[0].set_ylabel(source, fontsize=11)
        row_axes[1].set_xlabel(f'YAML crop: {entry.get("crop")} | rotation: {entry.get("rotation_deg", 0)} deg')
        row_axes[2].set_xlabel(f'crop [top, left, height, width]: {crop}\nresize: {entry.get("resize")}')
        report['images'].append(dict(source=source, target=entry['target'],
            original_shape=list(original.shape), configured_crop=entry.get('crop'),
            rotation_deg=entry.get('rotation_deg', 0), rotation_expand=entry.get('rotation_expand', True),
            configured_shape=list(configured.shape),
            center_crop=crop, output_shape=list(centered.shape),
            configured_equals_original=bool(np.array_equal(configured, original)),
            center_crop_equals_original=bool(np.array_equal(centered, original))))
    fig.suptitle(f'{args.input.name}\nepisode {args.episode} | source frame {frame_index} | orange box = center crop', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(args.out_dir / 'comparison.png', dpi=150)
    plt.close(fig)
    (args.out_dir / 'metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Comparison: {(args.out_dir / "comparison.png").resolve()}')


if __name__ == '__main__':
    main()
