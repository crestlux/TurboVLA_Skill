"""Compare unchanged RGB reads with bounded PyAV decoder threading (CPU only)."""
import argparse
import datetime
import gc
import json
import statistics
import time
from pathlib import Path

import av
import numpy as np

from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_indices, get_frames_by_timestamps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    roots = sorted(Path('/home/jack/datasets/RoboTwin/Clean').glob('*/videos'))
    paths = [next(path.glob('chunk-*/observation.images.*/episode_000000.mp4'))
             for path in roots[:12]]
    configs = {'auto': {}, 'one_thread_close': {'thread_count': 1, 'close_codec': True},
               'two_threads_close': {'thread_count': 2, 'close_codec': True}}
    timings = {name: [] for name in configs}
    sampled_threads = {name: [] for name in configs}
    for index, path in enumerate(paths):
        path.read_bytes()  # Warm the file cache equally before comparing decoders.
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            count = int(stream.frames)
        indices = [0, count // 2, count - 1]
        timestamps = np.asarray(indices) / fps
        expected = None
        ordered = list(configs.items())
        ordered = ordered[index % len(ordered):] + ordered[:index % len(ordered)]
        for name, kwargs in ordered:
            gc.collect()
            start = time.perf_counter()
            result = get_frames_by_timestamps(str(path), timestamps, 'pyav', kwargs)
            timings[name].append((time.perf_counter() - start) * 1000)
            sampled_threads[name].append(len(list(Path('/proc/self/task').iterdir())))
            if expected is None:
                expected = result
            else:
                assert np.array_equal(expected, result), (path, name)
            indexed = get_frames_by_indices(str(path), indices, 'pyav', kwargs)
            assert np.array_equal(expected, indexed), (path, name, 'indices')
    report = dict(status='passed',recorded_at=datetime.datetime.now().astimezone().isoformat(),
                  videos=len(paths), frames_per_read=3, equality='exact RGB equality at start/middle/end for timestamp and index reads',
                  scope='CPU decode only, warmed files and rotating variant order; not end-to-end training speed',
                  results={name: dict(median_ms=statistics.median(values), mean_ms=statistics.mean(values),
                                      max_threads_after_read=max(sampled_threads[name]))
                           for name,values in timings.items()})
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
