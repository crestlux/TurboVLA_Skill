"""Check identical CPU/CUDA masks and time only the token-mask preparation."""
import argparse
import datetime
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from turbovla.text.bert import generate_masks_with_special_tokens
from turbovla.models.text_encoder import TurboVLATextEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.03, 0)
    tokenizer = AutoTokenizer.from_pretrained('pretrained/backbones/bert-base-uncased', local_files_only=True)
    special = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '.', '?'])
    encoder = SimpleNamespace(tokenizer=tokenizer, special_tokens=special,
                              config=SimpleNamespace(max_length=256))
    tasks = sorted(Path('/home/jack/datasets/RoboTwin/Clean').glob('*/meta/tasks.jsonl'))
    captions = [json.loads(path.read_text().splitlines()[0])['task'] for path in tasks]
    cases = [captions[:1], captions[:12], captions[12:28],
             ['', '.', '?', 'pick a bottle. place it?', 'move ' * 300]]

    def prep(instructions, on_cpu):
        if on_cpu:
            return TurboVLATextEncoder._tokenize_group(encoder, instructions, torch.device('cuda'), None)
        tokens = tokenizer(instructions, padding='longest', truncation=True,
                           max_length=256, return_tensors='pt')
        if not on_cpu:
            tokens = tokens.to('cuda')
        masks, positions = generate_masks_with_special_tokens(tokens, special, tokenizer)
        return tokens.to('cuda'), masks.to('cuda'), positions.to('cuda')

    for instructions in cases:
        old = prep(instructions, False)
        new = prep(instructions, True)
        assert all(torch.equal(old[0][key], new[0][key]) for key in old[0])
        assert torch.equal(old[1], new[1]) and torch.equal(old[2], new[2])
    measurements = {}
    for batch in ([] if args.verify_only else [1, 12]):
        times = {'cuda_scalar_masks': [], 'cpu_masks_then_transfer': []}
        for _ in range(3):
            prep(captions[:batch], False)
            prep(captions[:batch], True)
        for i in range(30):
            # Alternate ordering; the active training job is a timing confounder.
            for on_cpu in ([False, True] if i % 2 == 0 else [True, False]):
                torch.cuda.synchronize()
                start = time.perf_counter()
                prep(captions[:batch], on_cpu)
                torch.cuda.synchronize()
                times['cpu_masks_then_transfer' if on_cpu else 'cuda_scalar_masks'].append(
                    (time.perf_counter() - start) * 1000)
        measurements[str(batch)] = {key: dict(median_ms=statistics.median(value),
                                               mean_ms=statistics.mean(value))
                                    for key, value in times.items()}
    report = dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),
                  status='passed', equality_cases=len(cases), bitwise_equal=True,
                  scope='tokenization and masks only; concurrent training prevents standalone throughput claims',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  measurements=measurements)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
