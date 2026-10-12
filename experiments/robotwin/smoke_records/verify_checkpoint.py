"""Verify smoke-run artifacts and strictly reload the EMA for one inference call."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch

from starVLA.model.framework.base_framework import baseframework

torch.set_num_threads(4)
run = Path('/data/jack/robotwin2_turbovla/runs/turbovla_clean50_smoke_20260927_a')
assert (run / 'exit_code.txt').read_text().strip() == '0'
audits = [json.loads((run / f'smoke_audit_rank{rank}.json').read_text()) for rank in (0, 1)]
for audit in audits:
    assert audit['status'] == 'passed'
    assert audit['optimizer_updates'] == audit['scheduler_updates'] == audit['ema_updates'] == 20
    assert audit['micro_batches'] == 320 and audit['global_batch'] == 192
    assert audit['batch_eval_calls'] == 2 and audit['checkpoint_save_calls'] == 1

files = [run / 'checkpoints/steps_20_model.safetensors',
         run / 'checkpoints/steps_20_ema_model.safetensors',
         run / 'final_model/model.safetensors',
         run / 'final_model/ema_ema_model.safetensors']
records = []
for path in files:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            hasher.update(chunk)
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        keys = list(handle.keys())
        for key in keys:
            assert torch.isfinite(handle.get_tensor(key)).all(), (path, key)
    records.append({'path': str(path.relative_to(run)), 'bytes': path.stat().st_size,
                    'sha256': hasher.hexdigest(), 'tensors': len(keys)})
    print('CHECKPOINT_VALID', records[-1], flush=True)
assert records[0]['sha256'] == records[2]['sha256'], 'Raw final and step-20 weights differ'
assert records[1]['sha256'] == records[3]['sha256'], 'EMA final and step-20 weights differ'

sample_name = audits[0]['sample_parameter']
with safe_open(str(files[0]), framework='pt', device='cpu') as raw:
    with safe_open(str(files[1]), framework='pt', device='cpu') as ema:
        raw_ema_difference = float((raw.get_tensor(sample_name) - ema.get_tensor(sample_name)).abs().max())
assert raw_ema_difference > 0, 'EMA unexpectedly equals the raw action-head sample'

model = baseframework.from_pretrained(str(files[1]), allow_partial_load=False)
model = model.to(dtype=torch.bfloat16, device='cuda:1').eval()
example = {'image': [Image.new('RGB', (224, 224)) for _ in range(3)],
           'lang': 'Adjust the bottle.', 'state': np.zeros((1, 14), dtype=np.float32)}
with torch.inference_mode():
    actions = model.predict_action([example])['normalized_actions']
assert actions.shape == (1, 50, 14), actions.shape
assert np.isfinite(actions).all()
report = {'status': 'passed', 'run_dir': str(run), 'checkpoints': records,
          'ema_strict_load': True, 'raw_ema_parameter_max_difference': raw_ema_difference,
          'inference_input': 'synthetic three-view RGB and zero state; no simulator',
          'inference_device': 'cuda:1', 'inference_dtype': 'bfloat16',
          'action_shape': list(actions.shape), 'actions_finite': True,
          'action_min': float(actions.min()), 'action_max': float(actions.max())}
(run / 'checkpoint_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
