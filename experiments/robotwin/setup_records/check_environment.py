"""Validate installation and CPU model loading without training or a model forward pass."""
import importlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

report = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "training_executed": False,
    "model_forward_executed": False,
    "backward_executed": False,
    "optimizer_created": False,
    "versions": {},
    "imports": [],
}
base = Path('/home/jack/.cache/turbovla-robotwin-setup')
torch.set_num_threads(4)
for package in [
    'torch', 'torchvision', 'transformers', 'huggingface-hub', 'numpy', 'pandas',
    'accelerate', 'deepspeed', 'av', 'flash-attn', 'pyarrow', 'pydantic',
    'numpydantic', 'pytorch3d', 'omegaconf', 'tdigest', 'timm', 'wandb',
]:
    report['versions'][package] = importlib.metadata.version(package)

for module in [
    'accelerate', 'deepspeed', 'av', 'flash_attn', 'flash_attn_2_cuda',
    'pyarrow.parquet', 'pandas', 'pytorch3d.transforms', 'tdigest',
    'experiments.robotwin.train', 'starVLA.dataloader.lerobot_datasets',
    'starVLA.model.framework.VLM4A.TurboVLA',
]:
    importlib.import_module(module)
    report['imports'].append(module)
    print('IMPORT_OK', module, flush=True)

assert torch.cuda.is_available(), 'CUDA is not accessible'
report['torch_cuda_version'] = torch.version.cuda
report['gpus'] = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    report['gpus'].append({
        'index': index, 'name': properties.name,
        'memory_bytes': properties.total_memory,
        'capability': list(torch.cuda.get_device_capability(index)),
    })
assert len(report['gpus']) == 2, report['gpus']
print('GPU_IDENTIFICATION_OK', report['gpus'], flush=True)

from starVLA.dataloader.gr00t_lerobot.transform.state_action import RotationTransform
rotation = RotationTransform('axis_angle', 'rotation_6d')
vectors = torch.tensor([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]], dtype=torch.float32)
with torch.no_grad():
    recovered = rotation.inverse(rotation.forward(vectors))
assert torch.allclose(vectors, recovered, atol=1e-5)
report['rotation_transform_roundtrip'] = True

from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
from pyarrow import parquet
episode_root = Path(os.environ['ROBOTWIN_DATA_ROOT']) / 'Clean/adjust_bottle'
video = episode_root / 'videos/chunk-000/observation.images.cam_high/episode_000000.mp4'
frames = get_frames_by_timestamps(str(video), [0.0, 1 / 15, 2 / 15], video_backend='pyav')
assert frames.shape == (3, 480, 640, 3), frames.shape
table = parquet.read_table(episode_root / 'data/chunk-000/episode_000000.parquet')
report['sample_data_read'] = {'video_frames_shape': list(frames.shape), 'parquet_rows': table.num_rows}
print('SAMPLE_DATA_READ_OK', report['sample_data_read'], flush=True)

from omegaconf import OmegaConf
from safetensors import safe_open
from starVLA.model.framework.base_framework import build_framework

cfg = OmegaConf.load(Path(os.environ['TURBOVLA_ROOT']) / 'experiments/robotwin/configs/clean50.yaml')
model = build_framework(cfg)
assert all(p.device.type == 'cpu' for p in model.parameters())
vision = model.model.vision_encoder.backbone
assert vision.config.hidden_size == 1024
assert vision.config.num_hidden_layers == 24
weights = Path(os.environ['DINOV3_MODEL_PATH']) / 'model.safetensors'
state = vision.state_dict()
with safe_open(str(weights), framework='pt', device='cpu') as f:
    assert set(f.keys()) == set(state), 'DINO checkpoint keys differ from model keys'
    for name in f.keys():
        assert tuple(f.get_slice(name).get_shape()) == tuple(state[name].shape), name
    for name in [n for n in sorted(state) if n.endswith('weight')][:2]:
        assert torch.equal(state[name], f.get_tensor(name)), name
report['cpu_model_load'] = {
    'framework': type(model).__name__,
    'total_parameters': sum(p.numel() for p in model.parameters()),
    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'dino_layers': vision.config.num_hidden_layers,
    'dino_hidden_size': vision.config.hidden_size,
    'dino_state_tensors': len(state),
    'vision_attention_backend': vision.config._attn_implementation,
    'text_attention_backend': model.model.text_encoder.bert.config._attn_implementation,
    'groundingdino_initialization': 'loaded by framework constructor',
    'bert_frozen': all(not p.requires_grad for p in model.model.text_encoder.bert.parameters()),
}
report['status'] = 'passed_without_training'
(base / 'environment-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
