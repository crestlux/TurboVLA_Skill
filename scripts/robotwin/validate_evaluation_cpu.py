#!/usr/bin/env python3
"""Check installed evaluation software/assets without using a training GPU.

This does not certify GPU rendering or rollout execution. The full evaluation
launcher performs two real rollout trials after training releases the GPUs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['PYTHONNOUSERSITE'] = '1'

import argparse
import ast
import datetime
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-path', type=Path, required=True)
    parser.add_argument('--asset-ready', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(repo), str(repo / 'third_party/starvla_runtime'),
                    str(repo / 'experiments/robotwin/evaluation'), str(args.robotwin_path)]
    import torch
    import torchvision
    import numpy as np
    import sapien
    import mplib
    import open3d
    import toppra
    import model2robotwin_interface
    from pytorch3d.ops import sample_farthest_points

    assert not torch.cuda.is_available(), 'This validation must not use a GPU'
    assert torch.__version__ == '2.4.1+cu124', torch.__version__
    assert torchvision.__version__ == '0.19.1+cu124', torchvision.__version__
    assert np.__version__ == '1.26.4', np.__version__
    points = torch.arange(96, dtype=torch.float32).reshape(1, 32, 3)
    sampled, indices = sample_farthest_points(points, K=8)
    assert sampled.shape == (1, 8, 3) and torch.isfinite(sampled).all()
    extensions = ['lbfgs_step_cu', 'kinematics_fused_cu', 'line_search_cu', 'tensor_step_cu', 'geom_cu']
    for extension in extensions:
        importlib.import_module('curobo.curobolib.' + extension)
    # MotionGen creates CUDA tensors in default arguments during import. Its
    # high-level import and all task runtime imports belong in the GPU preflight.

    tree = ast.parse((repo / 'experiments/robotwin/data_registry/data_config.py').read_text())
    tasks = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_CLEAN50_TASKS' for t in node.targets))
    for task in tasks:
        task_file = args.robotwin_path / 'envs' / f'{task}.py'
        task_tree = ast.parse(task_file.read_text())
        assert any(isinstance(n, ast.ClassDef) and n.name == task for n in task_tree.body), task
        compile(task_tree, str(task_file), 'exec')
        assert (args.robotwin_path / 'description/task_instruction' / f'{task}.json').is_file(), task

    ready = json.loads(args.asset_ready.read_text())
    assert ready['status'] == 'ready' and ready['archives_sha256_verified'] and ready['zip_crc_verified']
    for folder in ['objects', 'embodiments', 'background_texture']:
        assert (args.robotwin_path / 'assets' / folder).resolve() == Path(ready['extracted_root']) / folder
    robot_assets = args.robotwin_path / 'assets/embodiments/aloha-agilex'
    assert (robot_assets / 'config.yml').is_file()
    for arm in ['left', 'right']:
        planner_config = robot_assets / f'curobo_{arm}.yml'
        assert planner_config.is_file(), planner_config
        assert '$ASSETS_PATH' not in planner_config.read_text()
        assert '${ASSETS_PATH}' not in planner_config.read_text()
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    record = dict(
        status='ready', validation_scope='installed_software_and_assets_cpu_only',
        gpu_rollout_validation='pending_after_training',
        post_training_preflight='evaluate.sh: adjust_bottle, two trials, 30 minute timeout',
        recorded_at=datetime.datetime.now().astimezone().isoformat(),
        robotwin_path=str(args.robotwin_path), robotwin_python=sys.executable,
        robotwin_commit=subprocess.check_output(['git', '-C', str(args.robotwin_path), 'rev-parse', 'HEAD'], text=True).strip(),
        task_modules_syntax_checked=len(tasks), task_runtime_imports='deferred_to_gpu_preflight',
        curobo_motion_gen_import='requires_cuda_tensors_at_import; deferred_to_gpu_preflight',
        pytorch3d_cpu_extension_test='passed',
        curobo_compiled_extensions_imported=extensions, pip_check=check.stdout.strip(),
        assets=ready,
        packages={name: importlib.metadata.version(name) for name in
                  ['torch', 'torchvision', 'numpy', 'sapien', 'mplib', 'open3d', 'pytorch3d',
                   'nvidia-curobo', 'warp-lang', 'websockets', 'omegaconf']},
    )
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
