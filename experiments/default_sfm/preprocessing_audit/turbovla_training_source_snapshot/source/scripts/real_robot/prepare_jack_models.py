#!/usr/bin/env python3
"""Package Jack checkpoints and matching Kingston preprocessing; never start a server."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/jack_ablation_runs/jack_exp_demo_8way_20261020'
OUTPUT = ROOT / 'deploy/jack_models_20261020'
REMOTE = '/home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def all_model_commands(catalog):
    lines = [
        '\n전체 모델별 서버·클라이언트 명령어',
        '',
        '아래에서 평가할 모델 한 개의 명령어 쌍을 선택합니다. 서버가 Ready를 출력한 뒤 Kingston 클라이언트를 실행합니다.',
        '모델을 바꿀 때는 클라이언트에서 F12로 저장·종료하고 서버에서 Ctrl+C로 종료한 뒤 다음 명령어 쌍을 실행합니다.',
        '서버는 GPU 0 / RAW 가중치 / 포트 45587 기준입니다. --gpu 1 또는 --weights ema로 변경할 수 있습니다.',
        '각 클라이언트 명령은 모델에 맞는 Delta 또는 Relative 전처리 번들을 자동으로 선택합니다.',
        '',
    ]
    for index, (name, row) in enumerate(catalog['models'].items(), 1):
        lines.extend([
            f"{index}. {row['description']} (`{name}`), request_steps={row['request_steps']}",
            '',
            '학습 서버에서:',
            '',
            '```bash',
            f"cd {ROOT}",
            f"bash deploy/jack_models_20261020/serve_model.sh {name} --gpu 0 --weights raw --port 45587",
            '```',
            '',
            'Kingston에서:',
            '',
            '```bash',
            f"cd {catalog['client_install_path']}",
            f"bash run_client.sh {name} --trainer_ip {catalog['trainer_ip']} --trainer_port 45587 --request_steps {row['request_steps']}",
            '```',
            '',
        ])
    return '\n'.join(lines)


def replace_once(path, before, after):
    content = path.read_text()
    if content.count(before) != 1:
        raise ValueError('Unexpected source for isolated runtime patch: ' + str(path))
    path.write_text(content.replace(before, after, 1))


def upgrade_preprocessing_api(output):
    """Match Kingston's client API while retaining the trained image/state path."""
    inference = output / 'runtime/preprocess/inference.py'
    if '    def uses_opening_gripper(self):' not in inference.read_text():
        replace_once(inference, '    def denormalize_actions(self, values):', '''    @property
    def uses_opening_gripper(self):
        gripper = self.config['actions'].get('gripper')
        return isinstance(gripper, dict) and gripper.get('label_version') == 2

    def denormalize_actions(self, values):''')
    if '    def execution_config(self, config):' not in inference.read_text():
        replace_once(inference, '    def denormalize_actions(self, values):', '''    def execution_config(self, config):
        """Keep the packaged legacy gripper commands in their trained units."""
        if self.uses_opening_gripper:
            raise ValueError('This Jack runtime supports legacy gripper labels only; '
                             'use the current project preprocessor for label_version=2 bundles')
        return config

    def denormalize_actions(self, values):''')


def prepare_runtime(output):
    code = output / 'server_code'
    if code.exists():
        upgrade_preprocessing_api(output)
        shutil.copy2(ROOT / 'scripts/real_robot/check_jack_client.py', output / 'check_client.py')
        return
    for name in ['turbovla', 'scripts/real_robot']:
        shutil.copytree(RUN / 'code' / name, code / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # The FAFLOW training adapter preserved declared_formats in schema.action.
    # Ignore that descriptive field for schema validation, leaving all saved stats exact.
    rpc = code / 'turbovla/deployment/clean_share_rpc.py'
    replace_once(rpc, "        schema = self.stats.get('schema', {})\n", "        schema = self.stats.get('schema', {})\n        schema_action = dict(schema.get('action', {}))\n        schema_action.pop('declared_formats', None)\n")
    replace_once(rpc, "schema.get('action') != action", "schema_action != action")
    runtime = output / 'runtime'
    shutil.copytree(ROOT / 'deploy/go_models_20261019/runtime/preprocess', runtime / 'preprocess',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Vendor only the state decoder dependency, without shadowing Kingston's tools package.
    shutil.copy2('/home/jack/jack_exp_demo_preprocessing_20261019/runtime/tools/demo_viz/schema.py', runtime / 'preprocess/_schema.py')
    replace_once(runtime / 'preprocess/convert.py', 'from tools.demo_viz.schema import', 'from ._schema import')
    inference = runtime / 'preprocess/inference.py'
    replace_once(inference, 'from .convert import StateDecoder, transform_observation', 'from .convert import StateDecoder, transform_observation, latest')
    replace_once(inference, "        signature = self.stats.get('schema', {})\n", "        signature = self.stats.get('schema', {})\n        schema_action = dict(signature.get('action', {}))\n        schema_action.pop('declared_formats', None)\n")
    replace_once(inference, "signature.get('action') != action", "schema_action != action")
    replace_once(inference, '        result = transform_observation(observation, self.config[\'observations\'], decoder)', '''        if self.config.get('jack_source_image_resize'):
            import cv2
            height, width = self.config['jack_source_image_resize']
            observation = dict(observation)
            for entry in self.config['observations']:
                if entry['type'] == 'image':
                    frame = latest(observation[entry['source']], 3)
                    observation[entry['source']] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        result = transform_observation(observation, self.config['observations'], decoder)''')
    shutil.copy2(rpc, runtime / 'rpc_contract.py')
    for source, target in [('jack_serve_model.py', 'serve_model.py'), ('jack_run_client.py', 'run_client.py'),
                           ('verify_jack_bundles.py', 'verify_bundles.py'), ('check_jack_client.py','check_client.py')]:
        shutil.copy2(ROOT / 'scripts/real_robot' / source, output / target)
    for name in ['serve_model', 'run_client']:
        path = output / (name + '.sh')
        path.write_text('#!/usr/bin/env bash\nset -euo pipefail\nSCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
                        + f'exec python3 "$SCRIPT_DIR/{name}.py" "$@"\n')
        path.chmod(0o755)
    upgrade_preprocessing_api(output)


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    prepare_runtime(output)
    sys.path.insert(0, str(output / 'server_code'))
    from turbovla.data.real_robot import RealRobotDataset
    from turbovla.training.real_robot import read_checkpoint
    from turbovla.deployment import clean_share_rpc as rpc
    from turbovla.deployment.real_robot import validate_bundle
    catalog = dict(server_project_root=str(output/'server_code'),
                   server_python='/home/jack/miniforge3/envs/turbovla-libero/bin/python',
                   client_project_root='/home/csilab/share_projects/clean_share',
                   client_python='/home/csilab/miniforge3/envs/hilserl/bin/python', client_install_path=REMOTE,
                   trainer_ip='115.145.172.56', default_port=45587, default_weights='raw',
                   instruction='Insert the USB connector into the USB port.',
                   gpu_uuids={'0':'GPU-1bb2706c-5988-67f4-6ca0-1dd0e962f41e', '1':'GPU-7e7d7f97-c512-32b6-0a17-bfa6c8d19fe5'},
                   prepared_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), models={})
    old_catalog = json.loads((output/'models.json').read_text()) if (output/'models.json').exists() else {'models':{}}
    preferences_path=output/'client_preferences.json'
    preferences=json.loads(preferences_path.read_text()) if preferences_path.exists() else {}
    for job in json.loads((RUN/'schedule.json').read_text())['jobs']:
        model_id = 'jack_' + job['name']
        folder = Path(job['output'])
        status = json.loads((folder/'training_status.json').read_text()) if (folder/'training_status.json').exists() else {}
        ready = status.get('status') == 'completed' and status.get('samples_seen') == job['samples']
        checkpoint = folder / f"checkpoint_{status.get('step', 0):06d}.pt" if ready else None
        config = json.loads((folder/'recipe.json').read_text())
        if ready:
            print('Packaging completed model: '+model_id, flush=True)
            payload = read_checkpoint(checkpoint)
            assert payload['step'] == status['step'] and payload['samples_seen'] == job['samples']
            config, metadata = payload['config'], payload['dataset_metadata']
        else:
            dataset = RealRobotDataset({**config['data'], 'preload_images':False})
            metadata = dataset.metadata()
            dataset.close()
            payload = dict(config=config, dataset_metadata=metadata)
        stats = metadata['normalization']
        source = Path('/data/jack/jack_exp_demo_preprocessed') / job['mode']
        inference = json.loads((source/'inference_config.json').read_text())
        inference['observations'] = deepcopy(stats['schema']['observations'])
        inference['actions'] = deepcopy(metadata['action_contract'])
        if job['dataset'] == 'faflow':
            inference['jack_source_image_resize'] = [224,224]
        bundle = output/'models'/model_id/'bundles'/job['mode']
        write(bundle/'inference_config.json', inference)
        write(bundle/'normalization.json', stats)
        env = deepcopy(inference['collection_config'])
        env.pop('_source_path', None)
        env['image_preprocess'] = dict(enable=False, use_clahe=False, cameras={})
        env['train'].update(action_chunk_mode=inference['actions']['interpreter_mode'], policy_arm='right', policy_action_period_s=.1)
        env_file = output/'models'/model_id/'client_configs/ff_usb.json'
        write(env_file, env)
        write(bundle/'env_config.json', env)
        loaded = rpc.InferenceBundle(bundle)
        validate_bundle(payload, loaded)
        cached = old_catalog['models'].get(model_id, {})
        request_steps=preferences.get(model_id,{}).get('request_steps',cached.get('request_steps',4))
        if type(request_steps) is not int or request_steps<1:
            raise ValueError('request_steps must be a positive integer: '+model_id)
        checkpoint_sha = (cached.get('checkpoint_sha256') if cached.get('checkpoint') == str(checkpoint) else None)
        if checkpoint is not None and checkpoint_sha is None:
            checkpoint_sha = digest(checkpoint)
        row = dict(description=f"Jack {'기존' if job['dataset']=='original' else 'FAFLOW'} / {job['mode']} / "
                   + ('GRU 500Hz' if job['gru'] else 'FT history 미사용'),
                   ready=ready, training_status=status.get('status', 'queued'), step=status.get('step', 0),
                   target_steps=job['epochs']*((job['frames']+63)//64), checkpoint=str(checkpoint) if checkpoint else None,
                   checkpoint_sha256=checkpoint_sha, mode=job['mode'], dataset=job['dataset'],
                   weights_available=['raw','ema'] if ready else [], bundle=str(bundle.relative_to(output)),
                   bundle_id=rpc.bundle_id(loaded), client_config=str(env_file.relative_to(output)),
                   uses_ft_history=job['gru'], action_shape=stats['schema']['action_shape'],
                   request_steps=request_steps, source_dataset=config['data']['dataset_dir'])
        write(output/'models'/model_id/'manifest.json', row)
        catalog['models'][model_id] = row
        del payload
    write(output/'models.json', catalog)
    rows=['| 모델 ID | 상태 | step | H | request_steps |', '|---|---|---:|---:|---:|']
    for name, row in catalog['models'].items():
        rows.append(f"| `{name}` | {'학습 완료 / 사용 가능' if row['ready'] else row['training_status']} | {row['step']:,}/{row['target_steps']:,} | {row['action_shape'][0]} | {row['request_steps']} |")
    guide='\n'.join(rows)+f'''

기준 시각: {catalog['prepared_kst']}. RAW가 기본이며 --weights ema로 EMA를 선택할 수 있습니다.
학습 중이거나 대기 중인 모델은 번들만 준비되어 있습니다. 서버 실행기는 ready=false 모델의 실행을 거절합니다.
각 모델의 가중치는 학습 호스트에 보존합니다. Kingston에는 가중치를 복사하지 않습니다.

학습 호스트 115.145.172.56에서, 사용자가 서버를 시작할 때:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh --list
bash deploy/jack_models_20261020/serve_model.sh jack_faflow_delta_gru500hz --gpu 0 --weights raw --port 45587
```

위 명령은 물리 GPU 0을 지정합니다. --gpu 1로 GPU 1을 선택할 수도 있습니다.
Ready: http://0.0.0.0:45587 출력 후 Kingston에서 같은 모델 ID로 클라이언트를 실행합니다.

```bash
cd {REMOTE}
bash run_client.sh --list
bash run_client.sh jack_faflow_delta_gru500hz --trainer_port 45587
```

원래 요청한 모듈 형식으로 Kingston에서 실행할 경우:

```bash
export JACK_MODELS={REMOTE}
export PYTHONPATH="$JACK_MODELS/runtime:/home/csilab/share_projects/clean_share${{PYTHONPATH:+:$PYTHONPATH}}"
export PATH="/home/csilab/miniforge3/envs/hilserl/bin:$PATH"
cd /home/csilab/share_projects/clean_share/Real_Robo
python3 -m Real_Robo.env_rpc_client \\
  --trainer_ip 115.145.172.56 --trainer_port 45587 \\
  --env_path "$JACK_MODELS/models/jack_faflow_delta_gru500hz/client_configs" \\
  --env_name ff_usb \\
  --preprocess-bundle "$JACK_MODELS/models/jack_faflow_delta_gru500hz/bundles/delta" \\
  --request_steps 4 \\
  --prompt 'Insert the USB connector into the USB port.' \\
  --record-dir "$JACK_MODELS/rpc_sessions/jack_faflow_delta_gru500hz"
```

trainer_ip는 이 체크포인트가 있는 실제 호스트인 .56입니다. 예시의 .57과 다릅니다.
서버/클라이언트에서 같은 모델 ID와 포트를 선택해야 합니다. 모니터는 http://127.0.0.1:8125/ 입니다.
클라이언트 조작: 오른쪽 Ctrl=성공, F9=실패, F10=건너뛰기, F12=저장 후 종료입니다.
평가 기록은 rpc_sessions/<모델ID>에 저장됩니다. --episodes 10으로 에피소드 수를 제한할 수 있습니다.
환경 파일은 Jack 데이터의 수집 설정(오른팔, 카메라 serial/exposure, reset pose, spacemouse)을 기반으로 합니다.
공용 ff_usb.json 대신 모델 전용 ff_usb.json을 --env_path로 선택합니다.
spacemouse intervention은 기존 환경 설정을 따릅니다.
제어 액션은 10Hz입니다. request_steps=4는 약 0.4초 간격의 재추론 요청입니다.
8개 모델 모두 request_steps=4로 설정했습니다. run_client.sh는 표에 기재된 모델별 request_steps를 사용합니다.
필요하면 bash run_client.sh <모델ID> --request_steps 4처럼 명령행에서 바꿀 수 있습니다.
기존 H=50, FAFLOW H=5이며 반환되는 전체 action 중 실제 실행 범위는 클라이언트가 결정합니다.
Delta는 매 스텝 TCP 기준 증분, Relative는 예측 시점 TCP 기준 relative_tcp입니다.
두 형식 모두 held gripper -1=닫힘, +1=열림 라벨을 학습했습니다.
FT history 미사용 모델도 현재 state의 FT 채널을 사용합니다. GRU는 원본 raw FT [50,6]을 서버에서 정규화합니다.
FAFLOW 번들도 TurboVLA 어댑터의 오른팔 state19/action7과 raw FT를 사용해야 합니다.
따라서 FAFLOW 원본 state38/action14용 번들을 직접 지정하지 말고 위 모델별 번들을 사용합니다.
FAFLOW 카메라는 raw -> 224x224 AREA -> 640x480 LINEAR -> crop -> 224x168 AREA -> 상하 28 black padding을 적용합니다.
패키지의 runtime/preprocess가 그 과정을 수행합니다. 위 PYTHONPATH 또는 run_client.sh를 사용해야 합니다.
현재 state/actions의 저장 정규화를 반복 적용하지 않으며 서버 출력은 normalized action이고 클라이언트가 한 번 역변환합니다.

서버/로봇을 열지 않는 전처리 검증:

```bash
/home/csilab/miniforge3/envs/hilserl/bin/python {REMOTE}/check_client.py
```

이 준비 작업은 추론 서버나 로봇을 시작하지 않습니다. 실제 USB 삽입 성공률은 로봇에서 사용자가 측정합니다.
추가 모델의 학습이 끝나면 학습 호스트에서 아래 패키지 갱신 명령을 실행합니다.

```bash
USE_TF=0 /home/jack/miniforge3/envs/turbovla-libero/bin/python scripts/real_robot/prepare_jack_models.py
USE_TF=0 /home/jack/miniforge3/envs/turbovla-libero/bin/python scripts/real_robot/verify_jack_policies.py
/home/jack/miniforge3/envs/turbovla-libero/bin/python scripts/real_robot/install_jack_models.py --update
```
'''
    (output/'COMMANDS.md').write_text(guide + all_model_commands(catalog))
    print(json.dumps(dict(output=str(output), ready=sum(x['ready'] for x in catalog['models'].values()), total=8), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    prepare(parser.parse_args().output.resolve())
