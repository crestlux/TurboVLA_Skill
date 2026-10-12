"""Export the completed public 150k evaluation; never modify original run files.

Run with /home/jack/miniforge3/envs/turbovla-robotwin/bin/python (PyYAML).
This reads existing logs only; it does not load a model or run the simulator.
"""
import ast
import csv
import datetime
import json
import re
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUTPUT = REPO / 'experiments/robotwin/evaluation_records/released150k_20260929'
GUIDE = REPO / 'experiments/robotwin/EVALUATION_LOG_GUIDE_KO.md'
SIM = Path('/home/jack/practice/RoboTwin-TurboVLA-bf44be5')
STORE = Path('/data/jack/robotwin2_turbovla')
LABELS = {
    'adjust_bottle': '병을 세운 자세로 집어 올리기',
    'beat_block_hammer': '망치로 블록 두드리기',
    'blocks_ranking_rgb': '빨강·초록·파랑 블록을 왼쪽부터 순서대로 정렬하기',
    'blocks_ranking_size': '큰·중간·작은 블록을 왼쪽부터 순서대로 정렬하기',
    'click_alarmclock': '알람시계 윗면 버튼 누르기',
    'click_bell': '벨 윗부분 누르기',
    'dump_bin_bigbin': '작은 쓰레기통의 내용물을 큰 쓰레기통에 붓기',
    'grab_roller': '양팔로 롤러를 잡아 들기',
    'handover_block': '왼팔에서 오른팔로 블록을 넘겨 패드에 놓기',
    'handover_mic': '마이크를 한 팔에서 다른 팔로 넘기기',
    'hanging_mug': '머그컵을 거치대에 걸기',
    'lift_pot': '양팔로 냄비 손잡이를 잡아 들기',
    'move_can_pot': '캔을 냄비 옆으로 옮기기',
    'move_pillbottle_pad': '약병을 지정된 패드에 놓기',
    'move_playingcard_away': '카드 상자를 테이블 가장자리 쪽으로 옮기기',
    'move_stapler_pad': '스테이플러를 지정된 패드에 놓기',
    'open_laptop': '노트북 덮개 열기',
    'open_microwave': '전자레인지 문 열기',
    'pick_diverse_bottles': '서로 다른 종류의 병 2개를 들어 올리기',
    'pick_dual_bottles': '양팔로 병 2개를 들어 올리기',
    'place_a2b_left': '물체 A를 물체 B의 왼쪽에 놓기',
    'place_a2b_right': '물체 A를 물체 B의 오른쪽에 놓기',
    'place_bread_basket': '빵을 바구니에 담기',
    'place_bread_skillet': '빵을 스킬렛에 담아 들어 올리기',
    'place_burger_fries': '햄버거와 감자튀김을 쟁반에 놓기',
    'place_can_basket': '캔을 바구니에 담고 바구니 들어 올리기',
    'place_cans_plasticbox': '캔 2개를 플라스틱 상자에 담기',
    'place_container_plate': '용기를 접시 위에 놓기',
    'place_dual_shoes': '신발 2개를 방향을 맞춰 신발 상자 위에 놓기',
    'place_empty_cup': '빈 컵을 컵받침에 놓기',
    'place_fan': '선풍기를 로봇 쪽으로 향하게 패드에 놓기',
    'place_mouse_pad': '마우스를 지정된 패드에 놓기',
    'place_object_basket': '물체를 바구니에 담고 바구니 들어 올리기',
    'place_object_scale': '물체를 전자저울 위에 놓기',
    'place_object_stand': '물체를 받침대 위에 놓기',
    'place_phone_stand': '휴대전화를 거치대에 놓기',
    'place_shoe': '신발 1개를 방향을 맞춰 매트에 놓기',
    'press_stapler': '스테이플러 누르기',
    'put_bottles_dustbin': '병 3개를 쓰레기통에 넣기',
    'put_object_cabinet': '캐비닛 서랍을 열고 물체 넣기',
    'rotate_qrcode': 'QR 코드 표지판을 로봇 쪽으로 돌리기',
    'scan_object': '스캐너와 물체를 들어 스캔 위치에 맞추기',
    'shake_bottle_horizontally': '병을 집어 수평으로 흔들기',
    'shake_bottle': '병을 집어 흔들기',
    'stack_blocks_three': '블록 3개 쌓기',
    'stack_blocks_two': '블록 2개 쌓기',
    'stack_bowls_three': '그릇 3개 쌓기',
    'stack_bowls_two': '그릇 2개 쌓기',
    'stamp_seal': '지정된 패드에 도장 찍기',
    'turn_switch': '스위치 조작하기',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def existing(path):
    path = Path(path)
    assert path.exists(), str(path)
    return str(path.resolve())


def link(label, path):
    return f'[{label}](<{path}>)'


def score(successes, trials):
    return dict(successes=successes, failures=trials - successes,
                trials=trials, success_rate_pct=round(100 * successes / trials, 4))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def main():
    audit = read_json(OUTPUT / 'completion_audit.json')
    queue = read_json(HERE / 'benchmark_queue.json')
    validation = read_json(HERE / 'fixed150k_transformers4576.json')
    videos = {str(Path(v['path']).resolve()): v
              for v in read_json(OUTPUT / 'video_stream_audit.json')}
    assert audit['status'] == 'passed' and queue['stage'] == 'completed'
    task_ids = list(audit['results'][0]['tasks'])
    assert len(task_ids) == 50 and set(task_ids) == set(LABELS)
    step_limits = yaml.safe_load((SIM / 'task_config/_eval_step_limit.yml').read_text())
    generated_at = datetime.datetime.now().astimezone().isoformat()
    tasks, episodes, variants = {}, [], []
    for task in task_ids:
        source = SIM / f'envs/{task}.py'
        predicate = next(n for n in ast.walk(ast.parse(source.read_text()))
                         if isinstance(n, ast.FunctionDef) and n.name == 'check_success')
        tasks[task] = dict(task_id=task, description_ko=LABELS[task],
                           control_step_limit=int(step_limits[task]),
                           success_check=dict(source=existing(source), line=predicate.lineno),
                           instruction_templates=existing(SIM / f'description/task_instruction/{task}.json'),
                           results={})
    for result in audit['results']:
        mode, run = result['mode'], result['run_id']
        pipeline = STORE / 'pipelines' / run
        status = read_json(pipeline / 'status.json')
        assert status['stage'] == 'completed' and status['exit_code'] == 0
        assert status['checkpoint'] == audit['checkpoint']
        assert status['eval_trials_per_task'] == 10 and status['eval_seed'] == 0
        assert status['instruction_sample_num'] == 100
        root = Path(status['evaluation_dir'])
        with (root / 'task_results.csv').open() as handle:
            csv_rows = {r['task']: r for r in csv.DictReader(handle)}
        assert set(csv_rows) == set(result['tasks']) == set(task_ids)
        raw_summary = read_json(root / 'summary.json')
        dr = yaml.safe_load((SIM / f'task_config/{mode}.yml').read_text())['domain_randomization']
        variants.append(dict(
            mode=mode, label='Clean' if mode == 'demo_clean' else 'Randomized',
            run_id=run, status=status['stage'], exit_code=status['exit_code'],
            task_count=50, trials_per_task=10, **score(result['successes'], result['trials']),
            started_at=result['started_at'], completed_at=result['completed_at'],
            elapsed_seconds=result['elapsed_seconds'], perfect_task_count=result['perfect_tasks'],
            failed_episodes_at_step_limit=result['failures_at_step_limit'],
            caught_expert_setup_tracebacks=sum(x['count'] for x in result['caught_expert_tracebacks']),
            caught_tracebacks_note='Before policy trials; skipped scene validation attempts, not graded failures.',
            domain_randomization=dr,
            logs={
                'evaluation_directory': existing(root),
                'pipeline_log': existing(pipeline / 'pipeline.log'),
                'evaluation_log': existing(pipeline / 'evaluation.log'),
                'events_jsonl': existing(pipeline / 'events.jsonl'),
                'status_json': existing(pipeline / 'status.json'),
                'exit_code_txt': existing(pipeline / 'exit_code.txt'),
                'metadata_json': existing(pipeline / 'metadata.json'),
                'summary_json': existing(root / 'summary.json'),
                'task_results_csv': existing(root / 'task_results.csv'),
                'provenance_directory': existing(pipeline / 'provenance'),
            }))
        for task in task_ids:
            paths = list(root.glob(f'{task}_{mode}_*_eval.log'))
            assert len(paths) == 1, (task, mode, paths)
            log = paths[0]
            server = log.with_name(log.name.replace('_eval.log', '_server.log'))
            text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', log.read_text(errors='replace'))
            starts = list(re.finditer(r'^EVAL_EPISODE (.+)$', text, re.M))
            ends = list(re.finditer(r'Success rate:\s*(\d+)/(\d+)\s*=>[^\n]+current seed:\s*(\d+)', text))
            assert len(starts) == len(ends) == len(result['tasks'][task]) == 10
            result_path = SIM / re.search(r'Data has been saved to (.+)', text)[1]
            result_file = existing(result_path)
            eval_log, server_log = existing(log), existing(server)
            task_episodes, previous = [], 0
            for idx, (start, end, original) in enumerate(zip(starts, ends, result['tasks'][task])):
                assert start.end() < end.start()
                if idx < 9:
                    assert end.end() < starts[idx + 1].start()
                entry = json.loads(start[1])
                cumulative, count, seed = map(int, end.groups())
                assert count == idx + 1 and cumulative - previous in (0, 1)
                success = bool(cumulative - previous)
                previous = cumulative
                assert entry['task'] == task and seed == entry['seed'] == original['seed']
                assert success == original['success'] and entry['instruction'] == original['instruction']
                section = text[start.end():end.start()]
                assert 'Traceback (most recent call last):' not in section
                assert re.findall(r'^(Success!|Fail!)$', section, re.M) == ['Success!' if success else 'Fail!']
                steps = [(int(n), int(limit)) for n, limit in re.findall(r'step:\s*(\d+) / (\d+)', section)]
                assert steps and [s[0] for s in steps] == list(range(1, steps[-1][0] + 1))
                assert all(limit == step_limits[task] for _, limit in steps)
                if not success:
                    assert steps[-1][0] == step_limits[task]
                video_path = existing(result_path.parent / f'episode{idx}.mp4')
                stream = videos[video_path]
                episode = dict(
                    episode_id=f'{mode}/{task}/{idx}', mode=mode, task_id=task,
                    run_id=run, episode_index=idx, scene_seed=seed,
                    instruction=entry['instruction'], success=success,
                    outcome='success' if success else 'step_limit_reached',
                    control_steps=steps[-1][0], control_step_limit=int(step_limits[task]),
                    eval_log=eval_log, server_log=server_log, simulator_result_file=result_file,
                    video=dict(path=video_path, bytes=stream['bytes'], codec=stream['codec_name'],
                               width=stream['width'], height=stream['height'],
                               duration_seconds=float(stream['duration']), frames=int(stream['nb_frames'])))
                task_episodes.append(episode)
                episodes.append(episode)
            assert previous == int(csv_rows[task]['successes'])
            assert int(csv_rows[task]['trials']) == 10
            assert abs(float(csv_rows[task]['success_rate_pct']) - 10 * previous) < 1e-8
            assert float(result_path.read_text().strip().splitlines()[-1]) == previous / 10
            tasks[task]['results'][mode] = dict(
                **score(previous, 10), eval_log=eval_log, server_log=server_log,
                simulator_result_file=result_file, video_directory=existing(result_path.parent),
                scene_seeds=[e['scene_seed'] for e in task_episodes],
                failed_episode_indices=[e['episode_index'] for e in task_episodes if not e['success']],
                instruction_example=task_episodes[0]['instruction'])
        mode_eps = [e for e in episodes if e['mode'] == mode]
        assert len(mode_eps) == result['trials'] == raw_summary['trials'] == 500
        assert sum(e['success'] for e in mode_eps) == result['successes']
        assert abs(raw_summary['mean_success_pct'] - 100 * result['successes'] / 500) < 1e-8
    assert len(episodes) == len({e['episode_id'] for e in episodes}) == len(videos) == 1000
    assert {e['video']['path'] for e in episodes} == set(videos)
    assert sum(e['success'] for e in episodes) == audit['successes'] == 882
    for task in tasks.values():
        task['overall'] = score(sum(r['successes'] for r in task['results'].values()), 20)
    with (OUTPUT / 'task_comparison.csv').open() as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 50
    for row in comparisons:
        task = tasks[row['task']]
        assert task['results']['demo_clean']['successes'] == int(row['clean_successes'])
        assert task['results']['demo_randomized']['successes'] == int(row['randomized_successes'])
        assert task['overall']['successes'] == int(row['total_successes'])
    provenance = Path(variants[0]['logs']['provenance_directory'])
    deploy_path = provenance / 'source/experiments/robotwin/evaluation/deploy_policy.yml'
    deploy = yaml.safe_load(deploy_path.read_text())
    config = yaml.safe_load((provenance / 'config.yaml').read_text())
    assert deploy['action_ensemble'] is True and deploy['instruction_type'] == 'unseen'
    report = dict(
        schema_version=1, generated_at=generated_at, status='completed',
        description_ko='공개 TurboVLA 150k EMA: RoboTwin 2.0 전체 50개 태스크, Clean·Randomized 각 10회 평가',
        evaluation_started_at=queue['created_at'], evaluation_completed_at=queue['completed_at'],
        model=dict(name='TurboVLA public 150k EMA', weights='EMA', optimizer_steps=150000,
                   training_performed_in_this_run=False, huggingface_revision=audit['model_revision'],
                   checkpoint=existing(audit['checkpoint']), checkpoint_sha256=audit['checkpoint_sha256'],
                   config=existing(provenance / 'config.yaml'), config_sha256=validation['config_sha256'],
                   statistics=existing(provenance / 'dataset_statistics.json'),
                   statistics_sha256=validation['statistics_sha256'],
                   feature_source=validation['feature_source'], compatibility_profile=validation['compatibility_profile']),
        protocol=dict(
            type='closed_loop_simulator_rollout', simulator='RoboTwin 2.0', embodiment='aloha-agilex',
            simulator_path=existing(SIM), simulator_commit='bf44be51cf5717a5595ce59447f2cf5263d2aa95',
            policy_conda_env='/home/jack/miniforge3/envs/turbovla-robotwin',
            simulator_conda_env='/home/jack/miniforge3/envs/robotwin-turbovla',
            torch_version=validation['torch_version'], transformers_version=validation['transformers_version'],
            precision=validation['precision'], task_count=50, trials_per_task_per_variant=10,
            instruction_type='unseen', instruction_sample_num=100,
            evaluation_seed=0, initial_candidate_scene_seed=100000,
            scene_selection='Increment scene seed; accept only stable scenes with successful expert planning and check_success. Reset accepted scene before policy rollout.',
            instruction_selection='Sample from 100 generated unseen descriptions for the accepted scene; actual selected instruction is logged per episode.',
            language_reproducibility_note='Upstream Python random is not seeded; equal scene seeds do not guarantee equal instructions.',
            observations=dict(rgb_views=['head', 'left_wrist', 'right_wrist'],
                              image_size_hw=[224, 224], state_dim=14, language=True),
            action=dict(action_dim=14, meaning='12 absolute arm joint positions plus 2 gripper controls',
                        action_mode=deploy['action_mode'], normalization_mode=deploy['normalization_mode'],
                        unnorm_key=deploy['unnorm_key'], binary_threshold=deploy['binary_threshold'],
                        horizon=config['framework']['action']['horizon'],
                        temporal_ensemble=deploy['action_ensemble'],
                        ensemble_history=deploy['action_ensemble_horizon'],
                        adaptive_ensemble_alpha=deploy['adaptive_ensemble_alpha'],
                        policy_query_interval_control_steps=1, executed_actions_per_query=1,
                        ensemble_method='Weighted average of overlapping current-step predictions, using exp(alpha * cosine_similarity_to_latest).'),
            success_definition='Binary environment check_success/eval_success within the task control-step limit. Stop on success or step limit.',
            task_description_note='Korean descriptions summarize intended tasks; the simulator predicate, not a human visual rating, determines success.',
            predicate_caveat='shake_bottle and shake_bottle_horizontally check bottle height only, not completed shaking cycles.',
            metric='successes / graded_trials * 100; equal trial counts make macro and micro averages equal.',
            scheduling='2 RTX 3090 GPUs, one task per GPU; Clean then Randomized.',
            randomized_table_height_offset_m=[-0.03, 0.0],
            deploy_config=existing(deploy_path),
            scene_config_sources={v['mode']: existing(SIM / f"task_config/{v['mode']}.yml") for v in variants},
            episode_num_config_note='task_config episode_num=50 is a data-collection setting; ROBOTWIN_TEST_NUM=10 controlled this evaluation.',
            evaluation_driver=existing(provenance / 'simulator_eval_policy.py')),
        overall=dict(task_count=50, variant_count=2, task_variant_count=100,
                     **score(audit['successes'], audit['trials']),
                     macro_task_variant_success_rate_pct=88.2),
        variants=variants, tasks=list(tasks.values()),
        integrity=dict(
            original_audit_status=audit['status'], original_audit_checked_at=audit['checked_at'],
            export_consistency_checked_at=generated_at,
            checks='100 task logs, 1000 episode inputs/outcomes/step counts, CSV totals, simulator result files, and all recorded video paths agree.',
            graded_policy_tracebacks=0, failed_episodes_at_step_limit=118,
            videos_checked_in_original_audit=1000,
            video_check_scope='Existing ffprobe stream audit: positive frames and duration; not full-frame decoding. Export checks path existence without rerunning ffprobe.'),
        limitations=[
            '10 trials per task and variant, versus the official 100; this is a reduced evaluation.',
            'Public checkpoint inference evaluation; this does not reproduce the training process.',
            'Clean and Randomized are not guaranteed to use identical accepted seeds or language instructions.',
            'Log and video paths are absolute paths on this machine; JSON files do not embed videos.',
        ],
        official_comparison=read_json(OUTPUT / 'official_comparison.json'),
        files=dict(episode_results_json=str(OUTPUT / 'episode_results.json'), guide_markdown=str(GUIDE),
                   task_comparison_csv=existing(OUTPUT / 'task_comparison.csv'),
                   completion_audit_json=existing(OUTPUT / 'completion_audit.json'),
                   video_stream_audit_json=existing(OUTPUT / 'video_stream_audit.json'),
                   export_script=existing(Path(__file__))))
    episode_report = dict(
        schema_version=1, generated_at=generated_at, evaluation_completed_at=queue['completed_at'],
        summary_json=str(OUTPUT / 'evaluation_results.json'),
        overall=score(882, 1000),
        field_notes=dict(
            episode_index='Zero-based within each task and variant; maps directly to episode{index}.mp4.',
            scene_seed='Actual accepted scene seed, distinct from evaluation_seed=0 and episode_index.',
            control_steps='Count from the final step: N / LIMIT line within the graded policy rollout; not physics substeps.',
            outcome='success or step_limit_reached; scene/expert rejection attempts are not included.',
            video_duration_seconds='Encoded playback length at 10 fps; not rollout wall-clock duration.',
            paths='Absolute local paths. Original files are referenced, not copied.'),
        episodes=episodes)
    write_json(OUTPUT / 'evaluation_results.json', report)
    write_json(OUTPUT / 'episode_results.json', episode_report)
    write_guide(report, episodes)
    print(json.dumps(dict(status='passed', task_count=len(tasks), episodes=len(episodes),
                          successes=882, failures=118,
                          files=[str(OUTPUT / 'evaluation_results.json'),
                                 str(OUTPUT / 'episode_results.json'), str(GUIDE)]), ensure_ascii=False, indent=2))


def write_guide(report, episodes):
    lines = [
        '# 공개 150k EMA 평가 — 태스크·방식·로그·JSON 안내', '',
        f"정리 시각: {report['generated_at']}. 평가 종료: 2026-09-29 03:10 KST.", '',
        '**50개 태스크 × Clean/Randomized 2조건 × 조건별 10회 = 1,000회**, 882회 성공(88.2%).',
        '공개 150,000-step EMA 가중치를 사용했다. 이번 실행에는 추가 학습이 없다.', '',
        '| 조건 | 성공 / 시도 | 성공률 | 소요 시간 |',
        '|---|---:|---:|---|',
        '| Clean | 443 / 500 | 88.6% | 5시간 26분 33초 |',
        '| Randomized | 439 / 500 | 87.8% | 6시간 27분 24초 |',
        '| 합계 | 882 / 1,000 | 88.2% | 전체 큐 약 11시간 54분 |', '',
        '공식 평가의 태스크·조건별 100회보다 작은 **10회 평가**다. 공개 가중치의 평가 결과이며 학습 과정 재현을 뜻하지 않는다.', '',
        '## 바로 열어볼 파일', '',
        '- ' + link('evaluation_results.json — 전체·조건별·50개 태스크별 점수, 한글 설명, 설정, 로그 경로', OUTPUT / 'evaluation_results.json'),
        '- ' + link('episode_results.json — 1,000회 각각의 성공 여부, seed, 실제 지시문, 제어 step 수, 영상 경로', OUTPUT / 'episode_results.json'),
        '- ' + link('task_comparison.csv — 50개 태스크의 Clean/Randomized 점수 표', OUTPUT / 'task_comparison.csv'),
        '- ' + link('기존 완료 보고서 — 공식 보고값 비교와 호환성 검증', REPO / 'experiments/robotwin/EVALUATION_RELEASED150K_KO.md'), '',
        'JSON의 `success_rate_pct` 단위는 %다. `0.886`이 아닌 `88.6`으로 저장한다. 원본 `_result.txt`는 0~1 비율이다.',
        'JSON은 UTF-8이며 영상 자체를 포함하지 않는다. 파일 경로는 이 컴퓨터의 절대 경로다.', '',
        '## 평가 방식', '',
        '1. RoboTwin 2.0에서 ALOHA-AgileX 양팔 로봇과 물체 장면을 만든다. 안정성과 expert 실행 성공을 검증한 seed만 채택하고, 같은 seed로 장면을 다시 초기화한다.',
        '2. 해당 장면의 `unseen` 지시문을 생성·선택한다. 생성 개수 설정은 100이고 실제 정책 rollout은 태스크·조건별 10회다. 실제 선택된 문장은 회차별 JSON에 보존했다.',
        '3. 정책에 머리·왼쪽 손목·오른쪽 손목 RGB 영상(각 224×224), 영어 지시문, 14차원 로봇 상태를 넣는다.',
        '4. 매 제어 step마다 50-step 행동 묶음을 새로 예측한다. 현재 시점에 겹치는 예측을 adaptive temporal ensemble로 가중평균하고 행동 1개를 실행한 뒤 다시 관찰한다.',
        '5. 태스크의 `check_success()`를 충족하면 성공으로 종료하고, 제어 step 제한까지 충족하지 못하면 실패로 종료한다. 점수는 성공 횟수 / 채점한 rollout 수다.', '',
        '시뮬레이터에서 직접 행동하고 그 결과를 다시 관찰하는 폐루프 평가다. 모든 태스크의 시도 수가 같아 태스크 평균과 전체 성공률이 같다.', '',
        '| 설정 | Clean | Randomized |',
        '|---|---|---|',
        '| 배경 무작위화 | 꺼짐 | 켜짐, clean background 비율 설정 0.02 |',
        '| 주변 물체 추가 | 꺼짐 | 켜짐 |',
        '| 조명 무작위화 | 꺼짐 | 켜짐, 강한 조명 변화 비율 설정 0.02 |',
        '| 테이블 높이 무작위 오프셋 | 0 | −3 cm~0 cm |',
        '| 머리 카메라 위치 무작위화 | 0 | 0 |', '',
        '평가 seed는 0이며 후보 scene seed는 100000부터 증가한다. expert 검증에서 건너뛴 장면 때문에 실제 채택 seed에 공백이 있다. Clean과 Randomized는 채택 seed와 지시문이 완전히 같은 짝 비교가 아니다.',
        '기존 simulator가 Python `random`을 고정하지 않으므로 scene seed만으로 지시문까지 같아지지는 않는다.', '',
        f"사용 가중치: {link('150k EMA checkpoint', report['model']['checkpoint'])}.",
        f"배포 revision: `{report['model']['huggingface_revision']}`.",
        '정책 환경은 `turbovla-robotwin`(Torch 2.6.0+cu124, Transformers 4.57.6, BF16), 시뮬레이터 환경은 `robotwin-turbovla`다.',
        '`feature_source=last_hidden_state`, `compatibility_profile=robotwin`으로 공식 특징 출력을 맞췄다. 해당 배포의 정규화 통계를 사용했다.',
        '행동은 12개 팔 관절의 절대 위치 + 2개 그리퍼 제어다. Action horizon=50, ensemble history=50, alpha=0.1이다.', '',
        '## 로그 위치와 읽는 법', '',
        '| 조건 | 태스크 로그 폴더 | 실행 전체 로그 | 작업 실행·집계 로그 |',
        '|---|---|---|---|',
    ]
    for v in report['variants']:
        p = v['logs']
        lines.append(f"| {v['label']} | {link('폴더', p['evaluation_directory'])} | {link('pipeline.log', p['pipeline_log'])} | {link('evaluation.log', p['evaluation_log'])} |")
    lines += [
        '', '`*_eval.log`에는 환경 설정, `EVAL_EPISODE`의 seed·지시문, `step: 현재 / 제한`, `Success!`/`Fail!`, 누적 성공률이 있다.',
        '`*_server.log`에는 모델 서버의 체크포인트 로딩, feature/profile, 추론 관련 메시지가 있다. 작업마다 두 로그가 한 쌍이다.',
        '각 실행의 `status.json`과 `exit_code.txt`는 정상 종료를, `provenance/`는 당시의 설정·소스·패키지 목록과 해시를 보존한다. 정확한 경로는 통합 JSON의 `variants[].logs`에 있다.',
        '영상은 태스크·조건·실행별 폴더의 `episode0.mp4`~`episode9.mp4`다. 번호는 0부터 시작하며 scene seed와 다르다.',
        '영상은 머리 카메라 320×240, 10 fps로 저장됐다. 영상 재생 길이는 실제 평가 소요 시간이 아니다.', '',
        '## 50개 태스크와 결과', '',
        'Clean/Randomized 점수를 누르면 해당 태스크의 원본 평가 로그가 열린다. 태스크 ID를 누르면 성공 판정 코드를 볼 수 있다.',
        '한글 설명은 의도된 작업을 요약한다. 실제 점수는 코드 조건에 따른다. 예를 들어 `shake_bottle` 두 종류는 현재 simulator에서 병의 높이로 성공을 판정하며 흔들기 횟수까지 검증하지 않는다.',
        'step 제한은 정책 제어 횟수이며 physics 내부 step 수와 다르다.', '',
        '| # | 태스크 ID | 수행하는 작업 | 제한 step | Clean | Randomized |',
        '|---:|---|---|---:|---:|---:|',
    ]
    for index, t in enumerate(report['tasks'], 1):
        c, r = t['results']['demo_clean'], t['results']['demo_randomized']
        source = f"{t['success_check']['source']}:{t['success_check']['line']}"
        lines.append(f"| {index} | {link(t['task_id'], source)} | {t['description_ko']} | {t['control_step_limit']} | {link(str(c['successes']) + '/10', c['eval_log'])} | {link(str(r['successes']) + '/10', r['eval_log'])} |")
    failure = next(e for e in episodes if e['task_id'] == 'turn_switch' and not e['success'])
    lines += [
        '', '## JSON에서 실패 사례 찾기', '',
        '`evaluation_results.json`의 주요 필드는 `model`, `protocol`, `overall`, `variants`, `tasks`다.',
        '`tasks[].results.demo_clean` 및 `.demo_randomized`에서 점수와 원본 로그를 찾을 수 있다.',
        '`episode_results.json`의 `episodes[]`에서 `success=false`만 고르면 실패 118개를 볼 수 있다.', '',
        f"예: Clean `turn_switch`, episode_index={failure['episode_index']}, scene_seed={failure['scene_seed']}, {failure['control_steps']}/{failure['control_step_limit']} step 실패.",
        f"실제 지시문: “{failure['instruction']}”", '',
        link('이 회차의 영상', failure['video']['path']) + ' · ' + link('원본 평가 로그', failure['eval_log']), '',
        '다음 명령은 실패 회차와 영상 경로를 출력한다.', '',
        '```bash', "python3 - <<'PY'", 'import json',
        "from pathlib import Path",
        f"p = Path({str(OUTPUT / 'episode_results.json')!r})",
        "data = json.loads(p.read_text())",
        "for e in data['episodes']:",
        "    if not e['success']:",
        "        print(e['mode'], e['task_id'], e['episode_index'],",
        "              e['scene_seed'], e['control_steps'], e['video']['path'])",
        'PY', '```', '',
        '## 정상 완료 확인 범위', '',
        '두 실행 모두 `completed`, `exit_code=0`이다. 원본 로그 100개, 회차 1,000개의 입력·결과·step 수, CSV 및 simulator `_result.txt` 점수를 대조해 JSON을 생성했다.',
        '실패 118회는 모두 step 제한에 도달했다. 채점 중 traceback은 없었다. 채점 전 expert/장면 검증에서 잡힌 traceback은 조건별 16개로, 채점한 1,000회에 포함되지 않는다.',
        '기존 완료 감사에서 영상 1,000개의 ffprobe stream 정보·양수 frame 수·길이를 확인했다. 전체 프레임 디코딩 검사는 아니다. 이번 정리는 그 감사 결과와 실제 파일 경로를 대조했다.', '',
        link('원본 완료 감사 JSON', OUTPUT / 'completion_audit.json') + ' · ' +
        link('원본 영상 감사 JSON', OUTPUT / 'video_stream_audit.json') + ' · ' +
        link('JSON·안내문 생성 스크립트', HERE / 'export_evaluation_results.py'), '',
    ]
    GUIDE.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
