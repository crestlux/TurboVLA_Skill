"""Collect paired 2/3-worker resource measurements with all rollout evidence."""
import datetime
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
plan = json.loads((root/'plan.json').read_text())
records = []
ansi = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
for stage in plan['stages']:
    directory = root/stage['name']
    resource = json.loads((directory/'summary.json').read_text())
    assert resource['status'] == 'completed', directory
    pipeline = Path('/data/jack/robotwin2_turbovla/pipelines')/stage['run_id']
    metadata = json.loads((pipeline/'metadata.json').read_text())
    status = json.loads((pipeline/'status.json').read_text())
    assert status['stage'] == 'completed' and status['exit_code'] == 0
    logs = Path(metadata['evaluation_dir'])
    evaluation = json.loads((logs/'summary.json').read_text())
    assert evaluation['trials'] == len(plan['tasks'])*metadata['eval_trials_per_task']
    episodes = []
    for task in plan['tasks']:
        paths = list(logs.glob(f'{task}_{stage["mode"]}_*_eval.log'))
        assert len(paths) == 1, paths
        text = ansi.sub('', paths[0].read_text(errors='replace'))
        starts = list(re.finditer(r'^EVAL_EPISODE (.+)$', text, re.MULTILINE))
        assert len(starts) == metadata['eval_trials_per_task']
        for index, match in enumerate(starts):
            episode = json.loads(match.group(1))
            tail = text[match.end():starts[index+1].start() if index+1 < len(starts) else len(text)]
            steps = re.findall(r'step:\s*(\d+)\s*/\s*(\d+)', tail)
            assert steps and ('Success!' in tail or 'Fail!' in tail)
            episode.update(control_steps=int(steps[-1][0]), success='Success!' in tail)
            episodes.append(episode)
    records.append(dict(name=stage['name'], mode=stage['mode'], jobs_per_gpu=metadata['jobs_per_gpu'],
                        profile_directory=str(directory), pipeline_directory=str(pipeline),
                        evaluation_directory=str(logs), resource=resource, evaluation=evaluation,
                        episodes=episodes, total_control_steps=sum(e['control_steps'] for e in episodes),
                        episodes_per_hour=evaluation['trials']*3600/resource['wall_seconds']))


def scene_keys(record):
    return sorted((e['task'], e['seed'], e['instruction']) for e in record['episodes'])


def outcomes(record):
    return sorted((e['task'], e['seed'], e['control_steps'], e['success']) for e in record['episodes'])


comparisons = []
source_signatures = []
for record in records:
    hashes = json.loads((Path(record['pipeline_directory'])/'provenance/SHA256.json').read_text())
    source_signatures.append({key: value for key, value in hashes.items()
                              if key.startswith('source/') or key == 'simulator_eval_policy.py'})
same_source_code = all(signature == source_signatures[0] for signature in source_signatures)
for mode in ['demo_clean', 'demo_randomized']:
    two = next(r for r in records if r['mode'] == mode and r['jobs_per_gpu'] == 2)
    three = next(r for r in records if r['mode'] == mode and r['jobs_per_gpu'] == 3)
    comparisons.append(dict(mode=mode, same_scenes_and_instructions=scene_keys(two)==scene_keys(three),
                            same_success_and_control_steps=outcomes(two)==outcomes(three),
                            two_worker_seconds=two['resource']['wall_seconds'],
                            three_worker_seconds=three['resource']['wall_seconds'],
                            end_to_end_speedup=two['resource']['wall_seconds']/three['resource']['wall_seconds'],
                            wall_time_reduction_pct=100*(1-three['resource']['wall_seconds']/two['resource']['wall_seconds'])))
two_seconds = sum(r['resource']['wall_seconds'] for r in records if r['jobs_per_gpu']==2)
three_seconds = sum(r['resource']['wall_seconds'] for r in records if r['jobs_per_gpu']==3)
paired = all(c['same_scenes_and_instructions'] for c in comparisons)
generated_at = datetime.datetime.now().astimezone().isoformat()
plan.update(status='completed', completed_at=generated_at, current_stage=None,
            completed_stages=[s['name'] for s in plan['stages']])
result = dict(status='completed', generated_at=generated_at, plan=plan, stages=records, comparisons=comparisons,
              same_model_and_evaluation_source_code=same_source_code,
              combined=dict(paired_scenes_and_instructions=paired, two_worker_seconds=two_seconds,
                            three_worker_seconds=three_seconds, end_to_end_speedup=two_seconds/three_seconds,
                            wall_time_reduction_pct=100*(1-three_seconds/two_seconds)),
              limitations=['Six tasks, two episodes each; one run per setting. Not a 50-task worst-case memory bound.',
                           'All time includes startup, expert scene validation, video recording and cleanup.',
                           'Same task order; task durations can differ, so scheduling tail effects remain.',
                           'GPU 0 only; GPU 1 LIBERO continues and shares host CPU/RAM.',
                           'Memory peaks sampled at about 1 second. Process CPU 100% means one logical core; summed RSS can double-count shared pages.',
                           'Full-dataset or two-GPU timings have not been measured in this experiment.'])
(root/'profile_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
(root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
for record in records:
    r=record['resource']
    print(record['name'], json.dumps(dict(seconds=r['wall_seconds'], peak_vram_mib=r['gpu'][0]['peak_used_mib'],
          cpu_mean_cores=r['process_cpu_mean_pct']/100, cpu_peak_cores=r['process_cpu_peak_pct']/100,
          rss_peak_mib=r['process_rss_peak_mib'], trials=record['evaluation']['trials'],
          successes=sum(e['success'] for e in record['episodes'])), ensure_ascii=False))
print(json.dumps(comparisons, indent=2))
print(json.dumps(result['combined'], indent=2))
print(root/'profile_results.json')
