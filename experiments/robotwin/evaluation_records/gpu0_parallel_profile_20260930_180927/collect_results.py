"""Collect this bounded resource experiment, including paired rollout checks."""
import datetime
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
plan = json.loads((root / 'plan.json').read_text())
records = []
ansi = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
for stage in plan['stages']:
    directory = root / stage.get('profile_directory', stage['name'])
    resource = json.loads((directory / 'summary.json').read_text())
    assert resource['status'] == 'completed', directory
    run_id = resource['environment']['RUN_ID']
    pipeline = Path('/data/jack/robotwin2_turbovla/pipelines') / run_id
    metadata = json.loads((pipeline / 'metadata.json').read_text())
    status = json.loads((pipeline / 'status.json').read_text())
    assert status['stage'] == 'completed' and status['exit_code'] == 0
    logs = Path(metadata['evaluation_dir'])
    evaluation = json.loads((logs / 'summary.json').read_text())
    assert evaluation['trials'] == len(plan['tasks']) * metadata['eval_trials_per_task']
    episodes = []
    for task in plan['tasks']:
        matches = list(logs.glob(f'{task}_{stage["mode"]}_*_eval.log'))
        assert len(matches) == 1, matches
        text = ansi.sub('', matches[0].read_text(errors='replace'))
        starts = list(re.finditer(r'^EVAL_EPISODE (.+)$', text, re.MULTILINE))
        assert len(starts) == metadata['eval_trials_per_task']
        for index, match in enumerate(starts):
            episode = json.loads(match.group(1))
            tail = text[match.end(): starts[index+1].start() if index+1 < len(starts) else len(text)]
            steps = re.findall(r'step:\s*(\d+)\s*/\s*(\d+)', tail)
            assert steps and ('Success!' in tail or 'Fail!' in tail)
            episode.update(control_steps=int(steps[-1][0]), success='Success!' in tail)
            episodes.append(episode)
    records.append(dict(name=stage['name'], mode=stage['mode'], jobs_per_gpu=metadata['jobs_per_gpu'],
                        profile_directory=str(directory), pipeline_directory=str(pipeline),
                        evaluation_directory=str(logs), resource=resource, evaluation=evaluation,
                        episodes=episodes, episodes_per_hour=evaluation['trials']*3600/resource['wall_seconds']))
parallel = next(r for r in records if r['name'] == 'clean_j2')
single = next(r for r in records if r['name'] == 'clean_j1')
def keys(record):
    return sorted((e['task'], e['seed'], e['instruction']) for e in record['episodes'])
def outcomes(record):
    return sorted((e['task'], e['seed'], e['control_steps'], e['success']) for e in record['episodes'])
comparison = dict(same_scenes_and_instructions=keys(parallel)==keys(single),
                  same_success_and_control_steps=outcomes(parallel)==outcomes(single),
                  end_to_end_speedup=single['resource']['wall_seconds']/parallel['resource']['wall_seconds'],
                  wall_time_reduction_pct=100*(1-parallel['resource']['wall_seconds']/single['resource']['wall_seconds']))
assert comparison['same_scenes_and_instructions'], 'Unpaired scenes/instructions; cannot claim paired speedup'
completed_at = datetime.datetime.now().astimezone().isoformat()
plan.update(status='completed', completed_at=completed_at,
            completed_stages=[s['name'] for s in plan['stages']], current_stage=None)
result = dict(status='completed', generated_at=completed_at,
              plan=plan, stages=records, clean_comparison=comparison,
              limitations=['Only two tasks; not a 50-task capacity or success-rate estimate.',
                           'Wall time includes model/scene initialization, expert checks, video recording and cleanup.',
                           'One pass per setting; cache warmup and concurrent GPU 1 LIBERO work can affect timing.',
                           'Memory peaks are sampled about once per second; CPU and RSS refer to the evaluation process tree.',
                           'GPU 0 only; four simultaneous jobs across two GPUs have not been measured.'])
(root / 'profile_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
(root / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
for record in records:
    r=record['resource']
    print(record['name'], json.dumps(dict(seconds=r['wall_seconds'], peak_vram_mib=r['gpu'][0]['peak_used_mib'],
          cpu_mean_cores=r['process_cpu_mean_pct']/100, cpu_peak_cores=r['process_cpu_peak_pct']/100,
          rss_peak_mib=r['process_rss_peak_mib'], episodes=record['evaluation']['trials'],
          successes=sum(e['success'] for e in record['episodes'])), ensure_ascii=False))
print(json.dumps(comparison, indent=2))
print(root / 'profile_results.json')
