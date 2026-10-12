"""Print compact progress without changing the experiment."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
plan = json.loads((root/'plan.json').read_text())
result = dict(status=plan['status'], current_stage=plan.get('current_stage'),
              completed_stages=plan['completed_stages'])
if plan.get('current_stage'):
    stage = next(s for s in plan['stages'] if s['name']==plan['current_stage'])
    resource_path = root/stage['name']/'resources.jsonl'
    rows=[]
    if resource_path.exists():
        for line in resource_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A sample may be in the middle of its write.
    if rows:
        result.update(elapsed_seconds=round(rows[-1]['elapsed_seconds'],1),
                      latest_gpu=rows[-1]['gpu'],
                      peak_vram_gib=round(max(r['gpu'][0]['used_mib'] for r in rows)/1024,3),
                      cpu_mean_cores=round(sum(r['process_cpu_pct'] for r in rows[1:])/max(1,len(rows)-1)/100,2),
                      rss_peak_gib=round(max(r['process_rss_mib'] for r in rows)/1024,3))
    tasks=[]
    log_root=Path('/data/jack/robotwin2_turbovla/evaluation')/(stage['run_id']+'_seed0')
    for path in sorted(log_root.glob('*_eval.log')):
        text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',path.read_text(errors='replace'))
        rates=re.findall(r'Success rate:\s*(\d+)\s*/\s*(\d+)',text)
        steps=re.findall(r'step:\s*(\d+)\s*/\s*(\d+)',text)
        tasks.append(dict(task=path.name.split('_'+stage['mode'])[0],
                          completed_trials=int(rates[-1][1]) if rates else 0,
                          successes=int(rates[-1][0]) if rates else 0,
                          last_control_step=int(steps[-1][0]) if steps else None,
                          completed='Data has been saved to' in text))
    result['tasks']=tasks
    result['completed_trials']=sum(t['completed_trials'] for t in tasks)
    result['oom_logs']=[p.name for p in log_root.glob('*.log') if 'out of memory' in p.read_text(errors='replace').lower()]
print(json.dumps(result,ensure_ascii=False,indent=2))
