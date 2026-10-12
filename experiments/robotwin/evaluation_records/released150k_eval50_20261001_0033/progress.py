"""Read live progress without modifying evaluation files."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
plan = json.loads((root/'plan.json').read_text())
output = dict(status=plan['status'], started_at=plan.get('started_at'), stages=[])
for stage in plan['stages']:
    tasks = []
    for path in sorted(Path(stage['evaluation_dir']).glob('*_eval.log')):
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', path.read_text(errors='replace'))
        results = re.findall(r'Success rate:\s*(\d+)\s*/\s*(\d+)', text)
        successes, trials = map(int, results[-1]) if results else (0, 0)
        tasks.append(dict(task=path.name.split('_demo_')[0], successes=successes, trials=trials,
                          complete=trials == 50 and 'Data has been saved to' in text))
    output['stages'].append(dict(mode=stage['mode'], status=stage['status'],
        completed_tasks=sum(t['complete'] for t in tasks),
        reported_trials=sum(t['trials'] for t in tasks), target_trials=2500,
        reported_successes=sum(t['successes'] for t in tasks), tasks=tasks))
print(json.dumps(output, ensure_ascii=False, indent=2))
