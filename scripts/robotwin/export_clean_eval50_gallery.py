"""Export and validate the completed 2,500-video Clean evaluation."""
import concurrent.futures
import datetime
import json
import re
import shutil
from pathlib import Path

from export_pilot5k_rollout_gallery import (
    CONTROLS, CONTROL_SCRIPT, RECORDS, card, esc, page, prepare_video,
    read_json, sha256, write_json,
)

BASE = Path('/data2/jack/TurboVLA/rollout_videos')
OUTPUT = BASE / 'robotwin_public150k_clean_eval50_20261001'
SOURCE = RECORDS / 'released150k_eval50_randomized_retry_20261002_1647/preserved_clean_result.json'
RUN = 'turbovla_release150k_eval50_demo_clean_20261001_0033'
SIM = Path('/data/jack/robotwin2_turbovla/simulator_eval_result')


def video_card(episode, prefix=''):
    return card(episode, prefix).replace('</article>',
        '<details><summary>실제 평가 지시문</summary><p lang="en">'
        + esc(episode['instruction']) + '</p></details></article>')


def build_pages(output, manifest):
    tasks = sorted(manifest['tasks'], key=lambda t: (t['successes'], t['task']))
    links = ''.join(f'<a class="task-link" data-task href="{t["folder"]}/index.html">'
        f'<span class="score">{t["successes"]}/50 · {t["success_rate_pct"]:.0f}%</span>'
        f'{esc(t["description_ko"])}<small>{esc(t["task"])}</small></a>' for t in tasks)
    body = f'''<main class="page"><a href="../index.html">← 모든 평가</a>
<h1>공개 TurboVLA 150K EMA · Clean 50회 평가</h1>
<p class="muted">2026-10-01 평가 · 50개 태스크 × 50회 · 원본 머리 카메라 영상</p>
<div class="stats"><span>2,500개 영상</span><span>성공 2,201 · 실패 299</span><span>성공률 88.04%</span></div>
<p class="links"><a href="preview.html">실패·성공 사례 빠른 검토 →</a><a href="manifest.json">전체 영상·결과 JSON</a></p>
<p>태스크를 선택하면 50개 영상을 볼 수 있습니다. 성공 수가 적은 태스크부터 표시합니다.</p>
<p class="note">영상에는 머리 카메라만 녹화돼 있습니다. 모델은 머리·양쪽 손목 카메라를 사용합니다.
영상은 10fps로 저장되며 재생 길이는 실제 평가 소요 시간과 다릅니다. 이번 갤러리는 Clean 평가만 포함합니다.</p>
<input id="search" type="search" aria-label="태스크 검색" placeholder="태스크 이름 또는 작업 검색">
<nav class="grid">{links}</nav></main>'''
    script = '''document.getElementById('search').addEventListener('input', event => {
const q = event.target.value.trim().toLowerCase();
document.querySelectorAll('[data-task]').forEach(a => a.hidden = !a.textContent.toLowerCase().includes(q));
});'''
    (output/'index.html').write_text(page('공개 150K · Clean 2,500개 롤아웃', body, script=script))
    for task in tasks:
        body = f'''<main class="page"><a href="../index.html">← 전체 태스크</a>
<h1>{esc(task['description_ko'])}</h1><p>{esc(task['task'])} · Clean · 성공 {task['successes']}/50</p>
<p class="links"><a href="result.json">결과 JSON</a><a href="eval.log">평가 로그</a><a href="server.log">모델 서버 로그</a></p>
{CONTROLS}<div class="grid">{''.join(video_card(e, '../') for e in task['episodes'])}</div>
<p class="meta">EP는 0부터 시작합니다. 성공·실패는 시뮬레이터 판정입니다.</p></main>'''
        (output/task['folder']/'index.html').write_text(page(task['description_ko'], body, '../', CONTROL_SCRIPT))
    previews, selected = [], []
    for task in tasks[:10]:
        episodes = sorted([e for e in task['episodes'] if not e['success']][:2]
                          + [e for e in task['episodes'] if e['success']][:1], key=lambda e: e['episode_index'])
        selected.extend(dict(task=task['task'], video=e['video'], success=e['success']) for e in episodes)
        previews.append(f'<section class="task-section"><h2>{esc(task["description_ko"])}</h2>'
            f'<p>{esc(task["task"])} · 성공 {task["successes"]}/50 · '
            f'<a href="{task["folder"]}/index.html">50개 전체 보기 →</a></p>'
            f'<div class="grid">{"".join(video_card(e) for e in episodes)}</div></section>')
    body = '<main class="page"><a href="index.html">← 전체 2,500개 영상</a><h1>Clean · 실패와 성공 빠른 검토</h1>'
    body += '<p>성공률이 낮은 10개 태스크에서 EP 순서로 실패 최대 2개와 성공 최대 1개를 골랐습니다. 실패 원인을 분류한 것은 아닙니다.</p>'
    body += CONTROLS + ''.join(previews) + '</main>'
    (output/'preview.html').write_text(page('Clean 50회 평가 · 빠른 검토', body, script=CONTROL_SCRIPT))
    manifest['quick_preview'] = selected


def main():
    reference = read_json(SOURCE)
    assert reference['trials'] == 2500 and reference['successes'] == 2201
    assert len(reference['tasks']) == 50
    assert not OUTPUT.exists()
    output = OUTPUT.with_name(OUTPUT.name + '.building')
    output.mkdir(exist_ok=False)
    shutil.copy2(BASE/'robotwin_public150k_ema_20260929/gallery.css', output/'gallery.css')
    tasks, jobs, hashes = [], [], {}
    labels = {t['task_id']: t['description_ko'] for t in read_json(RECORDS/'released150k_20260929/evaluation_results.json')['tasks']}
    for index, original in enumerate(reference['tasks']):
        name = original['task']
        paths = list(Path(reference['evaluation_dir']).glob(name+'_demo_clean_*_eval.log'))
        assert len(paths) == 1
        log = paths[0]
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', log.read_text(errors='replace'))
        assert 'Data has been saved to' in text
        assert not re.search(r'OIDN Error:|cannot create buffer|out of memory|OutOfMemoryError', text, re.I)
        starts = list(re.finditer(r'^EVAL_EPISODE (.+)$', text, re.M))
        ends = list(re.finditer(r'Success rate:\s*(\d+)/(\d+)\s*=>[^\n]+current seed:\s*(\d+)', text))
        assert len(starts) == len(ends) == int(original['trials']) == 50
        folders = list((SIM/name/'model2robotwin_interface/demo_clean'/RUN).glob('*'))
        assert len(folders) == 1
        videos = folders[0]
        assert len(list(videos.glob('episode*.mp4'))) == 50
        folder = f'task_{index:02d}_{name}'
        (output/folder).mkdir()
        task = dict(task=name, task_index=index, folder=folder, description_ko=labels[name],
                    successes=int(original['successes']), trials=50,
                    success_rate_pct=float(original['success_rate_pct']), episodes=[])
        tasks.append(task)
        for source, dest in [(log, 'eval.log'), (Path(str(log).replace('_eval.log','_server.log')), 'server.log'), (videos/'_result.txt', '_result.txt')]:
            shutil.copy2(source, output/folder/dest)
            hashes[str(source)] = sha256(source)
            assert sha256(output/folder/dest) == hashes[str(source)]
        assert abs(float((videos/'_result.txt').read_text().strip().splitlines()[-1])-task['successes']/50) < 1e-9
        previous = 0
        for episode_index, (start, end) in enumerate(zip(starts, ends)):
            logged = json.loads(start[1]); cumulative, trial, seed = map(int, end.groups())
            assert start.end() < end.start() and trial == episode_index+1
            assert episode_index == 49 or end.end() < starts[episode_index+1].start()
            assert logged['task'] == name and logged['seed'] == seed
            success = cumulative-previous; assert success in (0,1)
            section = text[start.end():end.start()]
            assert 'Traceback (most recent call last):' not in section
            assert re.findall(r'^(Success!|Fail!)$', section, re.M) == ['Success!' if success else 'Fail!']
            steps = [tuple(map(int, s)) for s in re.findall(r'step:\s*(\d+) / (\d+)', section)]
            assert steps and [s[0] for s in steps] == list(range(1, steps[-1][0]+1))
            assert len({s[1] for s in steps}) == 1
            assert success or steps[-1][0] == steps[-1][1]
            relative = f'{folder}/episode_{episode_index:03d}_seed_{seed}_{"SUCCESS" if success else "FAIL"}.mp4'
            entry = dict(task=name, episode_index=episode_index, seed=seed, success=bool(success),
                         instruction=logged['instruction'], control_steps=steps[-1][0],
                         control_step_limit=steps[-1][1], video=relative, poster=str(Path(relative).with_suffix('.jpg')))
            jobs.append((videos/f'episode{episode_index}.mp4', output/relative, entry))
            previous = cumulative
        assert previous == task['successes']
    print('Validated all 50 task logs and 2,500 episode outcomes; preparing videos.', flush=True)
    task_map = {t['task']: t for t in tasks}
    # CPU-only decoding; keep resource use modest alongside the active evaluation.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for done, episode in enumerate(pool.map(prepare_video, jobs), 1):
            assert episode['fps'] == '10/1'
            task_map[episode['task']]['episodes'].append(episode)
            if done % 100 == 0:
                print(f'Copied, hashed, fully decoded, poster generated: {done}/2500', flush=True)
                write_json(output/'status.json', dict(status='building', verified_videos=done, target_videos=2500))
    episodes = [e for t in tasks for e in t['episodes']]
    assert len(episodes) == 2500 and sum(e['success'] for e in episodes) == 2201
    for task in tasks:
        assert len(task['episodes']) == 50 and sum(e['success'] for e in task['episodes']) == task['successes']
        write_json(output/task['folder']/'result.json', task)
    manifest = dict(status='complete', created_at=datetime.datetime.now().astimezone().isoformat(),
        model='Public TurboVLA 150K EMA', mode='demo_clean', run_id=RUN, tasks=tasks,
        trials=2500, successes=2201, failures=299, success_rate_pct=88.04,
        source_result=str(SOURCE), source_result_sha256=sha256(SOURCE), copied_log_sha256=hashes,
        camera='head', resolution_wh=[320,240], video_fps=10, videos_reencoded=False,
        path_convention='video and poster paths are relative to this gallery root, including in task result.json.',
        validation='Every outcome/seed/instruction/step sequence matched logs; all copied MP4 SHA256 matched; all frames decoded.')
    build_pages(output, manifest)
    write_json(output/'manifest.json', manifest)
    write_json(output/'reference_clean_result.json', reference)
    write_json(output/'summary.json', dict(status='complete', videos=2500, successes=2201, failures=299,
        success_rate_pct=88.04, verified_sha256=2500, fully_decoded=2500,
        video_bytes=sum(e['video_bytes'] for e in episodes), completed_at=manifest['created_at']))
    write_json(output/'status.json', dict(status='complete', verified_videos=2500))
    output.rename(OUTPUT)
    index = BASE/'index.html'
    text = index.read_text()
    addition = f'''<section class="note"><h2>공개 150K EMA · Clean 50회 평가</h2>
<p>2026-10-01 평가 · 50개 태스크 × 50회 · 성공 2,201 / 2,500 (88.04%)</p>
<p class="links"><a href="{OUTPUT.name}/index.html">Clean 2,500개 영상 →</a>
<a href="{OUTPUT.name}/preview.html">실패·성공 빠른 검토 →</a>
<a href="{OUTPUT.name}/manifest.json">전체 영상·결과 JSON</a></p></section>'''
    assert text.count('</main>') == 1 and OUTPUT.name not in text
    shutil.copy2(index, BASE/'index.before_clean_eval50.html')
    temporary = BASE/'index.html.tmp'
    temporary.write_text(text.replace('</main>', addition+'</main>'))
    temporary.replace(index)
    print(json.dumps(read_json(OUTPUT/'summary.json'), indent=2), flush=True)
    print(OUTPUT, flush=True)


if __name__ == '__main__':
    main()
