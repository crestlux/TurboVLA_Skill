"""Export existing public 150k EMA Clean/Randomized rollouts as a portable gallery.

Uses the 5k gallery's media validator and visual style. No policy or simulator runs.
"""
import argparse
import concurrent.futures
import datetime
import json
import re
import shutil
from pathlib import Path

from export_pilot5k_rollout_gallery import (
    CONTROLS, CONTROL_SCRIPT, QUICK_TASKS, RECORDS,
    card, esc, page, prepare_video, read_json, sha256, write_json,
)


SOURCE = RECORDS / 'released150k_20260929'
PILOT_FOLDER = 'robotwin_clean50_5k_ema_20260927'
DEFAULT_STYLE = Path('/data2/jack/TurboVLA/rollout_videos') / PILOT_FOLDER / 'gallery.css'


def validate_task(task, mode, episodes):
    result = task['results'][mode]
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', Path(result['eval_log']).read_text(errors='replace'))
    starts = list(re.finditer(r'^EVAL_EPISODE (.+)$', text, re.M))
    ends = list(re.finditer(r'Success rate:\s*(\d+)/(\d+)\s*=>[^\n]+current seed:\s*(\d+)', text))
    assert len(starts) == len(ends) == len(episodes) == 10
    previous = 0
    for idx, (start, end, episode) in enumerate(zip(starts, ends, episodes)):
        assert episode['episode_index'] == idx and start.end() < end.start()
        if idx < 9:
            assert end.end() < starts[idx + 1].start()
        logged = json.loads(start[1])
        cumulative, trial, seed = map(int, end.groups())
        assert trial == idx + 1 and cumulative - previous == int(episode['success'])
        assert logged['task'] == task['task_id'] == episode['task_id']
        assert logged['seed'] == seed == episode['scene_seed']
        assert logged['instruction'] == episode['instruction']
        section = text[start.end():end.start()]
        assert 'Traceback (most recent call last):' not in section
        assert re.findall(r'^(Success!|Fail!)$', section, re.M) == ['Success!' if episode['success'] else 'Fail!']
        steps = [tuple(map(int, item)) for item in re.findall(r'step:\s*(\d+) / (\d+)', section)]
        assert [step for step, _ in steps] == list(range(1, episode['control_steps'] + 1))
        assert all(limit == episode['control_step_limit'] == task['control_step_limit'] for _, limit in steps)
        if not episode['success']:
            assert episode['control_steps'] == episode['control_step_limit']
        previous = cumulative
    assert previous == result['successes']
    assert [e['scene_seed'] for e in episodes] == result['scene_seeds']
    assert [e['episode_index'] for e in episodes if not e['success']] == result['failed_episode_indices']
    assert float(Path(result['simulator_result_file']).read_text().strip().splitlines()[-1]) == previous / 10


def video_card(episode, prefix=''):
    return card(episode, prefix).replace(
        '</article>', '<details><summary>실제 평가 지시문</summary><p lang="en">'
        + esc(episode['instruction']) + '</p></details></article>')


def write_variant_pages(output, manifest):
    label, tasks = manifest['label'], manifest['tasks']
    nav = ''.join(
        f'<a class="task-link" href="#{esc(t["task"])}" data-task="{esc(t["task"])}">'
        f'<span class="score">{t["successes"]}/10</span>{esc(t["description_ko"])}'
        f'<small>{esc(t["task"])}</small></a>'
        for t in sorted(tasks, key=lambda t: (t['successes'], t['task_index'])))
    script = 'const tasks = ' + json.dumps(tasks, ensure_ascii=False).replace('<', '\\u003c') + ';\n' + '''
const e = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function showTask() {
  document.querySelectorAll('video').forEach(v => v.pause());
  const key = decodeURIComponent(location.hash.slice(1));
  const task = tasks.find(t => t.task === key) || tasks.find(t => t.task === 'adjust_bottle');
  document.querySelectorAll('.task-link').forEach(a => a.classList.toggle('active', a.dataset.task === task.task));
  document.getElementById('selected').innerHTML = `<h2>${e(task.description_ko)}</h2><p class="muted">${e(task.task)} · 성공 ${task.successes}/10 · 실패 ${10-task.successes}/10</p><p class="links"><a href="${e(task.folder)}/index.html">이 태스크만 열기</a><a href="${e(task.folder)}/result.json">결과 JSON</a><a href="${e(task.folder)}/eval.log">평가 로그</a></p>`;
  document.getElementById('videos').innerHTML = task.episodes.map(v => `<article class="video-card" data-outcome="${v.success?'success':'failure'}"><div class="card-top"><strong>EP ${String(v.episode_index).padStart(3,'0')}</strong><span class="badge ${v.success?'success':'failure'}">${v.success?'성공':'실패'}</span></div><video controls playsinline preload="none" poster="${e(v.poster)}"><source src="${e(v.video)}" type="video/mp4"></video><p class="meta">seed ${v.seed} · ${v.control_steps} / ${v.control_step_limit} step · ${v.duration_seconds.toFixed(1)}초</p><a href="${e(v.video)}">MP4 직접 열기 ↗</a><details><summary>실제 평가 지시문</summary><p lang="en">${e(v.instruction)}</p></details></article>`).join('');
  applyOutcome(); applySpeed();
}
document.getElementById('search').addEventListener('input', event => {
  const value = event.target.value.trim().toLowerCase();
  document.querySelectorAll('.task-link').forEach(a => a.hidden = !a.textContent.toLowerCase().includes(value));
});
window.addEventListener('hashchange', showTask);
''' + CONTROL_SCRIPT + '\nshowTask();'
    links = '<a href="../index.html">← 공개 150K 전체</a><a href="../clean/index.html">Clean</a><a href="../randomized/index.html">Randomized</a>'
    body = f'''<header><p class="links">{links}</p><h1>공개 TurboVLA · 150K EMA · {label}</h1>
<p class="muted">RoboTwin 2.0 · 2026-09-28~29 본 평가 · 원본 머리 카메라 320×240</p>
<div class="stats"><span>50개 태스크</span><span>500개 영상</span><span>성공 {manifest['successes']} · 실패 {manifest['failures']}</span><span>성공률 <strong>{manifest['success_rate_pct']:.1f}%</strong></span></div>
<p class="links"><a href="preview.html">먼저 볼 실패·성공 사례 →</a><a href="../WATCH_GUIDE.md">검토 가이드</a><a href="manifest.json">이 환경의 영상 JSON</a></p></header>
<div class="layout"><aside><input id="search" type="search" placeholder="태스크 이름 / 작업 검색" aria-label="태스크 검색"><p class="meta">성공 수가 적은 태스크부터 표시</p><nav class="task-list">{nav}</nav></aside>
<main><div id="selected"></div>{CONTROLS}<div id="videos" class="grid"></div><p class="meta">EP는 0부터 시작합니다. 10 fps 영상의 재생 길이는 실제 평가 소요 시간과 다릅니다.</p></main></div>
<noscript><a href="preview.html">빠른 검토 페이지</a>에서 영상을 보세요.</noscript>'''
    (output / 'index.html').write_text(page(f'공개 150K EMA · {label}', body, script=script), encoding='utf-8')
    for task in tasks:
        body = f'''<main class="page"><a href="../index.html#{esc(task['task'])}">← {label} 전체 태스크</a>
<h1>{esc(task['description_ko'])}</h1><p class="muted">{esc(task['task'])} · 공개 150K EMA · {label} · 성공 {task['successes']}/10</p>
<p class="links"><a href="result.json">결과 JSON</a><a href="eval.log">평가 로그</a><a href="server.log">모델 서버 로그</a></p>
{CONTROLS}<div class="grid">{''.join(video_card(ep, '../') for ep in task['episodes'])}</div></main>'''
        (output / task['folder'] / 'index.html').write_text(page(f'{label} · {task["description_ko"]}', body, '../', CONTROL_SCRIPT), encoding='utf-8')
    quick_tasks = list(QUICK_TASKS)
    # Add the three lowest-scoring tasks in this variant, keeping ties deterministic.
    for task in sorted(tasks, key=lambda t: (t['successes'], t['task_index']))[:3]:
        if task['task'] not in quick_tasks:
            quick_tasks.append(task['task'])
    selected, sections = [], []
    for task_name in quick_tasks:
        task = next(t for t in tasks if t['task'] == task_name)
        episodes = [e for e in task['episodes'] if not e['success']][:2] + [e for e in task['episodes'] if e['success']][:1]
        episodes.sort(key=lambda e: e['episode_index'])
        selected.extend(dict(task=task_name, episode_index=e['episode_index'], success=e['success'], video=e['video']) for e in episodes)
        sections.append(f'''<section class="task-section"><h2>{esc(task['description_ko'])}</h2><p class="muted">{task_name} · {label} · 성공 {task['successes']}/10 · <a href="{task['folder']}/index.html">10개 전체 보기</a></p><div class="grid">{''.join(video_card(e) for e in episodes)}</div></section>''')
    body = f'''<main class="page"><p class="links">{links}</p><h1>공개 150K EMA · {label} · 빠른 검토</h1><p><a href="index.html">이 환경의 전체 500개 영상 →</a></p>
<p>{len(quick_tasks)}개 태스크 · {len(selected)}개 영상. 5K 검토와 같은 7개 태스크에 이 환경에서 성공 수가 적은 3개 태스크를 추가했습니다.</p>
<p class="note">각 태스크에서 EP 번호가 앞선 실패 최대 2개·성공 최대 1개를 선택했습니다. 실패 원인을 미리 분류한 목록은 아닙니다. 영상마다 실제 평가 지시문을 펼쳐 볼 수 있습니다.</p>{CONTROLS}{''.join(sections)}</main>'''
    (output / 'preview.html').write_text(page(f'공개 150K EMA · {label} 빠른 검토', body, script=CONTROL_SCRIPT), encoding='utf-8')
    manifest['quick_preview'] = selected
    manifest['quick_preview_tasks'] = quick_tasks


def write_root(output, reference, variants):
    rows = ''.join(f'<tr><td>{v["label"]}</td><td>{v["successes"]}/500</td><td>{v["success_rate_pct"]:.1f}%</td><td><a href="{v["folder"]}/index.html">전체 500개</a> · <a href="{v["folder"]}/preview.html">빠른 검토 {len(v["quick_preview"])}개</a></td></tr>' for v in variants)
    body = f'''<main class="page"><a href="../index.html">← 모든 모델</a><h1>공개 TurboVLA · 150K EMA 롤아웃</h1>
<p class="muted">호환성 수정 후 2026-09-28~29 본 평가 · RoboTwin 2.0 · 머리 카메라 320×240</p>
<div class="stats"><span>50개 태스크 × 2개 환경 × 10회</span><span>성공 882 · 실패 118</span><span>전체 성공률 <strong>88.2%</strong></span></div>
<table><thead><tr><th>환경</th><th>성공</th><th>성공률</th><th>영상</th></tr></thead><tbody>{rows}</tbody></table>
<p class="links"><a href="preview.html">빠른 검토 안내 →</a><a href="WATCH_GUIDE.md">검토 가이드</a><a href="manifest.json">전체 영상 JSON</a><a href="reference_evaluation.json">원본 평가 결과 JSON</a><a href="../{PILOT_FOLDER}/index.html">우리 5K Clean 영상</a></p>
<p class="note">태스크·환경당 10회의 기존 평가입니다. 성공·실패는 시뮬레이터 판정입니다. Clean과 Randomized, 5K와 공개 150K는 장면 seed와 지시문이 같다고 가정할 수 없습니다.</p>
</main>'''
    (output / 'index.html').write_text(page('공개 150K EMA 롤아웃', body), encoding='utf-8')
    body = f'''<main class="page"><a href="index.html">← 공개 150K 전체</a><h1>공개 150K EMA · 빠른 영상 검토</h1>
<p>5K에서 먼저 보았던 7개 태스크와 각 환경에서 실패가 많았던 태스크를 함께 볼 수 있습니다.</p>
<div class="grid">{''.join(f'<section class="note"><h2>{v["label"]}</h2><p>{len(v["quick_preview_tasks"])}개 태스크 · 영상 {len(v["quick_preview"])}개</p><a href="{v["folder"]}/preview.html">실패·성공 사례 재생 →</a></section>' for v in variants)}</div>
<p class="links"><a href="../{PILOT_FOLDER}/preview.html">우리 5K 빠른 검토</a><a href="WATCH_GUIDE.md">검토 가이드</a></p>
<p>각 영상에 실제 지시문과 seed를 표시했습니다. 서로 다른 장면의 사례이므로 동일 조건의 일대일 비교로 해석하지 않습니다.</p></main>'''
    (output / 'preview.html').write_text(page('공개 150K EMA 빠른 검토', body), encoding='utf-8')
    guide = ['# 공개 150K EMA 영상 검토', '',
             '먼저 `clean/preview.html`, 이어서 `randomized/preview.html`을 브라우저로 열면 된다.', '',
             '5K 빠른 검토와 같은 7개 태스크에 환경별 성공 수가 가장 적은 3개 태스크를 추가했다.',
             '각 태스크의 첫 실패 최대 2개·첫 성공 최대 1개를 EP 순서대로 선택했다. 시각적 실패 원인을 미리 판정한 것은 아니다.', '',
             '| 태스크 | Clean 성공 / 10 | Randomized 성공 / 10 |', '|---|---:|---:|']
    chosen = set(t for v in variants for t in v['quick_preview_tasks'])
    for i, task in enumerate(reference['tasks']):
        if task['task_id'] in chosen:
            folder = f'task_{i:02d}_{task["task_id"]}'
            guide.append(f'| {task["task_id"]} | [{task["results"]["demo_clean"]["successes"]}](clean/{folder}/index.html) | [{task["results"]["demo_randomized"]["successes"]}](randomized/{folder}/index.html) |')
    guide += ['', '관찰 순서: 실제 지시문 → 물체 선택과 접근 → 파지 → 양팔 협동/운반 → 목표 위치·방향 → 해제와 종료.',
              '실패 118회는 모두 제어 step 제한에 도달했다. 이것만으로 인식·파지·제어 중 원인을 특정할 수 없다.',
              '실제 영어 지시문은 영상 카드에서 펼쳐 볼 수 있다. 한글 설명은 태스크 목적 요약이다.',
              '성공 판정은 시뮬레이터 코드 기준이다. 병 흔들기 태스크는 병 높이로 판정하므로 동작도 별도로 살펴본다.',
              '5K는 Clean 학습 모델이고 공개 150K는 Clean+Randomized 학습 모델이다. 스텝 수 외에도 학습 조건이 다르다.',
              '환경 간 또는 모델 간 장면과 지시문이 동일하다고 가정하지 않는다. 5K의 실제 지시문은 원래 기록되지 않았다.', '']
    (output / 'WATCH_GUIDE.md').write_text('\n'.join(guide), encoding='utf-8')
    (output / 'README.md').write_text('''# 공개 TurboVLA 150K EMA 롤아웃

2026-09-28~29 호환성 수정 후 본 평가의 원본 영상 1,000개다.
Clean 443/500(88.6%), Randomized 439/500(87.8%), 합계 882/1000(88.2%).

- `index.html`: 환경 선택. `preview.html`: 빠른 검토 안내.
- `clean/`, `randomized/`: 각 환경의 갤러리, 빠른 검토, 태스크별 영상·결과 JSON·원본 로그.
- `manifest.json`: 두 환경의 전체 영상 목록·실제 지시문·seed·결과·step·원본 경로·SHA256.
- `reference_evaluation.json`, `reference_episodes.json`: 원본 평가 결과.
- `summary.json`: 영상 내보내기 검증 요약. `WATCH_GUIDE.md`: 관찰 순서와 비교 시 유의점.

각 환경의 manifest.json 및 태스크 result.json에 있는 video/poster 경로는 해당 환경 폴더 기준이다.
전체 manifest.json의 variants 항목에서 folder와 하위 video/poster를 결합하면 루트 기준 경로가 된다.
MP4는 재인코딩 없이 원본 그대로 복사했다. 모든 영상의 SHA256 일치와 전체 프레임 디코딩을 확인했다.
새 학습이나 평가를 실행하지 않았다. 영상은 머리 카메라 320×240, 10 fps이며 손목 카메라는 녹화되지 않았다.
한 제어 step마다 동작 전 프레임이 기록되고, 성공 시 마지막 프레임이 한 장 추가된다.
재생 길이는 실제 평가 소요 시간이나 물리 시뮬레이션 시간과 다르다.

태스크·환경당 10회의 평가이며, 성공 여부는 기존 시뮬레이터 판정이다.
Clean/Randomized의 seed와 지시문은 같지 않을 수 있다. 5K와도 동일 조건의 일대일 비교가 아니다.
브라우저에서 HTML을 직접 열 수 있다. 이 폴더 전체를 복사하면 영상과 내부 페이지가 함께 열린다.
상위 전체 모델 목록과 5K 링크는 함께 배치한 5K 갤러리를 가리킨다.
''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--style-source', type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), output
    reference = read_json(SOURCE / 'evaluation_results.json')
    source_episodes = read_json(SOURCE / 'episode_results.json')['episodes']
    assert reference['status'] == 'completed' and reference['overall']['successes'] == 882
    assert len(source_episodes) == len({e['episode_id'] for e in source_episodes}) == 1000
    assert sum(e['success'] for e in source_episodes) == 882
    css = args.style_source.read_text() + '\ndetails{margin-top:12px;font-size:14px}summary{cursor:pointer;color:#b4d9f5}details p{overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;margin:22px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #334660}\n'
    output.mkdir(parents=True)
    (output / 'gallery.css').write_text(css, encoding='utf-8')
    created = datetime.datetime.now().astimezone().isoformat()
    jobs, variants, task_map, copied_sources = [], [], {}, {}
    for variant in reference['variants']:
        mode = variant['mode']
        folder = 'clean' if mode == 'demo_clean' else 'randomized'
        target = output / folder
        target.mkdir()
        (target / 'gallery.css').write_text(css, encoding='utf-8')
        item = dict(schema_version=1, status='complete', created_at=created, mode=mode,
                    label=variant['label'], folder=folder, run_id=variant['run_id'],
                    model=reference['model'], successes=variant['successes'], failures=variant['failures'],
                    trials=500, success_rate_pct=variant['success_rate_pct'], tasks=[],
                    path_convention='video/poster paths are relative to this variant folder, including in task result.json.')
        variants.append(item)
        for index, task in enumerate(reference['tasks']):
            name = task['task_id']
            result = task['results'][mode]
            episodes = sorted((e for e in source_episodes if e['mode'] == mode and e['task_id'] == name), key=lambda e: e['episode_index'])
            assert all(e['run_id'] == variant['run_id'] for e in episodes)
            validate_task(task, mode, episodes)
            task_folder = f'task_{index:02d}_{name}'
            (target / task_folder).mkdir()
            task_entry = dict(task=name, task_index=index, description_ko=task['description_ko'],
                              folder=task_folder, mode=mode, successes=result['successes'],
                              failures=result['failures'], trials=10, success_rate_pct=result['success_rate_pct'], episodes=[])
            item['tasks'].append(task_entry)
            task_map[(mode, name)] = task_entry
            for key, filename in [('eval_log', 'eval.log'), ('server_log', 'server.log'), ('simulator_result_file', '_result.txt')]:
                source, dest = Path(result[key]), target / task_folder / filename
                shutil.copy2(source, dest)
                checksum = sha256(source)
                assert sha256(dest) == checksum
                copied_sources[str(source)] = checksum
            for ep in episodes:
                filename = f'episode_{ep["episode_index"]:03d}_seed_{ep["scene_seed"]}_{"SUCCESS" if ep["success"] else "FAIL"}.mp4'
                relative = str(Path(task_folder) / filename)
                entry = dict(task=name, mode=mode, episode_index=ep['episode_index'],
                             episode_id=ep['episode_id'], seed=ep['scene_seed'], success=ep['success'],
                             outcome=ep['outcome'], control_steps=ep['control_steps'],
                             control_step_limit=ep['control_step_limit'], instruction=ep['instruction'],
                             video=relative, poster=str(Path(relative).with_suffix('.jpg')))
                source = Path(ep['video']['path'])
                assert source.stat().st_size == ep['video']['bytes']
                jobs.append((source, target / relative, entry))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for done, episode in enumerate(pool.map(prepare_video, jobs), 1):
            assert episode['fps'] == '10/1'
            task_map[(episode['mode'], episode['task'])]['episodes'].append(episode)
            if done % 100 == 0:
                print(f'Copied, SHA256 matched, fully decoded, poster generated: {done}/1000', flush=True)
    for variant in variants:
        episodes = [e for t in variant['tasks'] for e in t['episodes']]
        assert len(episodes) == 500 and sum(e['success'] for e in episodes) == variant['successes']
        for task in variant['tasks']:
            assert len(task['episodes']) == 10 and sum(e['success'] for e in task['episodes']) == task['successes']
            write_json(output / variant['folder'] / task['folder'] / 'result.json', task)
        write_variant_pages(output / variant['folder'], variant)
        write_json(output / variant['folder'] / 'manifest.json', variant)
    for source_name, filename in [('evaluation_results.json', 'reference_evaluation.json'), ('episode_results.json', 'reference_episodes.json'), ('completion_audit.json', 'reference_completion_audit.json'), ('task_comparison.csv', 'reference_task_comparison.csv')]:
        source, dest = SOURCE / source_name, output / filename
        shutil.copy2(source, dest)
        assert sha256(source) == sha256(dest)
        copied_sources[str(source)] = sha256(source)
    write_root(output, reference, variants)
    episodes = [e for v in variants for t in v['tasks'] for e in t['episodes']]
    manifest = dict(schema_version=1, status='complete', created_at=created,
                    model=reference['model'], protocol=reference['protocol'],
                    reference_evaluation='reference_evaluation.json', source_records=str(SOURCE),
                    overall=reference['overall'], variants=variants, copied_source_sha256=copied_sources,
                    camera='original head camera', resolution_wh=[320, 240], video_fps=10,
                    videos_reencoded=False, videos_rerendered=False,
                    path_convention='Join variants[].folder and each episode video/poster path for a path relative to this gallery root.',
                    validation='All 1000 copied videos match original SHA256 and pass full ffmpeg decoding. Frame counts equal logged control steps plus one final frame on success. Every outcome, scene seed, instruction, and step sequence was matched against the original evaluation logs.')
    write_json(output / 'manifest.json', manifest)
    summary = dict(status='complete', model='Public TurboVLA 150K EMA', tasks=50, task_variants=100,
                   videos=1000, successes=882, failures=118, success_rate_pct=88.2,
                   video_bytes=sum(e['video_bytes'] for e in episodes),
                   video_sha256_matches=1000, fully_decoded_videos=1000,
                   total_decoded_frames=sum(e['decoded_frames'] for e in episodes),
                   quick_preview_videos=sum(len(v['quick_preview']) for v in variants),
                   variants=[{k:v[k] for k in ['mode','label','folder','successes','failures','trials','success_rate_pct']} for v in variants],
                   completed_at=datetime.datetime.now().astimezone().isoformat())
    write_json(output / 'summary.json', summary)
    write_json(output / 'status.json', dict(status='complete', completed_task_variants=100, total_videos=1000, completed_at=summary['completed_at']))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
