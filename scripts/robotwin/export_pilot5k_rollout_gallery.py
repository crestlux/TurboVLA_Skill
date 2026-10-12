"""Build a portable gallery from the 500 existing 5k EMA evaluation recordings.

No model or simulator is run. Source videos are copied byte-for-byte. Every copy
is hashed, probed, and fully decoded; a first-frame JPEG is generated alongside it.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / 'experiments/robotwin/evaluation_records'
RUN = 'turbovla_clean50_pilot5k_2x3090_20260927_0236'
SOURCE = RECORDS / RUN / 'validation.json'
QUICK_TASKS = ['adjust_bottle', 'lift_pot', 'place_container_plate',
               'handover_block', 'blocks_ranking_rgb', 'stack_blocks_three', 'grab_roller']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_steps(task):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', Path(task['eval_log']).read_text(errors='replace'))
    matches = list(re.finditer(r'Success rate:\s*(\d+)/(\d+)\s*=>[^\n]+current seed:\s*(\d+)', text))
    assert len(matches) == len(task['episodes']) == 10
    previous, cursor, details = 0, 0, []
    for match, episode in zip(matches, task['episodes']):
        cumulative, trial, seed = map(int, match.groups())
        assert trial == episode['episode_index'] + 1 and seed == episode['seed']
        assert cumulative - previous == int(episode['success'])
        section = text[cursor:match.start()]
        assert re.findall(r'^(Success!|Fail!)$', section, re.M) == ['Success!' if episode['success'] else 'Fail!']
        steps = [tuple(map(int, p)) for p in re.findall(r'step:\s*(\d+) / (\d+)', section)]
        assert steps and [p[0] for p in steps] == list(range(1, steps[-1][0] + 1))
        assert all(p[1] == steps[-1][1] for p in steps)
        if not episode['success']:
            assert steps[-1][0] == steps[-1][1]
        details.append(dict(control_steps=steps[-1][0], control_step_limit=steps[-1][1]))
        previous, cursor = cumulative, match.end()
    assert previous == task['successes']
    return details


def prepare_video(job):
    source, target, episode = job
    shutil.copy2(source, target)
    original_hash = sha256(source)
    assert sha256(target) == original_hash
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=codec_name,width,height,r_frame_rate,nb_frames,duration', '-of', 'json', str(target)],
        capture_output=True, text=True, check=True, timeout=30)
    stream = json.loads(probe.stdout)['streams'][0]
    frames = int(stream['nb_frames'])
    # RoboTwin saves one pre-action frame per step and an extra final success frame.
    assert episode['control_steps'] > 0
    assert frames == episode['control_steps'] + int(episode['success'])
    assert (stream['width'], stream['height'], stream['codec_name']) == (320, 240, 'h264')
    poster = target.with_suffix('.jpg')
    # The null output consumes the complete video; the JPEG output takes frame 0.
    decode = subprocess.run(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-nostdin', '-nostats',
         '-progress', 'pipe:1', '-threads', '1', '-i', str(target),
         '-map', '0:v:0', '-an', '-threads', '1', '-f', 'null', '-',
         '-map', '0:v:0', '-an', '-frames:v', '1', '-q:v', '4', '-threads', '1', '-update', '1', str(poster)],
        capture_output=True, text=True, check=True, timeout=90)
    decoded = int(re.findall(r'^frame=(\d+)$', decode.stdout, re.M)[-1])
    assert decoded == frames and 'progress=end' in decode.stdout
    assert poster.stat().st_size > 0
    return dict(episode, source_video=str(source), video_sha256=original_hash,
                video_bytes=target.stat().st_size, codec=stream['codec_name'],
                width=320, height=240, fps=stream['r_frame_rate'], frames=frames,
                duration_seconds=float(stream['duration']), full_decode_passed=True,
                decoded_frames=decoded)


def esc(value):
    return html.escape(str(value), quote=True)


def card(episode, prefix=''):
    label = '성공' if episode['success'] else '실패'
    css = 'success' if episode['success'] else 'failure'
    return f'''<article class="video-card" data-outcome="{css}">
      <div class="card-top"><strong>EP {episode['episode_index']:03d}</strong><span class="badge {css}">{label}</span></div>
      <video controls playsinline preload="none" poster="{esc(prefix + episode['poster'])}"><source src="{esc(prefix + episode['video'])}" type="video/mp4"></video>
      <p class="meta">seed {episode['seed']} · {episode['control_steps']} / {episode['control_step_limit']} step · {episode['duration_seconds']:.1f}초</p>
      <a href="{esc(prefix + episode['video'])}">MP4 직접 열기 ↗</a>
    </article>'''


def page(title, body, assets='', script=''):
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><link rel="stylesheet" href="{assets}gallery.css"></head>
<body>{body}<script>{script}</script></body></html>'''


CONTROLS = '''<div class="controls"><label>결과 <select id="outcome"><option value="all">전체</option><option value="failure">실패만</option><option value="success">성공만</option></select></label>
<label>재생 속도 <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
<button id="pause">모두 일시정지</button></div>'''


CONTROL_SCRIPT = '''
function applyOutcome() {
  const mode = document.getElementById('outcome').value;
  document.querySelectorAll('.video-card').forEach(card => {
    card.hidden = mode !== 'all' && card.dataset.outcome !== mode;
    if (card.hidden) card.querySelector('video').pause();
  });
}
function applySpeed() {
  document.querySelectorAll('video').forEach(video => video.playbackRate = Number(document.getElementById('speed').value));
}
document.getElementById('outcome').addEventListener('change', applyOutcome);
document.getElementById('speed').addEventListener('change', applySpeed);
document.getElementById('pause').addEventListener('click', () => document.querySelectorAll('video').forEach(video => video.pause()));
'''


def write_pages(output, manifest):
    tasks = manifest['tasks']
    css = '''[hidden]{display:none!important}*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e4edf8;font:16px/1.65 system-ui,-apple-system,sans-serif}a{color:#80c7ff;text-decoration:none}a:hover{text-decoration:underline}button,select,input{font:inherit;color:inherit;background:#142137;border:1px solid #334660;border-radius:8px;padding:8px 12px}button{cursor:pointer}button:hover{background:#22334b}header,.page{max-width:1500px;margin:auto;padding:28px}header{padding-bottom:14px}h1{font-size:30px;margin:0 0 8px}h2{font-size:23px;margin:6px 0}p{margin:8px 0}.muted,.meta{color:#a8b8cc}.meta{font-size:13px}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.stats span{background:#142137;border:1px solid #24354e;border-radius:10px;padding:8px 16px}.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:16px 0 22px}.layout{display:grid;grid-template-columns:295px minmax(0,1fr);gap:28px;max-width:1500px;padding:0 28px 40px;margin:auto}aside{position:sticky;top:18px;align-self:start}aside input{width:100%;margin-bottom:10px}.task-list{display:grid;gap:5px;max-height:72vh;overflow-y:auto}.task-link{display:block;padding:10px;border:1px solid transparent;border-radius:9px;color:#d7e7fb}.task-link small{display:block;color:#9aafc9;font-size:12px}.task-link.active{background:#18334b;border-color:#3d91bc}.task-link .score{float:right;color:#8ad4ed}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:18px}.video-card{background:#142137;padding:14px;border:1px solid #253650;border-radius:12px}.video-card[hidden]{display:none}.video-card video{display:block;width:100%;aspect-ratio:4/3;background:#050a11;margin:12px 0 6px;border-radius:6px}.card-top{display:flex;justify-content:space-between;align-items:center}.badge{border-radius:99px;padding:3px 10px;font-size:13px}.success{color:#91ebbb;background:#153e31}.failure{color:#ffb3a9;background:#492a32}.note{background:#142137;border-left:3px solid #63b5d8;padding:12px 16px;margin:14px 0}.task-section{margin:30px 0 48px}.links{display:flex;gap:18px;flex-wrap:wrap}noscript{display:block;padding:20px}.empty{padding:20px;color:#afc1d6}@media(max-width:850px){.layout{grid-template-columns:1fr}aside{position:static}.task-list{max-height:210px}.grid{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}header,.page{padding:20px}.layout{padding:0 20px 30px}h1{font-size:25px}}'''
    (output / 'gallery.css').write_text(css, encoding='utf-8')
    nav = ''.join(f'<a class="task-link" href="#{esc(t["task"])}" data-task="{esc(t["task"])}"><span class="score">{t["successes"]}/10</span>{esc(t["description_ko"])}<small>{esc(t["task"])}</small></a>' for t in sorted(tasks, key=lambda t: (t['successes'], t['task_index'])))
    embedded = json.dumps(tasks, ensure_ascii=False).replace('<', '\\u003c')
    script = 'const tasks = ' + embedded + ';\n' + '''
const e = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function showTask() {
  document.querySelectorAll('video').forEach(v => v.pause());
  const key = decodeURIComponent(location.hash.slice(1));
  const task = tasks.find(t => t.task === key) || tasks.find(t => t.task === 'adjust_bottle');
  document.querySelectorAll('.task-link').forEach(a => a.classList.toggle('active', a.dataset.task === task.task));
  document.getElementById('selected').innerHTML = `<h2>${e(task.description_ko)}</h2><p class="muted">${e(task.task)} · 성공 ${task.successes}/10 · 실패 ${10-task.successes}/10</p><p class="links"><a href="${e(task.folder)}/index.html">이 태스크만 열기</a><a href="${e(task.folder)}/result.json">결과 JSON</a><a href="${e(task.folder)}/eval.log">평가 로그</a></p>`;
  document.getElementById('videos').innerHTML = task.episodes.map(v => `<article class="video-card" data-outcome="${v.success?'success':'failure'}"><div class="card-top"><strong>EP ${String(v.episode_index).padStart(3,'0')}</strong><span class="badge ${v.success?'success':'failure'}">${v.success?'성공':'실패'}</span></div><video controls playsinline preload="none" poster="${e(v.poster)}"><source src="${e(v.video)}" type="video/mp4"></video><p class="meta">seed ${v.seed} · ${v.control_steps} / ${v.control_step_limit} step · ${v.duration_seconds.toFixed(1)}초</p><a href="${e(v.video)}">MP4 직접 열기 ↗</a></article>`).join('');
  applyOutcome(); applySpeed();
}
document.getElementById('search').addEventListener('input', event => {
  const value = event.target.value.trim().toLowerCase();
  document.querySelectorAll('.task-link').forEach(a => a.hidden = !a.textContent.toLowerCase().includes(value));
});
window.addEventListener('hashchange', showTask);
''' + CONTROL_SCRIPT + '\nshowTask();'
    body = f'''<header><h1>TurboVLA · 5K EMA 롤아웃</h1><p class="muted">RoboTwin Clean50 · 2026-09-27 본 평가 · 원본 머리 카메라 영상 320×240</p>
<div class="stats"><span>50개 태스크</span><span>500개 영상</span><span>성공 140 · 실패 360</span><span>전체 성공률 <strong>28.0%</strong></span></div>
<p class="links"><a href="preview.html">먼저 볼 실패·성공 사례 →</a><a href="WATCH_GUIDE.md">검토 가이드</a><a href="manifest.json">전체 영상 JSON</a><a href="README.md">영상 안내</a></p></header>
<div class="layout"><aside><input id="search" type="search" placeholder="태스크 이름 / 작업 검색" aria-label="태스크 검색"><p class="meta">성공 수가 적은 태스크부터 표시</p><nav class="task-list">{nav}</nav></aside>
<main><div id="selected"></div>{CONTROLS}<div id="videos" class="grid"></div><p class="meta">EP는 0부터 시작합니다. 영상은 10 fps로 저장됐으며 재생 길이는 실제 평가 소요 시간과 다릅니다.</p></main></div>
<noscript>브라우저에서 JavaScript를 켜거나 <a href="preview.html">빠른 검토 페이지</a>에서 영상을 보세요.</noscript>'''
    (output / 'index.html').write_text(page('TurboVLA 5K EMA 롤아웃', body, script=script), encoding='utf-8')
    chosen = []
    preview_sections = []
    for task in tasks:
        local_cards = ''.join(card(ep, '../') for ep in task['episodes'])
        body = f'''<main class="page"><a href="../index.html#{task['task']}">← 전체 태스크</a><h1>{esc(task['description_ko'])}</h1>
<p class="muted">{esc(task['task'])} · 성공 {task['successes']}/10 · 5K EMA · Clean</p>
<p class="links"><a href="result.json">결과 JSON</a><a href="eval.log">평가 로그</a><a href="server.log">모델 서버 로그</a></p>{CONTROLS}<div class="grid">{local_cards}</div></main>'''
        (output / task['folder'] / 'index.html').write_text(page(task['description_ko'], body, '../', CONTROL_SCRIPT), encoding='utf-8')
    for task_name in QUICK_TASKS:
        task = next(t for t in tasks if t['task'] == task_name)
        selected = [ep for ep in task['episodes'] if not ep['success']][:2] + [ep for ep in task['episodes'] if ep['success']][:1]
        selected.sort(key=lambda ep: ep['episode_index'])
        chosen.extend(dict(task=task_name, episode_index=ep['episode_index'], success=ep['success'], video=ep['video']) for ep in selected)
        preview_sections.append(f'''<section class="task-section"><h2>{esc(task['description_ko'])}</h2><p class="muted">{task_name} · 본 평가 {task['successes']}/10 성공 · <a href="{task['folder']}/index.html">10개 전체 보기</a></p><div class="grid">{''.join(card(ep) for ep in selected)}</div></section>''')
    body = '''<main class="page"><a href="index.html">← 전체 500개 영상</a><h1>5K EMA · 먼저 볼 실패와 성공</h1>
<p>7개 태스크에서 EP 번호가 앞선 실패 최대 2개와 성공 최대 1개를 골랐습니다.</p>
<p class="note">선정 기준은 평가 결과와 EP 순서입니다. 각 영상의 실패 원인을 분류한 목록은 아닙니다. 성공·실패 사례는 서로 다른 seed의 장면입니다.</p>''' + CONTROLS + ''.join(preview_sections) + '</main>'
    (output / 'preview.html').write_text(page('5K EMA 빠른 영상 검토', body, script=CONTROL_SCRIPT), encoding='utf-8')
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = read_json(SOURCE)
    labels = {t['task_id']: t['description_ko'] for t in read_json(RECORDS / 'released150k_20260929/evaluation_results.json')['tasks']}
    assert reference['scored_trials'] == 500 and reference['successes'] == 140
    jobs, tasks = [], []
    for index, original in enumerate(reference['tasks_detail']):
        task_name = original['task']
        folder = f'task_{index:02d}_{task_name}'
        (output / folder).mkdir()
        for key, name in [('eval_log', 'eval.log'), ('server_log', 'server.log'), ('simulator_result_file', '_result.txt')]:
            source = Path(original[key])
            expected = reference['audited_file_sha256'].get(str(source))
            if expected:
                assert sha256(source) == expected, str(source)
            shutil.copy2(source, output / folder / name)
        steps = parse_steps(original)
        tasks.append(dict(task=task_name, task_index=index, description_ko=labels[task_name],
                          folder=folder, successes=original['successes'], trials=10,
                          success_rate_pct=10.0 * original['successes'], episodes=[]))
        for episode, detail in zip(original['episodes'], steps):
            name = f"episode_{episode['episode_index']:03d}_seed_{episode['seed']}_{'SUCCESS' if episode['success'] else 'FAIL'}.mp4"
            relative = str(Path(folder) / name)
            entry = dict(task=task_name, episode_index=episode['episode_index'], seed=episode['seed'],
                         success=episode['success'], **detail, instruction=None,
                         video=relative, poster=str(Path(relative).with_suffix('.jpg')))
            jobs.append((Path(episode['video']), output / relative, entry))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for done, episode in enumerate(pool.map(prepare_video, jobs), 1):
            next(task for task in tasks if task['task'] == episode['task'])['episodes'].append(episode)
            if done % 50 == 0:
                print(f'Copied, SHA256 matched, fully decoded, and poster generated: {done}/500', flush=True)
    episodes = [ep for task in tasks for ep in task['episodes']]
    assert len(episodes) == len({ep['video'] for ep in episodes}) == 500
    assert sum(ep['success'] for ep in episodes) == 140
    assert sum(not ep['success'] for ep in episodes) == 360
    for task in tasks:
        assert len(task['episodes']) == 10 and sum(ep['success'] for ep in task['episodes']) == task['successes']
        write_json(output / task['folder'] / 'result.json', task)
    manifest = dict(schema_version=1, status='complete', created_at=datetime.datetime.now().astimezone().isoformat(),
                    run_id=RUN, model='Our TurboVLA 5,000-update EMA', eval_mode='demo_clean',
                    reference_validation=str(SOURCE), reference_validation_sha256=sha256(SOURCE),
                    checkpoint=reference['checkpoint_validation_record'], successes=140, failures=360,
                    trials=500, success_rate_pct=28.0, camera='original head camera', resolution_wh=[320, 240],
                    video_fps=10, videos_rerendered=False, videos_reencoded=False,
                    path_convention='video/poster paths are relative to the gallery root, including inside task result.json.',
                    instruction_note='Selected instruction text was not logged during the 5k run; task descriptions are summaries only.',
                    timing_note='Encoded playback at 10 fps: one pre-action frame per control step plus one final frame on success; not rollout wall-clock or guaranteed physical simulation time.',
                    validation='All 500 copies match original SHA256, have valid stream metadata, and pass complete ffmpeg decoding; frames equal logged control steps plus the final success frame when applicable.',
                    tasks=tasks)
    manifest['quick_preview'] = write_pages(output, manifest)
    write_json(output / 'manifest.json', manifest)
    shutil.copy2(SOURCE, output / 'reference_evaluation.json')
    summary = dict(status='complete', run_id=RUN, tasks=50, videos=500, successes=140, failures=360,
                   success_rate_pct=28.0, copied_video_bytes=sum(ep['video_bytes'] for ep in episodes),
                   copied_video_sha256_matches=500, fully_decoded_videos=500,
                   total_decoded_frames=sum(ep['decoded_frames'] for ep in episodes),
                   zero_success_tasks=[t['task'] for t in tasks if t['successes'] == 0],
                   created_at=manifest['created_at'], quick_preview_videos=len(manifest['quick_preview']))
    write_json(output / 'summary.json', summary)
    write_json(output / 'status.json', dict(status='complete', completed_tasks=50, total_tasks=50,
                                           total_videos=500, completed_at=manifest['created_at']))
    (output / 'README.md').write_text('''# TurboVLA 5K EMA 롤아웃 영상

우리 모델의 2026-09-27 RoboTwin Clean50 본 평가 영상이다. 50개 태스크 × 10회, 총 500개이며
성공 140개·실패 360개, 전체 성공률 28.0%다. 2회의 사전 검사는 포함하지 않는다.

- `index.html`: 태스크 검색, 성공/실패 필터, 재생 속도 선택. 파일을 브라우저로 직접 열면 된다.
- `preview.html`: 7개 태스크에서 첫 실패 최대 2개·첫 성공 최대 1개를 선택한 빠른 검토.
- `task_XX_태스크/index.html`: 해당 태스크의 10개 영상. MP4도 직접 재생할 수 있다.
- `manifest.json`: 모든 회차의 seed·결과·step·원본 경로·SHA256·영상 검사 결과.
- `reference_evaluation.json`: 기존 평가 검증 기록. `summary.json`: 이번 내보내기 요약.
- `WATCH_GUIDE.md`: 먼저 볼 태스크와 관찰할 부분.

`episode_000_seed_100001_FAIL.mp4`처럼 파일명에 EP·seed·성공 여부를 표시했다. EP는 0부터 시작한다.
태스크별 result.json의 영상 경로는 전체 갤러리 루트 기준이다. 원본 MP4를 그대로 복사했으며
모든 500개 영상의 해시 일치와 전체 프레임 디코딩을 확인했다. 새 평가를 실행하지 않았다.

원래 기록된 머리 카메라 320×240 영상이다. 모델은 카메라 3개를 입력받았지만 손목 카메라 영상은
이 녹화에 포함되지 않는다. 10 fps 재생 길이는 실제 평가 소요 시간이나 물리 시뮬레이션 시간과 다르다.
성공·실패는 기존 시뮬레이터 판정이며, 실제 선택된 영어 지시문은 이 5K 실행에서 기록하지 않았다.
한글 태스크 설명은 작업의 목적을 요약한다. 추가 녹화·행동 궤적 NPZ는 이번 내보내기에 없다.
''', encoding='utf-8')
    guide = ['# 먼저 볼 태스크', '',
             '다음은 평가 점수와 작업 종류를 기준으로 고른 검토 순서다. 영상에서 실패 원인을 미리 판정한 목록은 아니다.', '',
             '| 태스크 | 성공 / 10회 | 영상에서 확인할 부분 |', '|---|---:|---|']
    questions = {
        'adjust_bottle': '실패·성공 사례의 병 접근, 파지 위치, 들어 올리기 비교',
        'lift_pot': '두 손의 손잡이 접근, 파지 시점, 함께 들어 올리기',
        'place_container_plate': '성공 사례와 드문 실패 사례의 최종 위치·그리퍼 해제 비교',
        'handover_block': '첫 파지, 두 팔의 전달 위치, 받는 손의 파지, 마지막 배치 순서',
        'blocks_ranking_rgb': '색상별 대상 선택, 배치 순서, 내려놓은 위치',
        'stack_blocks_three': '첫 블록 이후 작업 진행, 윗블록 정렬, 쌓은 물체의 안정성',
        'grab_roller': '10/10 성공 태스크에서 보이는 접근·양팔 파지·들기 동작 확인',
    }
    for name in QUICK_TASKS:
        task = next(t for t in tasks if t['task'] == name)
        guide.append(f"| [{name}]({task['folder']}/index.html) | {task['successes']} | {questions[name]} |")
    guide += ['', '[빠른 검토 페이지](preview.html)에서 바로 재생할 수 있다.', '',
              '첫 접근 → 파지 → 이동/협동 → 목표 배치 → 종료의 어느 단계에서 진행이 멈추는지 기록하면 좋다.',
              '실패 360회는 모두 제어 step 제한에 도달했다. 이것만으로 파지·인식·제어 중 어떤 원인인지 알 수는 없다.',
              '동일 태스크의 성공·실패 예시는 서로 다른 seed의 장면이며, 선정 비율은 전체 성공률이 아니다.',
              '기존 성공 판정은 태스크 코드 기준이다. 특히 병 흔들기 태스크는 병 높이로 판정하므로 동작의 완성도도 영상에서 별도로 볼 수 있다.', '']
    (output / 'WATCH_GUIDE.md').write_text('\n'.join(guide), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
