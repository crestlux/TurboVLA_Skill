#!/usr/bin/env python3
"""Snapshot real-robot training logs and produce a settings and artifact catalog."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
import zipfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {
    "ff_original": ("FF USB v2 원본 그리퍼", "FF USB v2 / original gripper"),
    "ff_latched": ("FF USB v2 그리퍼 리라벨", "FF USB v2 / latched gripper"),
    "go_latched": ("GO 그리퍼 리라벨", "GO / latched gripper"),
    "go_original": ("GO 원본 그리퍼", "GO / original gripper"),
    "go_ft500": ("GO FT GRU 500Hz 명목", "GO / GRU / nominal 500 Hz"),
    "go_ft100": ("GO FT GRU 100Hz 명목", "GO / GRU / nominal 100 Hz"),
}
STATUS = {"completed": "완료", "running": "학습 중", "queued": "대기", "cancelled": "사용자 취소",
          "interrupted": "중단 보관", "paused": "일시 정지 기록", "failed": "실패", "stopped_for_test": "검증용 중단"}
TEXT_NAMES = ("config.json", "recipe.json", "data_contract.json", "data_signature.json", "training_status.json",
              "job_status.json", "queue_status.json", "train.log", "metrics.jsonl", "evaluation.log", "preflight.log",
              "worker.log", "verification.log", "progress.log", "progress.jsonl", "progress.json", "interruption.json",
              "training_time.json")


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def discover():
    jobs, archives, containers = [], [], []
    for root in (Path("/data/jack/turbovla_ff_usb_v2_runs"), ROOT / "outputs/go_runs",
                 ROOT / "outputs/go_no_gripper_relabel_runs", ROOT / "outputs/go_ft_rate_runs"):
        if not root.exists():
            continue
        for run in sorted(root.iterdir()):
            if not run.is_dir():
                continue
            stages = sorted(run.glob("ft_*hz"), reverse=True) if (run / "schedule.json").exists() else [run]
            containers.append(run)
            for stage in stages:
                for mode in ("delta", "relative"):
                    folder = stage / mode
                    if (folder / "config.json").exists() or (folder / "recipe.json").exists():
                        jobs.append((folder.resolve(), "production", None))
            for folder in sorted(run.glob("interrupted/*")):
                if (folder / "config.json").exists():
                    archives.append((folder.resolve(), "interrupted_attempt", None))
            for folder in sorted(run.glob("pauses/*")):
                if (folder / "training_status.json").exists() and (run / "delta/config.json").exists():
                    archives.append((folder.resolve(), "pause_snapshot", run / "delta/config.json"))
    return jobs + archives, containers


def family(folder):
    text = str(folder)
    if "ft_500hz" in text:
        return "go_ft500"
    if "ft_100hz" in text:
        return "go_ft100"
    if "go_no_gripper_relabel_runs" in text:
        return "go_original"
    if "go_runs" in text:
        return "go_latched"
    return "ff_latched" if "gripper_latched" in text else "ff_original"


def snapshot_file(source, destination, manifest):
    data = source.read_bytes()
    destination.write_bytes(data)
    manifest.append(dict(source=str(source), snapshot=str(destination), bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest()))
    return data


def metric_rows(path):
    if not path.exists():
        return [], 0
    rows, invalid = {}, 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
            rows[int(value["step"])] = value
        except (ValueError, KeyError):
            invalid += 1
    return [rows[key] for key in sorted(rows)], invalid


def gpu_details(folder, mode, job):
    uuid = job.get("gpu_uuid", job.get("gpu"))
    index = job.get("gpu_index")
    for parent in list(folder.parents)[:4]:
        for name in ("launch_manifest.json", "launch.json", "schedule.json"):
            payload = read_json(parent / name)
            candidate = payload.get("jobs", {}).get(mode, payload if "gpu_uuid" in payload else {})
            if candidate and (uuid is None or candidate.get("gpu_uuid") == uuid):
                uuid = uuid or candidate.get("gpu_uuid")
                index = candidate.get("gpu_index", index)
    known = {"GPU-1bb2706c-5988-67f4-6ca0-1dd0e962f41e": 0, "GPU-7e7d7f97-c512-32b6-0a17-bfa6c8d19fe5": 1}
    return index if index is not None else known.get(uuid), uuid


def cached_replay_loss(config, normalization, npz_path):
    """Recover normalized masked L1 from saved float32 physical predictions.

    This is training-demo replay with dropout/jitter off, not a held-out loss.
    Re-encoding float32 physical actions introduces small rounding differences
    from the original normalized model outputs; no model inference is rerun.
    """
    import h5py
    import numpy as np

    stats = normalization["actions"]

    def encode(values):
        values = np.asarray(values, dtype=np.float64)
        if stats["method"] == "symlog_minmax":
            values = (values - np.asarray(stats["offset"])) / np.asarray(stats["scale"])
        elif stats["method"] != "symlog":
            raise ValueError("Unsupported replay action normalization")
        temperature = stats["temperature"]
        return np.sign(values) * np.log1p(np.abs(values) * temperature) / np.log1p(temperature)

    paths = sorted(Path(config["data"]["dataset_dir"]).glob("*.hdf5"))
    with np.load(npz_path, allow_pickle=False) as saved:
        predictions = saved["physical_actions"]
        names, starts = saved["episode_names"], saved["episode_starts"]
        source_targets = saved["source_targets"]
    if [p.name for p in paths] != names.tolist():
        raise ValueError("Cached replay episode order differs from the training directory")
    total, first_total = np.zeros(7), np.zeros(7)
    count, first_count = 0, 0
    target_hash = hashlib.sha256()
    for index, path in enumerate(paths):
        start, end = map(int, starts[index:index+2])
        with h5py.File(path,"r") as handle:
            target = handle["actions"][:].astype(np.float32)
            mask = handle["action_mask"][:]
        if end-start != len(target) or target.shape[1:] != predictions.shape[2:]:
            raise ValueError("Cached replay shape differs from the training targets")
        if config["data"].get("gripper_relabel"):
            future = np.arange(len(target))[:,None] + np.arange(target.shape[1])[None,:]
            grip = source_targets[start:end,6]
            physical = np.zeros((int(mask.sum()),7))
            physical[:,6] = grip[future[mask]]
            target[...,6][mask] = encode(physical)[:,6]
        target_hash.update(path.name.encode())
        target_hash.update(np.ascontiguousarray(target[mask,:6]).tobytes())
        target_hash.update(mask.tobytes())
        for prediction in predictions[:,start:end]:
            encoded = encode(prediction)
            error = np.abs(encoded[mask] - target[mask].astype(np.float64))
            if not np.isfinite(error).all():
                raise ValueError("Nonfinite cached replay loss on valid targets")
            total += error.sum(axis=0)
            count += len(error)
            first_error = np.abs(encoded[:,0] - target[:,0])
            first_total += first_error.sum(axis=0)
            first_count += len(first_error)
    channel = total/count
    result = dict(total_l1=float(channel.mean()), arm_l1=float(channel[:6].mean()), gripper_l1=float(channel[6]),
                  per_channel_l1=channel.tolist(), valid_chunk_entries=count, first_step_total_l1=float((first_total/first_count).mean()),
                  arm_target_sha256=target_hash.hexdigest(), source_npz=str(npz_path),
                  method="Re-encoded saved float32 physical actions; approximate normalized masked L1; all training demos, eval mode")
    assert np.isclose(result["total_l1"],(6*result["arm_l1"]+result["gripper_l1"])/7,atol=1e-12)
    return result


def aligned_training_losses(cases):
    comparisons=[]
    for mode in ("delta","relative"):
        baseline=next((c for c in cases if c["record_type"]=="production" and c["family"]=="go_original" and c["mode"]==mode),None)
        gru=next((c for c in cases if c["record_type"]=="production" and c["family"]=="go_ft500" and c["mode"]==mode),None)
        if not baseline or not gru or not baseline["metric_history"] or not gru["metric_history"]:
            continue
        assert baseline["train"]==gru["train"] and baseline["images"]==gru["images"]
        last=min(baseline["metric_history"][-1]["step"],gru["metric_history"][-1]["step"])
        shared=sorted(set(p["step"] for p in baseline["metric_history"] if p["step"]<=last) &
                      set(p["step"] for p in gru["metric_history"] if p["step"]<=last))[-20:]
        if not shared:
            continue
        base={p["step"]:p["loss"] for p in baseline["metric_history"]}
        current={p["step"]:p["loss"] for p in gru["metric_history"]}
        a=sum(base[s] for s in shared)/len(shared)
        b=sum(current[s] for s in shared)/len(shared)
        comparisons.append(dict(mode=mode,baseline_id=baseline["id"],gru_id=gru["id"],from_step=shared[0],to_step=shared[-1],
                                logged_points=len(shared),baseline_mean_l1=a,gru_mean_l1=b,relative_change_percent=(b/a-1)*100))
    return comparisons


def capture_case(folder, kind, inherited, destination, model_id, manifest):
    destination.mkdir()
    (destination / "live").symlink_to(folder, target_is_directory=True)
    snapshots = destination / "snapshots"
    snapshots.mkdir()
    names = set(TEXT_NAMES) | {p.name for p in folder.glob("metrics_before_resume_*.jsonl")}
    for name in sorted(names):
        source = folder / name
        if source.is_file():
            snapshot_file(source, snapshots / name, manifest)
    if inherited is not None:
        snapshot_file(inherited, snapshots / "inherited_config.json", manifest)
    config_name = next(name for name in ("config.json", "recipe.json", "inherited_config.json") if (snapshots / name).exists())
    config = read_json(snapshots / config_name)
    data, model, train = config["data"], config["model"], config["train"]
    contract = read_json(snapshots / "data_contract.json")
    normalization = contract.get("normalization") or read_json(Path(data["dataset_dir"]) / "normalization.json")
    action_contract = contract.get("action_contract", normalization.get("schema", {}).get("action", {}))
    relabel = action_contract.get("gripper_relabel") or data.get("gripper_relabel")
    row_gripper = "latched -1/+1" if relabel else "original signed/zero commands"
    gripper_source = "loader" if data.get("gripper_relabel") else ("stored relabel" if relabel else "stored original")
    status = read_json(snapshots / "training_status.json")
    job = read_json(snapshots / "job_status.json")
    if not job and folder.name in ("delta", "relative"):
        job = read_json(folder.parent / "job_status.json")
    rows, invalid_lines = metric_rows(snapshots / "metrics.jsonl")
    last = rows[-1] if rows else status.get("last_metric", {})
    mode = data["action_mode"]
    group = family(folder)
    index, uuid = gpu_details(folder, mode, job)
    state = status.get("status", job.get("status", job.get("phase", "queued")))
    frames, batch, budget = data["expected_frames"], train["batch_size"], train["max_samples"]
    epochs, tail = divmod(budget, frames)
    steps = epochs * math.ceil(frames / batch) + math.ceil(tail / batch)
    checkpoints = []
    latest = folder / "latest.pt"
    latest_stat = latest.stat() if latest.exists() else None
    latest_step = None
    for path in sorted(folder.glob("checkpoint_*.pt")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue  # A live run may prune its own old checkpoint during capture.
        step = int(re.search(r"checkpoint_(\d+)", path.name).group(1))
        is_latest = latest_stat is not None and (stat.st_ino, stat.st_dev) == (latest_stat.st_ino, latest_stat.st_dev)
        if is_latest:
            latest_step = step
        checkpoints.append(dict(path=str(path), step=step, bytes=stat.st_size, is_latest=is_latest))
    if latest_step is None and checkpoints:
        latest_step = checkpoints[-1]["step"]
    source_files = sorted(Path(data["dataset_dir"]).glob("*.hdf5"))
    physical_dir = str(source_files[0].resolve().parent) if source_files else None
    evaluations, replay_losses = {}, {}
    for weights in ("raw", "ema"):
        p = folder.parent / "evaluation" / f"{mode}_{weights}.json"
        if kind == "production" and p.exists():
            snapshot_file(p, snapshots / f"evaluation_{weights}.json", manifest)
            evaluations[weights] = read_json(snapshots / f"evaluation_{weights}.json")
            npz_path=p.with_suffix(".npz")
            if npz_path.exists():
                replay_losses[weights]=cached_replay_loss(config,normalization,npz_path)
    ft = model.get("ft_history", {})
    rate = data.get("ft_history_resampling")
    case = dict(id=model_id, record_type=kind, family=group, label=FAMILIES[group][0], mode=mode,
                source_dir=str(folder), report_dir=str(destination), config_source=config_name,
                status=state, phase=job.get("phase"), step=status.get("step", last.get("step", 0)),
                target_steps=status.get("target_steps", steps), samples_seen=status.get("samples_seen", last.get("samples_seen", 0)),
                target_samples=budget, target_epochs=budget / frames, effective_epochs=status.get("samples_seen", 0) / frames,
                latest_logged_step=last.get("step"), latest_logged_loss=last.get("loss"),
                recent_20_logged_losses_mean=sum(r["loss"] for r in rows[-20:]) / len(rows[-20:]) if rows else None,
                elapsed_seconds=status.get("elapsed_seconds", last.get("elapsed_seconds", 0)), eta=status.get("eta"),
                parameters=status.get("parameters"), gpu_index=index, gpu_uuid=uuid,
                episodes=data["expected_episodes"], frames=frames, dataset_dir=data["dataset_dir"], physical_dataset_dir=physical_dir,
                use_language=model.get("use_language", True), instruction=data.get("default_instruction"),
                gripper=row_gripper, gripper_source=gripper_source, gripper_relabel=relabel,
                current_ft_in_19d_state=True, use_ft_history=ft.get("enabled", False), ft_history_config=ft or None,
                ft_history_shape=[ft["history_length"], ft["input_dim"]] if ft.get("enabled") else None,
                ft_resampling=rate, nominal_ft_hz=rate.get("target_hz") if rate else None,
                action_hz=1 / action_contract.get("action_period_s", .1),
                action_normalization=normalization.get("actions", {}).get("method"),
                action_contract=action_contract, train=train, model=model, images=config["images"],
                checkpoint_latest=str(latest) if latest.exists() else (checkpoints[-1]["path"] if checkpoints else None),
                checkpoint_latest_step=latest_step, checkpoints=checkpoints,
                evaluations=evaluations, cached_replay_losses=replay_losses, metric_rows=len(rows), skipped_incomplete_metric_lines=invalid_lines,
                metric_history=rows, config=config, artifact_dir=str(folder.parent / "evaluation"))
    write_json(destination / "settings_and_status.json", {k:v for k,v in case.items() if k != "metric_history"})
    return case


def draw_curves(cases, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
    colors = {"delta": "#cb681a", "relative": "#1764ab"}
    for ax, (group, (_, title)) in zip(axes.flat, FAMILIES.items()):
        selected = [c for c in cases if c["family"] == group and c["record_type"] == "production"]
        any_points = False
        for c in selected:
            rows = c["metric_history"]
            if not rows:
                continue
            any_points = True
            x, y = np.array([r["step"] for r in rows]), np.array([r["loss"] for r in rows])
            smooth = np.array([y[max(i-19,0):i+1].mean() for i in range(len(y))])
            label = f"{c['mode'].capitalize()} ({c['status']})"
            ax.plot(x, y, color=colors[c["mode"]], alpha=.18, lw=.6)
            ax.plot(x, smooth, color=colors[c["mode"]], lw=1.5, label=label)
            ax.scatter(x[-1], smooth[-1], s=15, color=colors[c["mode"]])
        if any_points:
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        else:
            ax.text(.5,.5,"Queued: no training metrics yet",ha="center",va="center",transform=ax.transAxes)
        ax.set(title=title, xlabel="Optimizer update", ylabel="Masked L1 in normalized action units")
        ax.grid(alpha=.2)
    fig.suptitle("USB real-robot training logs", fontsize=16)
    fig.supxlabel("Faint: recorded batch loss. Solid: trailing mean of up to 20 logged points.\n"
                  "Different data, labels and normalization across panels; these are not validation losses.",fontsize=9)
    fig.savefig(output / "loss_curves.png", dpi=150)
    fig.savefig(output / "loss_curves.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    for ax,mode in zip(axes,("delta","relative")):
        pair=[next((c for c in cases if c["record_type"]=="production" and c["mode"]==mode and c["family"]==group),None)
              for group in ("go_original","go_ft500")]
        if all(c and c["metric_history"] for c in pair):
            end=min(c["metric_history"][-1]["step"] for c in pair)
            for c,color,label in zip(pair,("#6b7280","#1764ab"),("No FT history","GRU nominal 500 Hz")):
                rows=[p for p in c["metric_history"] if p["step"]<=end]
                y=np.array([p["loss"] for p in rows])
                smoothed=[y[max(i-19,0):i+1].mean() for i in range(len(y))]
                ax.plot([p["step"] for p in rows],smoothed,color=color,label=label,lw=1.6)
            ax.set_yscale("log")
            ax.legend(fontsize=9)
        ax.set(title=f"GO original gripper / {mode.capitalize()}",xlabel="Optimizer update (matched progress)",ylabel="Training masked L1")
        ax.grid(alpha=.2)
    fig.suptitle("GRU versus no FT history at the same training progress")
    fig.supxlabel("Same data, labels and optimizer settings. Trailing mean of 20 logged points; not held-out performance.",fontsize=9)
    fig.savefig(output/"matched_loss_curves.png",dpi=160)
    fig.savefig(output/"matched_loss_curves.pdf")
    plt.close(fig)


def fmt(value, digits=5):
    if value is None:
        return "—"
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def link(output, target, label):
    return f'<a href="{escape(os.path.relpath(target, output), quote=True)}">{escape(label)}</a>'


def render_html(cases, exports, supplementary, output, captured):
    primary = [c for c in cases if c["record_type"] == "production"]
    counts = Counter(c["status"] for c in primary)
    blocks = [f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>USB 모델 학습 로그와 설정</title><style>
body{{font-family:system-ui,"Noto Sans CJK KR",sans-serif;color:#192333;background:#f5f7fa;margin:0;line-height:1.65}}
main{{max-width:1400px;margin:auto;padding:30px 24px}}h1{{font-size:30px}}h2{{margin-top:36px}}
section,.box{{background:white;border:1px solid #dde3ec;border-radius:10px;padding:20px;margin:20px 0}}
.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #e5e9ef;padding:10px;text-align:left;vertical-align:top}}
th{{background:#edf2f7;white-space:nowrap}}a{{color:#1258a5}}code,pre{{font-family:monospace}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;padding:12px;font-size:12px}}
.muted{{color:#596579;font-size:13px}}.pill{{display:inline-block;padding:4px 12px;border-radius:18px;background:#eaf0f8;margin:4px}}
img{{width:100%;height:auto}}details{{margin:10px 0}}summary{{cursor:pointer}}input{{padding:10px;width:min(500px,90%);border:1px solid #b9c7d7;border-radius:5px}}
@media print{{main{{padding:0}}section{{break-inside:avoid}}input{{display:none}}}}
</style><main><h1>USB 모델 학습 로그와 설정</h1>
<p>기준 시각: <strong>{escape(captured)}</strong>. FF USB v2와 GO 실물 실험의 본학습 {len(primary)}개를 정리했습니다.
완료 {counts['completed']}개, 진행 {counts['running']}개, 대기 {counts['queued']}개, 취소 {counts['cancelled']}개입니다.
검증용 짧은 학습과 이전 중단·일시 정지 기록은 별도 목록에 있습니다.</p>
<p>이 보고서는 기준 시각의 스냅샷입니다. 각 모델의 <b>실시간 원본 폴더</b> 링크는 이후의 로그와 체크포인트를 계속 가리킵니다.
원본 로그·데이터·학습 세션은 변경하지 않았고 체크포인트 바이너리는 복제하지 않았습니다.</p>
<p>{link(output,output/'models.csv','모델 설정 CSV')} · {link(output,output/'catalog.json','전체 JSON')} ·
{link(output,output/'all_metrics.csv','기록된 전체 손실 CSV')} · {link(output,output/'loss_curves.pdf','손실 그래프 PDF')} ·
{link(output,output/'replay_loss_comparison.csv','팔과 그리퍼 재생 로스 CSV')} · {link(output,output/'matched_training_loss.csv','같은 단계 GRU 비교 CSV')} ·
{link(output,output/'logs_and_settings.zip','로그와 설정 ZIP')}</p>
<section><h2>모델별 현황</h2><input id="filter" placeholder="모델명, Delta, Relative, GRU 등으로 검색"><div class="table"><table>
<thead><tr><th>ID</th><th>실험</th><th>모드</th><th>상태</th><th>기록 step / 목표</th><th>마지막 배치 loss</th><th>언어 / FT history</th><th>로그</th></tr></thead><tbody>''']
    for c in primary:
        ft = f"GRU {c['nominal_ft_hz']}Hz" if c["use_ft_history"] else "history 미사용"
        local = Path(c["report_dir"])
        blocks.append(f'<tr class="data-row"><td><a href="#{c["id"]}">{c["id"]}</a></td><td>{escape(c["label"])}</td>'
                      f'<td>{c["mode"]}</td><td>{STATUS.get(c["status"],c["status"])}</td><td>{c["step"]:,} / {c["target_steps"]:,}</td>'
                      f'<td>{fmt(c["latest_logged_loss"])}</td><td>{"사용" if c["use_language"] else "미사용"} / {ft}</td>'
                      f'<td>{link(output,local/"snapshots","스냅샷")} · {link(output,local/"live","실시간 원본 폴더")}</td></tr>')
    blocks.append('''</tbody></table></div><p class="muted">완료 모델의 loss는 마지막으로 기록된 학습 배치의 값입니다.
최고 성능 체크포인트나 검증 오차를 뜻하지 않습니다. 진행 모델의 로그 step과 저장 체크포인트 step은 다를 수 있습니다.</p></section>
<section><h2>공통 구조와 실험 간 차이</h2>
<p>모든 모델은 현재 head·right wrist RGB 두 장(224×224)과 표준화된 19차원 state를 입력으로 받아,
3층 ACT decoder에서 50×7 액션 chunk를 예측합니다. 액션 시간 간격은 0.1초(10Hz)입니다.
19차원 state에는 현재 force·torque 6축이 포함됩니다. FT history 미사용 모델도 현재 FT는 사용합니다.</p>
<p>시각 backbone은 학습 가능한 DINOv3 ViT-B/16입니다. FF USB v2는 언어 경로를 제거한 약 9,219만 파라미터 모델이며,
GO는 frozen BERT와 6층 vision-language interaction을 포함한 216,085,805 파라미터 모델입니다.
GRU를 추가한 GO 모델은 216,171,565 파라미터입니다. GO task instruction은
<code>Insert the USB connector into the USB port.</code>입니다.</p>
<div class="table"><table><tr><th>구분</th><th>FF USB v2</th><th>GO 계열</th></tr>
<tr><td>데이터</td><td>51 episodes / 10,379 frames</td><td>50 episodes / 10,090 frames</td></tr>
<tr><td>예산</td><td>799,631 observations / 10,016 updates (고정 노출량 약 77.04 epochs)</td><td>807,200 observations / 12,640 updates / 80 epochs</td></tr>
<tr><td>Batch / workers</td><td>80 / 4</td><td>64 / 2</td></tr>
<tr><td>액션 정규화</td><td>symlog, temperature 20</td><td>symlog_minmax, temperature 20와 저장된 축별 scale·offset</td></tr>
<tr><td>증강</td><td>컬러지터 없음</td><td>brightness·contrast·saturation 0.2, hue 0.02; 상하 28픽셀 여백 제외, 두 카메라에 동일 변환</td></tr>
</table></div>
<p>공통: seed 42, AdamW, policy LR 5e-5 / vision LR 1e-5, betas (0.9, 0.95), weight decay 1e-6,
warmup 8,000 observations, cosine decay 하한 1%, BF16 AMP, gradient clipping 1.0,
EMA 0.999 per 64 observations. 손실은 유효 액션 chunk에 대한 masked L1이며, padding은 제외합니다.
RGB mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]를 사용합니다.</p>
<p>그리퍼 리라벨은 설정 파일의 <code>data.gripper_relabel</code>만으로 판별하지 않았습니다.
GO 리라벨 데이터는 HDF5에 이미 -1/+1 목표가 저장되어 있어 loader 설정은 null입니다.
FF USB v2의 리라벨은 loader에서 ±0.05 임계값으로 적용하며, GO 저장 리라벨은 ±0.5 임계값입니다.</p>
<p>GRU는 정규화된 FT → 단방향 1층 GRU(hidden 128) → Linear + LayerNorm → 256차원 ACT memory token으로 동작합니다.
각 입력 창마다 hidden state를 초기화합니다. 명목 500Hz는 원본 50개, 100Hz는 같은 명목 0.1초 구간의 10개를 사용합니다.
100Hz는 causal 21-tap Hamming FIR, cutoff 40Hz, 5배 다운샘플링(명목 group delay 20ms)이며,
변환 후 mean/std를 다시 구하고 std floor 0.01을 적용합니다. 샘플별 timestamp가 없어 FT 주파수는 수집 설정에 기반한 가정입니다.</p>
</section><section><h2>학습 손실</h2><img src="loss_curves.png" alt="모델 그룹별 학습 손실">
<p>옅은 선은 기록된 배치 loss, 진한 선은 최근 최대 20개 로그 지점의 평균입니다.
서로 다른 데이터·그리퍼 라벨·정규화 설정의 loss를 직접 성능 순위로 비교하지 않습니다.</p></section>''')
    blocks.append('<section><h2>같은 학습 단계에서 GRU 비교</h2><p>GO 원본 그리퍼 모델끼리 같은 optimizer step의 '
                  '최근 최대 20개 로그 지점을 맞췄습니다. 데이터·그리퍼 라벨·optimizer 설정은 같습니다. '
                  '아직 진행 중인 GRU 모델을 완료된 모델의 마지막 step과 비교하지 않습니다. 난수 seed는 같지만 stochastic augmentation·dropout draw까지 같다는 뜻은 아닙니다.</p>'
                  '<img src="matched_loss_curves.png" alt="동일 학습 단계의 GRU와 baseline 손실 비교"><div class="table"><table>'
                  '<tr><th>모드</th><th>공통 step 구간</th><th>로그 수</th><th>GRU 없음 평균 L1</th><th>500Hz GRU 평균 L1</th><th>GRU 변화율</th></tr>')
    for row in aligned_training_losses(cases):
        blocks.append(f'<tr><td>{row["mode"]}</td><td>{row["from_step"]:,}–{row["to_step"]:,}</td><td>{row["logged_points"]}</td>'
                      f'<td>{row["baseline_mean_l1"]:.6f}</td><td>{row["gru_mean_l1"]:.6f}</td><td>{row["relative_change_percent"]:+.2f}%</td></tr>')
    blocks.append('</table></div><p>음수는 이 구간의 GRU 학습 로스가 낮다는 뜻입니다. 학습 중간의 배치 통계이며 최종 성능 결론은 아닙니다.</p></section>')
    blocks.append('<section><h2>전체 학습 데모 재생 로스와 팔 동작 비교</h2><p>저장된 RAW·EMA 추론 액션을 해당 모델의 정규화 공간으로 되돌려 '
                  '모든 유효 chunk의 masked L1을 계산했습니다. float32 물리 액션에서 재구성한 근사값이며, jitter·dropout이 없는 평가 모드 결과입니다. '
                  '새 GPU 추론을 실행하지 않았습니다. 총 로스 = (6 × 팔 로스 + 그리퍼 로스) / 7입니다. '
                  '리라벨 유무가 다르면 그리퍼 목표도 다르므로 팔 6축의 공통 목표 로스를 별도로 보십시오.</p><div class="table"><table>'
                  '<tr><th>ID / 모드</th><th>라벨</th><th>가중치</th><th>총 L1</th><th>팔 6축 L1</th><th>그리퍼 L1</th><th>유효 chunk</th></tr>')
    for c in primary:
        for weights,loss in c["cached_replay_losses"].items():
            blocks.append(f'<tr><td>{c["id"]} / {c["mode"]}</td><td>{escape(c["label"])}</td><td>{weights.upper()}</td>'
                          f'<td>{loss["total_l1"]:.6f}</td><td>{loss["arm_l1"]:.6f}</td><td>{loss["gripper_l1"]:.6f}</td><td>{loss["valid_chunk_entries"]:,}</td></tr>')
    blocks.append('</table></div><p>GO 리라벨 모델과 원본 그리퍼 모델의 팔 목표는 모드별로 SHA256까지 같음을 확인했습니다. '
                  '완료된 전체 데모 재생 파일이 있는 모델만 포함했습니다. '
                  '위 재생 로스도 학습 데이터의 적합도이며 검증·테스트 로스가 아닙니다.</p></section>')
    blocks.append('<section><h2>저장된 재생 평가</h2><p>모두 학습 데모를 재생한 결과이며 held-out 평가나 자율 삽입 성공률이 아닙니다. '
                  'RAW와 EMA는 같은 checkpoint의 두 가중치입니다. 평가가 없는 모델은 빈칸으로 남겼습니다.</p><div class="table"><table>'
                  '<tr><th>ID / 모드</th><th>가중치 / step</th><th>위치 평균 mm</th><th>회전 평균 deg</th><th>그리퍼 MAE</th><th>그리퍼 이벤트 F1</th><th>평가 원본</th></tr>')
    for c in primary:
        for weights, evaluation in c["evaluations"].items():
            score = evaluation["first_step"]
            blocks.append(f'<tr><td>{c["id"]} / {c["mode"]}</td><td>{weights.upper()} / {evaluation["checkpoint_step"]:,}</td>'
                          f'<td>{fmt(score["position_mm"]["mean"])}</td><td>{fmt(score["rotation_deg"]["mean"])}</td>'
                          f'<td>{fmt(score["gripper"]["mae"])}</td><td>{fmt(score["gripper"].get("event_f1"))}</td>'
                          f'<td>{link(output,Path(c["artifact_dir"]),"평가 폴더")}</td></tr>')
    blocks.append('</table></div><p>GO 원본 그리퍼의 GRU 없는 두 모델은 RAW·EMA 재생 평가에서 모두 그리퍼 이벤트 F1=0입니다. '
                  '100개의 close/open 목표 이벤트에 대해 검출 이벤트가 없으며, 약 99%의 전체 정확도는 hold 라벨이 많은 영향입니다. '
                  '리라벨 모델의 그리퍼 state 정확도와 원본 명령의 이벤트 F1은 다른 지표입니다.</p></section>')
    for c in cases:
        destination = Path(c["report_dir"])
        ft = c["ft_history_config"]
        rows = {
            "상태 / 종류": f'{STATUS.get(c["status"],c["status"])} / {c["record_type"]}',
            "물리 데이터 경로": c["physical_dataset_dir"], "학습 데이터 adapter 경로": c["dataset_dir"],
            "그리퍼 라벨 / 적용 위치": f'{c["gripper"]} / {c["gripper_source"]}',
            "언어 / 현재 FT / FT history": f'{c["use_language"]} / True / {c["use_ft_history"]}',
            "FT history shape": c["ft_history_shape"], "GPU index / UUID": f'{c["gpu_index"]} / {c["gpu_uuid"]}',
            "학습 step / 목표": f'{c["step"]} / {c["target_steps"]}',
            "최신 로그 step / 배치 loss": f'{c["latest_logged_step"]} / {fmt(c["latest_logged_loss"])}',
            "최근 최대 20개 로그 loss 평균": fmt(c["recent_20_logged_losses_mean"]),
            "누적 observation / 목표": f'{c["samples_seen"]} / {c["target_samples"]}',
            "학습 loop 시간 / ETA": f'{c["elapsed_seconds"]/3600:.3f} h / {c["eta"] or "—"}',
            "저장 checkpoint step": c["checkpoint_latest_step"], "파라미터 수": c["parameters"],
        }
        blocks.append(f'<section id="{c["id"]}"><h2>{c["id"]} {escape(c["label"])} {c["mode"]}</h2><div class="table"><table>')
        blocks.extend(f'<tr><th>{escape(key)}</th><td>{escape(str(value))}</td></tr>' for key,value in rows.items())
        blocks.append('</table></div><p>' + link(output,destination/"snapshots","복사한 로그와 설정") + ' · ' +
                      link(output,destination/"live","실시간 원본 폴더") + ' · ' +
                      link(output,destination/"settings_and_status.json","구조화된 설정과 상태") + '</p>')
        if c["record_type"]=="pause_snapshot":
            blocks.append('<p>이 항목의 failed 상태와 step 7,426은 과거 중단 당시의 기록입니다. '
                          '보관된 checkpoint는 step 7,000이며, 이후 원래 본학습을 재개해 step 10,016까지 완료했습니다. '
                          '현재 완료 모델의 실패를 뜻하지 않습니다.</p>')
        if c["checkpoint_latest"]:
            blocks.append('<p>스냅샷 시점의 최신 저장 체크포인트: ' + link(output,Path(c["checkpoint_latest"]),f'step {c["checkpoint_latest_step"]}') + '</p>')
        blocks.append('<details><summary>전체 학습 설정 JSON</summary><pre>' + escape(json.dumps(c["config"],ensure_ascii=False,indent=2)) + '</pre></details></section>')
    blocks.append('<section><h2>배포본과 학습 최종본</h2><p>배포 manifest가 가리키는 step과 최종 학습 step을 분리했습니다. '
                  'FF USB v2 latched 배포본은 step 7,000이며, 이후 학습은 step 10,016까지 완료됐습니다.</p><ul>')
    for item in exports:
        blocks.append(f'<li>{escape(item["name"])}: step {item["step"]:,} · {link(output,Path(item["manifest"]),"배포 manifest")}</li>')
    blocks.append('</ul><h2>검증 및 보관 기록</h2><p>중단 시도와 일시 정지 checkpoint는 A로 시작하는 모델 ID로 기록했습니다. '
                  '아래 짧은 학습들은 GPU 검증용이며 본학습 모델 수에 포함하지 않습니다.</p><ul>')
    for item in supplementary:
        blocks.append(f'<li>{escape(item["label"])}: {escape(item["status"])} / step {item["step"]} · '
                      f'{link(output,Path(item["source"]),"원본 상태")}</li>')
    blocks.append('''</ul></section><section><h2>파일과 새 스냅샷 생성</h2>
<p><code>models.csv</code>는 모델당 한 행, <code>all_metrics.csv</code>는 기록된 학습 지점당 한 행입니다.
<code>catalog.json</code>에는 전체 모델 설정·평가·체크포인트 경로가 있습니다.
<code>snapshot_manifest.json</code>은 복사한 파일의 원본 경로·크기·SHA256을 기록합니다.
<code>logs_and_settings.zip</code>에는 텍스트 로그·설정·표·손실 그래프가 포함됩니다.
실시간 링크와 checkpoint 바이너리는 ZIP에 포함하지 않습니다.</p>
<pre>USE_TF=0 /home/jack/miniforge3/envs/turbovla-libero/bin/python scripts/real_robot/summarize_runs.py</pre>
<p>명령을 다시 실행하면 새 시각의 보고서를 별도 폴더에 생성합니다. 기존 보고서와 로그는 보존합니다.</p></section>
<script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();
document.querySelectorAll('.data-row').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></main></html>''')
    (output / "index.html").write_text("\n".join(blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    captured = datetime.now(ZoneInfo("Asia/Seoul"))
    output = (args.output or ROOT / "outputs/training_log_catalog" / captured.strftime("%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    (output / "containers").mkdir()
    jobs, containers = discover()
    manifest, cases, supplementary = [], [], []
    counters = Counter()
    for folder, kind, inherited in jobs:
        prefix = "M" if kind == "production" else "A"
        counters[prefix] += 1
        model_id = f"{prefix}{counters[prefix]:02d}"
        destination = output / "runs" / f"{model_id}_{family(folder)}_{folder.name}"
        cases.append(capture_case(folder, kind, inherited, destination, model_id, manifest))
    for i, container in enumerate(containers):
        destination = output / "containers" / f"{i+1:02d}_{container.name}"
        destination.mkdir()
        for pattern in ("launch*.json", "*status.json", "schedule.json", "queue_*.json", "training_controls.json",
                        "software_validation.json", "*queue.log", "*supervisor.log", "pause_request.json", "resume_supervisor.log"):
            for source in sorted(container.glob(pattern)):
                if source.is_file() and not (destination / source.name).exists():
                    snapshot_file(source,destination/source.name,manifest)
    preflights = [ROOT / f"outputs/{name}" for name in ("ff_usb_v2_preflight_20261015", "go_preflight_20261018",
                                                        "go_no_gripper_relabel_preflight_20261018")]
    preflights.extend(p for root in containers for p in root.glob("ft_*hz/preflight"))
    for root in preflights:
        for p in sorted(root.glob("smoke_*/training_status.json")):
            status = read_json(p)
            supplementary.append(dict(label=f'{root.parent.name}/{root.name}/{p.parent.name}', source=str(p),
                                      status=status["status"], step=status["step"]))
    exports = [dict(name=p.parent.name,manifest=str(p),**read_json(p)) for p in sorted((ROOT / "deploy").glob("*/manifest.json"))]
    draw_curves(cases,output)
    for mode in ("delta","relative"):
        pair=[next((c for c in cases if c["record_type"]=="production" and c["family"]==group and c["mode"]==mode),None)
              for group in ("go_latched","go_original")]
        if all(pair):
            for weights in ("raw","ema"):
                if all(weights in c["cached_replay_losses"] for c in pair):
                    assert pair[0]["cached_replay_losses"][weights]["arm_target_sha256"]==pair[1]["cached_replay_losses"][weights]["arm_target_sha256"]
    with (output/"replay_loss_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=("id","family","mode","weights","total_l1","arm_l1","gripper_l1","valid_chunk_entries","arm_target_sha256","source_npz")
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cases:
            for weights,loss in c["cached_replay_losses"].items():
                writer.writerow(dict(id=c["id"],family=c["family"],mode=c["mode"],weights=weights,
                                     **{key:loss[key] for key in fields[4:]}))
    aligned=aligned_training_losses(cases)
    if aligned:
        with (output/"matched_training_loss.csv").open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=list(aligned[0]));writer.writeheader();writer.writerows(aligned)
    fields = ("id","record_type","family","mode","status","phase","gpu_index","use_language","gripper","gripper_source",
              "current_ft_in_19d_state","use_ft_history","nominal_ft_hz","episodes","frames","step","target_steps",
              "samples_seen","target_samples","target_epochs","effective_epochs","latest_logged_step","latest_logged_loss",
              "recent_20_logged_losses_mean","elapsed_seconds","eta","parameters","action_hz","action_normalization",
              "checkpoint_latest_step","checkpoint_latest","physical_dataset_dir","dataset_dir","source_dir")
    with (output / "models.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(fields)+["batch_size","learning_rate","vision_learning_rate","color_jitter","ft_history_shape"])
        writer.writeheader()
        for c in cases:
            row={key:c.get(key) for key in fields}
            row.update({key:c["train"][key] for key in ("batch_size","learning_rate","vision_learning_rate")})
            row.update(color_jitter=bool(c["images"].get("color_jitter")),ft_history_shape=c["ft_history_shape"])
            writer.writerow(row)
    with (output / "all_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=("model_id","family","mode","step","samples_seen","loss","gradient_norm","elapsed_seconds","policy_lr","vision_lr","eta")
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cases:
            for point in c["metric_history"]:
                row={key:point.get(key) for key in fields if key not in ("model_id","family","mode","policy_lr","vision_lr")}
                row.update(model_id=c["id"],family=c["family"],mode=c["mode"],policy_lr=point.get("learning_rates",{}).get("policy"),
                           vision_lr=point.get("learning_rates",{}).get("vision"))
                writer.writerow(row)
    catalog=dict(captured_at=captured.isoformat(),scope="TurboVLA USB real-robot training",counts=dict(Counter(c["status"] for c in cases if c["record_type"]=="production")),
                 models=[{k:v for k,v in c.items() if k!="metric_history"} for c in cases],deployments=exports,supplementary=supplementary,
                 matched_training_loss=aligned,go_arm_targets_identical_across_gripper_labels=True)
    write_json(output / "catalog.json",catalog)
    write_json(output / "snapshot_manifest.json",manifest)
    render_html(cases,exports,supplementary,output,captured.isoformat(timespec="seconds"))
    (output / "LIVE_LOG_PATHS.txt").write_text("\n".join(f'{c["id"]}\t{c["source_dir"]}' for c in cases)+"\n")
    with zipfile.ZipFile(output / "logs_and_settings.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.is_symlink() and "live" not in path.relative_to(output).parts and path.name!="logs_and_settings.zip":
                archive.write(path,path.relative_to(output))
    print(json.dumps(dict(output=str(output),counts=catalog["counts"],archive_records=len(cases)-counters["M"],snapshot_files=len(manifest)),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
