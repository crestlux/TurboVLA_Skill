#!/usr/bin/env python3
"""Snapshot and launch training/evaluation; default to Delta then Relative on GPU 1."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
FM_EVALUATION = Path("/home/jack/practice/RepresentationFM/results/ff_usb_v2_trajectory_20261015_044346")


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def worker(run, mode, gpu):
    output, code = run / mode, run / "code"
    output.mkdir(exist_ok=True)
    config = code / "experiments/real_robot/configs" / f"ff_usb_v2_{mode}.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", WANDB_MODE="disabled",
               PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(code))
    status = {"status": "running", "phase": "training", "mode": mode, "gpu_uuid": gpu, "started_at": now(),
              "supervisor_pid": os.getpid()}
    path = output / "job_status.json"
    write_json(path, status)

    def execute(command, log_name):
        with (output / log_name).open("a", buffering=1) as log:
            child = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            status.update(child_pid=child.pid, command=command)
            write_json(path, status)
            code_number = child.wait()
        if code_number:
            raise RuntimeError(f"{status['phase']} exited with code {code_number}; see {log_name}")

    try:
        execute([sys.executable, "scripts/real_robot/train.py", "--config", str(config), "--output", str(output)], "train.log")
        training = json.loads((output / "training_status.json").read_text())
        final = output / f"checkpoint_{training['target_steps']:06d}.pt"
        if training["status"] != "completed" or training["samples_seen"] != training["target_samples"]:
            raise RuntimeError("Training exited without consuming the configured sample budget")
        if not final.is_file() or not final.samefile(output / "latest.pt"):
            raise RuntimeError("Final checkpoint does not match latest.pt")
        status.update(phase="trajectory_evaluation", training_completed_at=now(), final_step=training["step"],
                      samples_seen=training["samples_seen"])
        write_json(path, status)
        execute([sys.executable, "scripts/real_robot/evaluate.py", "--checkpoint", str(final),
                 "--output", str(run / "evaluation")], "evaluation.log")
        status.update(status="completed", phase="evaluation_complete", finished_at=now())
        write_json(path, status)
    except BaseException as error:
        status.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
        write_json(path, status)
        raise


def coordinate(run):
    status_path = run / "pipeline_status.json"
    status = {"status": "running", "phase": "training_and_evaluation", "started_at": now()}
    write_json(status_path, status)
    try:
        while True:
            jobs = {}
            for mode in ("relative", "delta"):
                path = run / mode / "job_status.json"
                jobs[mode] = json.loads(path.read_text()) if path.exists() else {"status": "starting"}
            status.update(jobs=jobs, checked_at=now())
            write_json(status_path, status)
            if any(job["status"] == "failed" for job in jobs.values()):
                raise RuntimeError("A worker failed; inspect its job_status.json and logs")
            if all(job["status"] == "completed" for job in jobs.values()):
                break
            time.sleep(15)
        status.update(phase="plotting_and_comparison")
        write_json(status_path, status)
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", USE_TF="0", HF_HUB_OFFLINE="1",
                   PYTHONPATH=str(run / "code"), PYTHONNOUSERSITE="1")
        with (run / "evaluation/plot.log").open("w") as log:
            subprocess.run([sys.executable, "scripts/real_robot/plot.py", "--directory", str(run / "evaluation")],
                           cwd=run / "code", env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, "scripts/real_robot/compare_models.py", "--turbovla", str(run / "evaluation"),
                            "--representationfm", str(FM_EVALUATION)], cwd=run / "code", env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        status.update(status="completed", phase="comparison_ready", finished_at=now(),
                      results_directory=str(run / "evaluation"))
        write_json(status_path, status)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
        write_json(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/jack/turbovla_ff_usb_v2_runs"))
    parser.add_argument("--preflight", type=Path, default=ROOT / "outputs/ff_usb_v2_preflight_20261015")
    parser.add_argument("--gpus", nargs="+", default=["1"],
                        help="One GPU for sequential jobs, or two distinct GPUs for parallel jobs")
    parser.add_argument("--order", nargs=2, choices=["relative", "delta"], default=["delta", "relative"])
    parser.add_argument("--run", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", choices=["relative", "delta"], help=argparse.SUPPRESS)
    parser.add_argument("--gpu", help=argparse.SUPPRESS)
    parser.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.run, args.worker, args.gpu)
        return
    if args.coordinator:
        coordinate(args.run)
        return
    if len(args.gpus) not in (1, 2) or len(set(args.gpus)) != len(args.gpus) or shutil.which("tmux") is None:
        raise RuntimeError("One GPU or two distinct GPU indices, and tmux, are required")
    if len(set(args.order)) != 2:
        raise RuntimeError("Both action modes must occur exactly once in --order")
    # Refuse to start long jobs with code that differs from the verified smoke.
    for mode in ("relative", "delta"):
        report = json.loads((args.preflight / f"verified_{mode}.json").read_text())
        if not report["passed"]:
            raise RuntimeError("Preflight did not pass")
        for name, digest in report["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Source changed since preflight: {name}")
        if not (args.preflight / "evaluation" / f"{mode}_ema.json").is_file():
            raise RuntimeError("Run raw/EMA trajectory evaluation preflight first")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    gpus = {parts[0]: {"uuid": parts[1], "memory_mib": int(parts[2])}
            for line in devices.splitlines() if (parts := [x.strip() for x in line.split(",")])}
    for gpu in args.gpus:
        if gpu not in gpus or gpus[gpu]["memory_mib"] > 512:
            raise RuntimeError(f"GPU {gpu} is unavailable or in use")
    for mode in ("relative", "delta"):
        for weights in ("raw", "ema"):
            if not (FM_EVALUATION / f"{mode}_{weights}.json").is_file():
                raise RuntimeError("RepresentationFM comparison results are missing")
    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    run = args.output_root.resolve() / f"ff_usb_v2_turbovla_image_state_{run_id}"
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    code = run / "code"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("turbovla", "scripts/real_robot", "experiments/real_robot/configs"):
        shutil.copytree(ROOT / name, code / name, ignore=ignore)
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(ROOT / name, code / name)
    audit = run / "preflight"
    audit.mkdir()
    for path in args.preflight.glob("*.json"):
        shutil.copy2(path, audit / path.name)
    manifest = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(code.rglob("*")) if p.is_file()}
    write_json(run / "source_manifest.json", manifest)
    launch = {"created_at": now(), "run_directory": str(run), "code_snapshot": str(code), "python": sys.executable,
              "automatic_trajectory_comparison": True, "representationfm_evaluation": str(FM_EVALUATION), "jobs": {},
              "execution": "sequential" if len(args.gpus) == 1 else "parallel", "execution_order": args.order}
    for index, mode in enumerate(args.order):
        gpu = args.gpus[index] if len(args.gpus) == 2 else args.gpus[0]
        launch["jobs"][mode] = {"gpu_index": gpu, "gpu_uuid": gpus[gpu]["uuid"], "tmux_session": f"turbo_usb_{mode}_{run_id}"}
        if len(args.gpus) == 1 and index:
            launch["jobs"][mode]["after"] = args.order[index - 1]
    launch["coordinator_session"] = f"turbo_usb_compare_{run_id}"
    write_json(run / "launch.json", launch)
    (run / "evaluation").mkdir()
    for mode, job in launch["jobs"].items():
        if "after" in job:
            command = [sys.executable, str(code / "scripts/real_robot/queue_after.py"), "--run", str(run),
                       "--mode", mode, "--after", job["after"], "--gpu", job["gpu_uuid"]]
        else:
            command = [sys.executable, str(code / "scripts/real_robot/launch.py"), "--run", str(run), "--worker", mode, "--gpu", job["gpu_uuid"]]
        subprocess.run(["tmux", "new-session", "-d", "-s", job["tmux_session"], "-c", str(code), shlex.join(command)], check=True)
    command = [sys.executable, str(code / "scripts/real_robot/launch.py"), "--run", str(run), "--coordinator"]
    subprocess.run(["tmux", "new-session", "-d", "-s", launch["coordinator_session"], "-c", str(code), shlex.join(command)], check=True)
    link = ROOT / "outputs" / run.name
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(run, target_is_directory=True)
    print(json.dumps(launch, indent=2), flush=True)


if __name__ == "__main__":
    main()
