"""Atomic file receiver with independent SHA-256 comparison over SSH/zstd."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path


def receive(host: str, root: str, original: dict, destination: Path, zstd: str, level: int) -> tuple[dict, str]:
    remote = shlex.join(
        [
            "python3",
            "-",
            "--root",
            root,
            "--file",
            original["path"],
            "--bytes",
            str(original["bytes"]),
            "--mtime-ns",
            str(original["mtime_ns"]),
            "--level",
            str(level),
        ]
    )
    helper = Path(__file__).with_name("stream_source.py").read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    ssh = decoder = None
    try:
        with tempfile.TemporaryFile() as diagnostics:
            ssh = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, remote],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
            )
            ssh.stdin.write(helper)
            ssh.stdin.close()
            decoder = subprocess.Popen(
                [zstd, "-q", "-d", "-c"], stdin=ssh.stdout, stdout=subprocess.PIPE, stderr=diagnostics
            )
            ssh.stdout.close()
            digest = hashlib.sha256()
            written = 0
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.", suffix=".sfm-partial", dir=destination.parent, delete=False
            ) as output:
                temp_path = Path(output.name)
                for block in iter(lambda: decoder.stdout.read(1024 * 1024), b""):
                    written += len(block)
                    if written > original["bytes"]:
                        raise RuntimeError("Stream exceeds inventoried file size")
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            decoder_code, ssh_code = decoder.wait(), ssh.wait()
            diagnostics.seek(0)
            message = diagnostics.read().decode("utf-8", errors="replace")
            if decoder_code or ssh_code:
                raise RuntimeError(f"Transfer failed (ssh={ssh_code}, zstd={decoder_code}): {message[-2000:]}")
            rows = [
                json.loads(line[len("SFM_SHA256 ") :])
                for line in message.splitlines()
                if line.startswith("SFM_SHA256 ")
            ]
            if len(rows) != 1:
                raise RuntimeError(f"Missing source SHA-256 record: {message[-2000:]}")
            source = rows[0]
            actual_hash = digest.hexdigest()
            if (
                source["path"] != original["path"]
                or written != original["bytes"]
                or source["bytes"] != written
                or source["mtime_ns"] != original["mtime_ns"]
                or source["sha256"] != actual_hash
            ):
                raise RuntimeError(f"Source/local size or SHA-256 mismatch: {original['path']}")
            os.chmod(temp_path, source["mode"])
            os.utime(temp_path, ns=(source["mtime_ns"], source["mtime_ns"]))
            temp_path.replace(destination)
            temp_path = None
            return source, actual_hash
    finally:
        for process in (decoder, ssh):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
