#!/usr/bin/env python3
"""SSH helper: hash and compress one source file in a single read; write nothing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--file", required=True)
parser.add_argument("--bytes", type=int, required=True)
parser.add_argument("--mtime-ns", type=int, required=True)
parser.add_argument("--level", type=int, default=6)
args = parser.parse_args()
root = args.root.resolve()
path = root / args.file
assert not path.is_symlink() and path.resolve().is_relative_to(root)
before = path.stat()
assert (before.st_size, before.st_mtime_ns) == (args.bytes, args.mtime_ns), "Source changed since inventory"
digest = hashlib.sha256()
compressor = subprocess.Popen(["zstd", "-q", "-c", f"-{args.level}"], stdin=subprocess.PIPE, stdout=sys.stdout.buffer)
try:
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            compressor.stdin.write(block)
    compressor.stdin.close()
    assert compressor.wait() == 0, "Compression failed"
finally:
    if compressor.poll() is None:
        compressor.terminate()
        compressor.wait()
after = path.stat()
assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Source changed during transfer"
print(
    "SFM_SHA256 "
    + json.dumps(
        {
            "path": args.file,
            "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "mode": before.st_mode & 0o777,
            "sha256": digest.hexdigest(),
        }
    ),
    file=sys.stderr,
    flush=True,
)
