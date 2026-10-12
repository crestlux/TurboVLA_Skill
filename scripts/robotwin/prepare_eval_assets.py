#!/usr/bin/env python3
"""Download pinned RoboTwin simulator assets, verify SHA256, then extract on HDD."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import hf_hub_download


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--robotwin-path', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.store / manifest['revision']
    archive_dir = root / 'archives'
    extracted = root / 'extracted'
    archive_dir.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        filename = entry['rfilename']
        expected = entry['lfs']['sha256']
        path = archive_dir / filename
        for attempt in range(1, 6):
            try:
                print(f'Downloading {filename}, {entry["size"]} bytes, attempt {attempt}', flush=True)
                hf_hub_download(manifest['repo_id'], filename, repo_type='dataset',
                                revision=manifest['revision'], local_dir=archive_dir,
                                endpoint='https://huggingface.co')
                assert path.stat().st_size == entry['size'], f'{filename}: wrong size'
                assert sha256(path) == expected, f'{filename}: wrong SHA256'
                print(f'Verified {filename}: {expected}', flush=True)
                return entry
            except Exception as exc:
                print(f'{filename}: {type(exc).__name__}: {exc}', flush=True)
                if attempt == 5:
                    raise
                time.sleep(10 * attempt)

    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = list(pool.map(fetch, sorted(manifest['assets'], key=lambda x: x['size'])))
    for entry in entries:
        marker = root / (entry['rfilename'] + '.extracted.json')
        if marker.exists():
            assert json.loads(marker.read_text())['sha256'] == entry['lfs']['sha256']
            continue
        with zipfile.ZipFile(archive_dir / entry['rfilename']) as archive:
            for info in archive.infolist():
                target = (extracted / info.filename).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise RuntimeError(f'Unsafe archive member: {info.filename}')
            total_bytes = sum(i.file_size for i in archive.infolist())
            print(f'Extracting {entry["rfilename"]}: {len(archive.infolist())} entries, {total_bytes} bytes', flush=True)
            archive.extractall(extracted)  # ZIP CRC checked while extracting each member.
        marker.write_text(json.dumps(dict(sha256=entry['lfs']['sha256'],
                                         extracted_bytes=total_bytes), indent=2) + '\n')
    for folder in ['objects', 'embodiments', 'background_texture']:
        target = extracted / folder
        assert target.is_dir(), (folder, list(extracted.iterdir()))
        link = args.robotwin_path / 'assets' / folder
        if link.is_symlink():
            assert link.resolve() == target.resolve(), link
        else:
            assert not link.exists(), f'Refusing to overwrite {link}'
            link.symlink_to(target, target_is_directory=True)
    ready = dict(status='ready', **manifest, extracted_root=str(extracted),
                 robotwin_path=str(args.robotwin_path), archives_sha256_verified=True,
                 zip_crc_verified=True)
    (root / 'READY.json').write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps(ready, indent=2), flush=True)


if __name__ == '__main__':
    main()
