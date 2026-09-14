"""Atomic files and local integrity checks for resumable training."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid4().hex[:8] + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if read_json(temporary) != value:
            raise ValueError('JSON roundtrip failed')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_files(root, mapping):
    root = Path(root).resolve()
    for relative, expected in mapping.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'File integrity check failed: {relative}')


@contextmanager
def run_lock(directory):
    path = Path(directory) / 'training.lock'
    try:
        handle = path.open('x', encoding='utf-8')
    except FileExistsError:
        raise ValueError(f'A training lock exists: {path}. If no training process remains, remove only this lock and rerun.') from None
    with handle:
        handle.write(f'process_id={os.getpid()}\n')
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
