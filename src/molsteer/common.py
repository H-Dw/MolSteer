import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fact(value=None, status='unavailable', source=None, reason=None):
    return dict(value=value, status=status, source=source, reason=reason)


def observation(packet, metric, view='prediction'):
    return next((m for m in packet['observations'] if m['view'] == view and m['metric_id'] == metric), None)
