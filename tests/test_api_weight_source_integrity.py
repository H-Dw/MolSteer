"""The live-test harness must detect changes to data and its copied inputs."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def harness(tmp_path):
    path = Path(__file__).resolve().parents[1]/'scripts/test_5i0b_api_weights.py'
    spec = importlib.util.spec_from_file_location('api_weight_harness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = tmp_path
    data = tmp_path/'data'
    data.mkdir()
    source = data/'pocket.pdb'
    source.write_bytes(b'original pocket')
    root = tmp_path/'test/one'
    (root/'inputs').mkdir(parents=True)
    copy = root/'inputs/pocket.pdb'
    copy.write_bytes(source.read_bytes())
    stat = source.stat()
    manifest = dict(data_files_before={'pocket.pdb':dict(
        size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=module.sha(source))},
        source_files=[dict(source=str(source), copy=str(copy), sha256=module.sha(source))])
    (root/'source_manifest.json').write_text(json.dumps(manifest))
    return module, root, source, copy


def test_audit_only_writes_test_outputs(harness):
    module, root, source, copy = harness
    before = (source.read_bytes(), source.stat().st_mtime_ns, copy.read_bytes())
    result = module.audit_sources(root, 'before')
    assert result['passed'] and result['data_unchanged']
    assert before == (source.read_bytes(), source.stat().st_mtime_ns, copy.read_bytes())
    assert (root/'source_integrity_before.json').is_file()


@pytest.mark.parametrize('change', ['data_content', 'data_added', 'data_deleted', 'copy_content'])
def test_audit_rejects_mutated_or_missing_input(harness, change):
    module, root, source, copy = harness
    if change == 'data_content':
        source.write_bytes(b'changed pocket')
    elif change == 'data_added':
        (source.parent/'extra.sdf').write_bytes(b'extra data')
    elif change == 'data_deleted':
        source.unlink()
    else:
        copy.write_bytes(b'changed copy')
    result = module.audit_sources(root, 'after')
    assert not result['passed']
    assert result['data_unchanged'] is (change == 'copy_content')
