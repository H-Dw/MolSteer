"""Verify the audited GLM-only API workflow and matched 5i0b GPU continuations."""
import argparse
import json
from pathlib import Path
import shutil
from test_5i0b_api_weights import REPO, preflight, execute, audit_sources, sha, write
from molsteer.agents.config import load_config


def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if (not source.is_relative_to(REPO/'test') or not destination.is_relative_to(REPO/'test')
        or source == destination):
        raise ValueError('Copied experiment inputs must stay in distinct test directories')
    manifest = json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=True)
    for record in manifest['source_files']:
        folder = 'source_trajectory' if Path(record['copy']).suffix == '.pt' else 'inputs'
        original = source/folder/Path(record['copy']).name
        copied = destination/folder/original.name
        if sha(original) != record['sha256']:
            raise ValueError('Existing test source failed its recorded hash')
        copied.parent.mkdir(parents=True, exist_ok=True)
        if copied.exists():
            if sha(copied) != record['sha256']:
                raise ValueError('Destination input already exists with a different hash')
        else:
            shutil.copy2(original, copied)
    if (destination/'source_manifest.json').exists():
        if json.loads((destination/'source_manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Destination manifest differs from the selected source experiment')
    else:
        write(destination/'source_manifest.json', manifest)
    provenance_source=source/'generation_provenance.json'
    provenance_copy=destination/provenance_source.name
    if provenance_copy.exists():
        if sha(provenance_copy) != sha(provenance_source):
            raise ValueError('Existing generation provenance differs from the source experiment')
    else:
        shutil.copy2(provenance_source, provenance_copy)
    if not audit_sources(destination, 'before')['passed']:
        raise ValueError('Read-only source integrity audit failed')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-test-root', type=Path, default=REPO/'test/5i0b_t050_api_weights_20260930')
    parser.add_argument('--test-root', type=Path, default=REPO/'test/glm_design_audit_20260930')
    parser.add_argument('--agent-config', type=Path, default=REPO/'configs/agents.json')
    parser.add_argument('--attempt', default='glm_run_01')
    args = parser.parse_args()
    config = load_config(args.agent_config)
    if config.mode != 'api' or {p.model for p in config.models.values()} != {'z-ai/glm-5.3'}:
        raise ValueError('This verification requires actual API mode and GLM-5.3 in every model profile')
    if not config.thinker.require_design_audit:
        raise ValueError('GLM expert verification must enable scientific design audits')
    if Path(args.attempt).name != args.attempt or args.attempt in ('', '.', '..'):
        raise ValueError('Use one nonempty experiment attempt directory name')
    root = prepare(args.source_test_root, args.test_root)
    try:
        if preflight(root, args.agent_config)['status'] != 'ready':
            raise RuntimeError('GLM API experiment preflight is blocked; see readiness.json')
        execute(root, args.agent_config, args.attempt)
    finally:
        if not audit_sources(root, 'after')['passed']:
            raise ValueError('Source integrity changed during the experiment')


if __name__ == '__main__':
    main()
