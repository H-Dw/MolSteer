import json
from pathlib import Path


def test_new_skill_contract_and_example_are_available():
    root = Path(__file__).resolve().parents[1]
    skill = root / 'skills/molthinker-conflict-aware-control'
    english = (skill / 'SKILL.md').read_text(encoding='utf-8')
    chinese = (skill / 'SKILL.zh-CN.md').read_text(encoding='utf-8')
    example = json.loads((skill / 'references/decision.example.json').read_text(encoding='utf-8'))
    assert 'molthinker-conflict-aware-control' in english
    assert len(chinese) > 1000
    assert example['execution_authorized'] is False
    assert example['status'] != 'authorized'
    assert all((skill / 'references' / filename).is_file() for filename in
               ['optimization-contract.md', 'decision-contract.md'])


def test_repository_secret_exclusions():
    root = Path(__file__).resolve().parents[1]
    for name in ['.gitignore', '.dockerignore', '.claudeignore']:
        assert 'secrets.local.json' in (root / name).read_text(encoding='utf-8')
    settings = json.loads((root / '.claude/settings.json').read_text(encoding='utf-8'))
    assert 'Read(./configs/secrets.local.json)' in settings['permissions']['deny']
    # Never read the actual local credential file during tests.
