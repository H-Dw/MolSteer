"""Deliver existing workflow references without encoding scientific choices in Skills."""
from copy import deepcopy
import hashlib
from pathlib import Path
import re


def sections(text):
    matches = list(re.finditer(r'^## (.+)$', text, re.M))
    return {match.group(1).strip(): text[match.start():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
            for i, match in enumerate(matches)}


def load_workflow_materials(skill_path):
    """Load the configured document and its local Markdown references, within its directory."""
    skill_path = Path(skill_path).resolve(strict=True)
    root = skill_path.parent
    materials, pending = {}, [skill_path]
    while pending and len(materials) < 16:
        path = pending.pop(0).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.suffix != '.md':
            continue
        ident = path.relative_to(root).as_posix()
        if ident in materials:
            continue
        text = path.read_text(encoding='utf-8')
        materials[ident] = dict(reference_id=ident, sha256=hashlib.sha256(text.encode()).hexdigest(),
                               text=text, sections=sections(text))
        for link in re.findall(r'\]\(([^)]+)\)', text):
            relative = link.split('#', 1)[0]
            if relative and not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', relative):
                candidate = (path.parent / relative).resolve()
                if candidate.is_relative_to(root) and candidate.suffix == '.md':
                    pending.append(candidate)
    return materials


def reference_catalog(materials):
    return [{k: deepcopy(row[k]) for k in ('reference_id', 'sha256')} |
            {'sections': list(row['sections'])} for row in materials.values()]


def read_reference(materials, reference_id, section=None):
    row = materials.get(reference_id)
    if row is None or section is not None and section not in row['sections']:
        return dict(status='needs_input', blocking=False, references=reference_catalog(materials),
                    hint='Use a literal reference_id and optional section returned by the reference catalog.')
    return dict(status='reference', reference_id=reference_id, sha256=row['sha256'],
                text=row['sections'][section] if section is not None else row['text'],
                scope='Existing generic reference material; expert choices and execution requirements come from the current task and actual contracts.')


def initial_role_reference(materials, role):
    """Supply a small relevant source section; the rest remains readable on demand.

    File/section routing only selects background material. It provides no molecule,
    fixed target set, formula, numerical parameter, or ranking decision.
    """
    preferences = ([('SKILL.md', '1. Ground the decision')] if role == 'biology' else
                   [('references/knowledge-guided-composition.md', '1. Build a function-basis ledger')])
    for ident, title in preferences:
        if ident in materials and title in materials[ident]['sections']:
            return read_reference(materials, ident, title)
    if not materials:
        return {}
    first = next(iter(materials))
    row = materials[first]
    return read_reference(materials, first, next(iter(row['sections']), None))
