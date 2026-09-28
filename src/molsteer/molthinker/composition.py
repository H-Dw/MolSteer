"""Small declarative objective trees; no model-supplied code is evaluated."""
from __future__ import annotations

import math

import torch


_BRANCHES = {'maximum', 'mean', 'lp_norm'}


def validate_design_record(design: dict, terms: list[dict], candidate_ids: set[str] | None = None) -> None:
    """Check target selection, scale provenance and comparison without executing model text."""
    ids={t['term_id'] for t in terms}
    required={'target_groups','normalizations','omitted','objective_tree',
              'architecture_reason','rejected_alternatives','mathematical_audit','evaluation_plan'}
    if not isinstance(design,dict) or set(design)!=required or not ids:
        raise ValueError('Reward design fields are incomplete or unexpected')
    def explanation(value):
        return isinstance(value,str) and 12<=len(value.strip())<=2000
    if not explanation(design['architecture_reason']):
        raise ValueError('Explain the selected mathematical architecture')
    alternatives=design['rejected_alternatives']
    if not isinstance(alternatives,list) or not alternatives or any(not explanation(x) for x in alternatives):
        raise ValueError('Compare at least one rejected architecture')
    audit=design['mathematical_audit']
    if (not isinstance(audit,dict)
            or set(audit)!={'zero_set','marginal_sensitivity','constraint_and_gradient_path','failure_mode'}
            or any(not explanation(x) for x in audit.values())):
        raise ValueError('Record the objective zero set, sensitivities, live path and failure mode')
    evaluation=design['evaluation_plan']
    if (not isinstance(evaluation,dict)
            or set(evaluation)!={'independent_measurement','matched_native_status'}
            or any(not explanation(x) for x in evaluation.values())):
        raise ValueError('Record independent evaluation and native comparison availability')
    groups=design['target_groups']
    if not isinstance(groups,list) or not groups:
        raise ValueError('At least one core repair target is required')
    covered=[]
    for group in groups:
        if (not isinstance(group,dict) or set(group)!={'role','term_ids','repair_predicate','rationale','falsifier'}
                or group['role'] not in {'repair','preservation'} or not isinstance(group['term_ids'],list)
                or not group['term_ids'] or not all(isinstance(x,str) for x in group['term_ids'])
                or any(not explanation(group[k]) for k in ('repair_predicate','rationale','falsifier'))):
            raise ValueError('Invalid target group or missing repair/falsification reasoning')
        covered.extend(group['term_ids'])
    if not any(g['role']=='repair' for g in groups) or len(covered)!=len(set(covered)) or set(covered)!=ids:
        raise ValueError('Target groups must partition selected terms and include repair')
    normalizations=design['normalizations']
    if not isinstance(normalizations,list) or len(normalizations)!=len(ids):
        raise ValueError('Every selected term needs a declared normalization')
    scales={}
    for norm in normalizations:
        if (not isinstance(norm,dict) or set(norm)!={'term_id','scale','origin'}
                or not isinstance(norm['term_id'],str) or type(norm['scale']) not in (int,float)
                or not math.isfinite(norm['scale']) or not 1e-6<=norm['scale']<=1e6
                or not explanation(norm['origin']) or norm['term_id'] in scales):
            raise ValueError('Invalid normalization or scale origin')
        scales[norm['term_id']]=norm['scale']
    if set(scales)!=ids or any(t['weight']!=1 or t['scale']!=scales[t['term_id']] for t in terms):
        raise ValueError('Creative term scale or weight disagrees with design')
    omitted=design['omitted']
    if not isinstance(omitted,list):
        raise ValueError('Omitted candidate dispositions are required')
    omitted_ids=[]
    for item in omitted:
        if (not isinstance(item,dict) or set(item)!={'term_id','role','reason'}
                or not isinstance(item['term_id'],str)
                or item['role'] not in {'monitor','counterevidence','deferred'}
                or not explanation(item['reason'])):
            raise ValueError('Invalid omitted candidate disposition')
        omitted_ids.append(item['term_id'])
    if len(omitted_ids)!=len(set(omitted_ids)) or set(omitted_ids)&ids:
        raise ValueError('Omitted candidates overlap selected terms or repeat')
    if candidate_ids is not None and set(omitted_ids)!=candidate_ids-ids:
        raise ValueError('Every unused candidate needs one explicit disposition')
    validate_objective_tree(design['objective_tree'],ids)


def validate_objective_tree(tree: dict, term_ids: set[str]) -> None:
    """Require a bounded, monotone tree that uses each selected term once."""
    if not term_ids:
        raise ValueError('An objective tree needs selected terms')
    seen: list[str] = []
    count = 0

    def visit(node, depth):
        nonlocal count
        count += 1
        if depth > 6 or count > 64 or not isinstance(node, dict):
            raise ValueError('Objective tree is too deep, large or malformed')
        op = node.get('op')
        if op == 'term':
            if set(node) != {'op', 'term_id'} or node['term_id'] not in term_ids:
                raise ValueError('Objective tree has an unknown term or leaf field')
            seen.append(node['term_id'])
            return
        if op not in _BRANCHES:
            raise ValueError('Unsupported objective-tree operator')
        allowed = {'op', 'children', 'p'} if op == 'lp_norm' else {'op', 'children'}
        if set(node) != allowed or not isinstance(node['children'], list) or len(node['children']) < 2:
            raise ValueError('Objective-tree branch needs at least two children and only its declared fields')
        if op == 'lp_norm':
            p = node['p']
            if type(p) not in (int, float) or not math.isfinite(p) or not 1 <= p <= 8:
                raise ValueError('lp_norm exponent must be finite and in [1, 8]')
        for child in node['children']:
            visit(child, depth + 1)

    visit(tree, 0)
    if len(seen) != len(set(seen)) or set(seen) != term_ids:
        raise ValueError('Objective tree must use each selected term exactly once')


def objective_value(tree: dict, penalties: dict[str, torch.Tensor]) -> torch.Tensor:
    """Evaluate a validated tree of nonnegative dimensionless penalties."""
    if tree['op'] == 'term':
        return penalties[tree['term_id']]
    values = torch.stack([objective_value(child, penalties) for child in tree['children']])
    if tree['op'] == 'maximum':
        return values.max()
    if tree['op'] == 'mean':
        return values.mean()
    if tree['op'] == 'lp_norm':
        p = tree['p']
        if p == 1:
            return values.mean()
        epsilon = 1e-12
        return (values.pow(p).mean() + epsilon).pow(1 / p) - epsilon ** (1 / p)
    raise ValueError('Unsupported objective-tree operator')


__all__ = ['validate_design_record', 'validate_objective_tree', 'objective_value']
