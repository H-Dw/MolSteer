"""Translate an expert's ordinal value ranking into explicit control allocation."""
from copy import deepcopy
import math


def allocate_priorities(design, biology, decay=0.5):
    """Supply a configurable ordinal preference, never a fabricated benefit estimate.

    Explicit expert weights take precedence. Constraints have preservation roles;
    unavailable goals retain biological rank but receive no executable coefficient.
    """
    result = deepcopy(design)
    strategy = result['strategy']
    executable = {d['direction_id'] for d in result['directions'] if d['status'] == 'executable'}
    ordered = sorted((d for d in biology['directions'] if d['disposition'] == 'optimize'
                      and d['direction_id'] in executable), key=lambda d: d['rank'])
    if strategy['mode'] == 'design_only':
        return result
    if 'priority_weights' not in strategy:
        if len(ordered) < 2:
            return result
        strategy['priority_weights'] = {d['direction_id']: decay**i for i, d in enumerate(ordered)}
        strategy['priority_basis'] = dict(method='ordinal_geometric', decay=decay,
            order=[d['direction_id'] for d in ordered],
            interpretation='Configured preference per rank position, not a probability or predicted intervention gain. Inspect normalized gradient responses.')
    weights = strategy['priority_weights']
    ids = {d['direction_id'] for d in ordered}
    if (not isinstance(weights, dict) or set(weights) != ids or
            any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights.values())):
        raise ValueError('Priority weights must name every executable optimization direction with a finite positive value')
    return result


def scaled_objectives(values, strategy):
    weights = strategy.get('priority_weights', {})
    return {key: value*weights.get(key, 1.) for key, value in values.items()}
