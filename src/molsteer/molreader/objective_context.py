"""Attach constraints and feedback only when explicitly supplied by their owner."""
from molsteer.common import fact


def compute(objectives=None):
    # Rich target/interval semantics require a validated future adapter. Reject rather than silently ignore.
    if objectives:
        raise ValueError('Custom objective adapter is not implemented; no objective will be inferred or silently ignored')
    missing = lambda reason: fact(reason=reason)
    return {
        'spec': missing('No property targets, directions or hard limits supplied'),
        'priors': missing('No anchor, shape, pharmacophore or ESP reference supplied'),
        'off_target_pockets': missing('No matched off-target evaluations'),
        'feedback': {k: missing('Requires controlled execution or population history') for k in
                     ['delta_E', 'delta_norm', 'rebound', 'pass_rate', 'validity_rate', 'atom_stability_rate', 'diversity', 'ESS']},
    }
