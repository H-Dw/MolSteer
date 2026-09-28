"""Declared evidence policy; never interpret these scores as potency probabilities."""
import math


def evidence_weight(card):
    factors=card.get('evidence_factors',{})
    keys=('source_quality','target_relevance','transformation_support','context_match','counterevidence_penalty')
    if set(factors)!=set(keys) or any(not isinstance(factors[k],(float,int)) or not math.isfinite(factors[k]) or not 0<=factors[k]<=1 for k in keys):
        raise ValueError('Five finite, explicit evidence factors in [0,1] required')
    # No source-count multiplier: duplicate papers or large searches cannot raise weight.
    transfer=.25+.75*factors['transformation_support']
    cap=factors['source_quality']*factors['target_relevance']*transfer*factors['context_match']*(1-factors['counterevidence_penalty'])
    if card['status']=='deferred':cap=0.
    return dict(cap=cap,initial=.5*cap,factors=factors,
        formula='cap = quality * target * (0.25 + 0.75 * transformation) * context * (1 - counterevidence); initial = cap/2',
        interpretation='Explicit uncalibrated experimental influence budget. Missing exact SAR permits only bounded, explicitly selected exploration.')
