"""Evidence contracts for an optional agent-operated Researcher inside MolThinker.

Web retrieval belongs to the agent/tool layer. This module never fabricates a
search or converts a literature association into an experimentally known gain.
"""
from copy import deepcopy
from molsteer.common import digest


def research_queries(target,scaffold,fragment,endpoint='binding affinity'):
    if not all(isinstance(s,str) and s.strip() for s in (target,scaffold,fragment,endpoint)):
        raise ValueError('Verified target, scaffold, fragment and endpoint are required')
    return [f'{target} {scaffold} structure activity relationship {endpoint}',
        f'{target} {fragment} co crystal binding mode matched molecular pair',
        f'{scaffold} {fragment} permeability solubility metabolism matched pairs',
        f'{target} {scaffold} inactive analogues selectivity counterexamples']


def assess_hypothesis(card,capabilities):
    """Gate source-bound hypotheses before any reward is activated."""
    required=('hypothesis','target','baseline','fragment_mapping','sources','claimed_endpoint',
        'transfer_limits','counterevidence','expected_tradeoffs','proposed_control')
    missing=[k for k in required if k not in card]
    if missing:raise ValueError('Missing research fields: '+', '.join(missing))
    if not card['sources'] or any(not all(s.get(k) for k in ('url','title','evidence_type','support')) for s in card['sources']):
        raise ValueError('Each claim needs a cited evidence source and support statement')
    if card['claimed_endpoint'] in ('affinity','potency') and not any(s.get('target_match') and s.get('transformation_match') for s in card['sources']):
        status='exploratory_transfer_only'
    else:status='candidate_for_validation'
    reasons=[]
    if card['proposed_control'].get('atom_count_change') and not capabilities.get('variable_atom_count'):
        reasons.append('Required atom-count change unsupported by this adapter')
    if not card['fragment_mapping'].get('attachment_validated'):
        reasons.append('Attachment chemistry and atom mapping require validation')
    if not card.get('differentiable_surrogate'):
        reasons.append('No declared soft-category or coordinate objective; use as research/ranking evidence')
    variables=card['proposed_control'].get('variables',[])
    categorical=any(k in variables for k in ('atomics','bonds','charges'))
    if card.get('differentiable_surrogate') and categorical and not capabilities.get('categorical_probability_guidance'):
        reasons.append('Categorical probability guidance capability not established')
    if card.get('differentiable_surrogate') and 'coords' in variables and not capabilities.get('coordinate_guidance'):
        reasons.append('Coordinate guidance capability not established')
    if card.get('differentiable_surrogate') and not variables:
        reasons.append('Controlled variables are unspecified')
    result=deepcopy(card);result.update(status=status if not reasons else 'deferred',defer_reasons=reasons,
        activation='Explicit experimental selection plus derivative, native-kernel and endpoint validation required',
        automatic_reward_activation=False)
    result['hypothesis_id']='rh_'+digest(result)[:24]
    return result
