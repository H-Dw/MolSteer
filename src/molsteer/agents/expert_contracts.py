"""Public, versioned expert handoffs. Mathematical summaries are not private reasoning."""
from typing import Literal
from pydantic import Field, field_validator
from .config import StrictModel
from molsteer.common import digest
from molsteer.molthinker.expressions import validate_expression, validate_observables


class BiologyDirection(StrictModel):
    direction_id: str = Field(min_length=1, max_length=100)
    rank: int = Field(ge=1)
    finding_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    mechanism: str = Field(min_length=12)
    evidence_class: Literal['observation', 'proxy', 'mechanistic_hypothesis', 'intervention_supported']
    optimization_direction: str = Field(min_length=8)
    repair_predicate: str = Field(min_length=12)
    preservation_conditions: list[str] = Field(description='IDs of explicit constraint directions; do not put unenforced prose here')
    chemical_state: str = Field(min_length=8)
    falsifier: str = Field(min_length=12)
    uncertainty: list[str]
    required: bool
    disposition: Literal['optimize', 'constraint', 'monitor', 'deferred']
    priority_reason: str = Field(min_length=12)


class BiologyPlan(StrictModel):
    kind: Literal['BiologyPlan'] = 'BiologyPlan'
    schema_version: Literal['1.0'] = '1.0'
    outcome: str = Field(min_length=12)
    independent_measurement: str = Field(min_length=12)
    directions: list[BiologyDirection] = Field(min_length=1, max_length=32)
    summary: str = Field(min_length=12)


class ModelDynamicsContext(StrictModel):
    kind: Literal['ModelDynamicsContext'] = 'ModelDynamicsContext'
    model_type: str = 'unknown'
    time_convention: str = 'unknown'
    prediction_parameterization: str = 'unknown'
    coordinate_mapping: str = 'unknown'
    editable_atom_ids: list[int] | None = None
    live_derivative: Literal['available', 'unavailable', 'not_run'] = 'unavailable'
    injection_convention: str = 'unknown'
    source: str = 'No live adapter supplied; coordinate-copy validation only'

    @field_validator('editable_atom_ids')
    @classmethod
    def unique_nonnegative_atoms(cls,value):
        if value is not None and (not value or len(value)!=len(set(value)) or any(i<0 for i in value)):
            raise ValueError('Editable atom IDs must be nonempty, distinct and nonnegative')
        return value


class MathematicalDirection(StrictModel):
    direction_id: str
    status: Literal['executable', 'design_only']
    retrieval_ids: list[str] = Field(min_length=1)
    source_ids: list[str]
    function_lineage: list[dict] = Field(description='Each entry: source_id, locator (chunk_id or observation_id), original_formula, adaptation. Copy the actual source formula; distinguish your derivation.')
    formula: str = Field(min_length=5)
    derivation_summary: str = Field(min_length=12)
    acceptable_set: str = Field(min_length=12)
    normalization: str = Field(min_length=12)
    assumptions: list[str]
    alternatives: list[str] = Field(min_length=1)
    gradient_path: str = Field(min_length=12)
    marginal_sensitivity: str = Field(min_length=12)
    failure_mode: str = Field(min_length=12)
    missing_requirements: list[str]
    observables: list[dict]
    expression: dict | None


class MathematicalDesign(StrictModel):
    kind: Literal['MathematicalDesign'] = 'MathematicalDesign'
    schema_version: Literal['2.0'] = '2.0'
    directions: list[MathematicalDirection] = Field(min_length=1, max_length=32)
    strategy: dict
    conflict_assessment: str = Field(min_length=12)
    independent_evaluation: str = Field(min_length=12)


def validate_biology(plan, report):
    value = BiologyPlan.model_validate(plan).model_dump()
    findings = {f['finding_id']: f for f in report['findings']+report.get('raw_state_findings', [])}
    directions = value['directions']
    ids = [d['direction_id'] for d in directions]
    if len(ids) != len(set(ids)) or sorted(d['rank'] for d in directions) != list(range(1, len(ids)+1)):
        raise ValueError('Direction identities and ranks must be unique and contiguous')
    covered = set()
    constraints = {d['direction_id'] for d in directions if d['disposition'] == 'constraint'}
    for direction in directions:
        fs, evidence = set(direction['finding_ids']), set(direction['evidence_ids'])
        if not fs <= set(findings) or not evidence <= set(report['evidence_index']):
            raise ValueError('Biological direction cites unknown diagnostic evidence')
        related = {e for f in fs for e in findings[f]['evidence_ids']}
        if not evidence <= related:
            raise ValueError('Evidence does not belong to the direction findings')
        if not set(direction['preservation_conditions']) <= constraints:
            raise ValueError('Preservation conditions must reference explicit constraint directions')
        if direction['required'] and direction['disposition']=='monitor':
            raise ValueError('A required repair cannot have a monitoring-only disposition')
        covered |= fs
    if covered != set(findings):
        raise ValueError('Every diagnostic finding requires an explicit disposition')
    return value


def validate_math(design, biology, packet, retrievals, sources):
    value = MathematicalDesign.model_validate(design).model_dump()
    targets = {d['direction_id']: d for d in biology['directions'] if d['disposition'] in ('optimize', 'constraint')}
    ids = [d['direction_id'] for d in value['directions']]
    if len(ids) != len(set(ids)) or set(ids) != set(targets):
        raise ValueError('Math must account for each selected direction without changing biological priorities')
    retrieved = {r['retrieval_id']: r for r in retrievals}
    for direction in value['directions']:
        referenced = []
        passages = {}
        for rid in direction['retrieval_ids']:
            if rid not in retrieved or retrieved[rid]['direction_id'] != direction['direction_id']:
                raise ValueError('Each direction requires its own actual local retrieval')
            referenced.extend(x['source_id'] for x in retrieved[rid]['records'])
            passages.update({x['chunk_id']:x for x in retrieved[rid]['records']})
        passages.update({x['observation_id']:x for x in sources.values() if 'observation_id' in x})
        if not set(direction['source_ids']) <= set(referenced) | {x['source_id'] for x in sources.values()}:
            raise ValueError('Formula cites an unretrieved source')
        if direction['status'] == 'executable':
            if not direction['source_ids'] or direction['missing_requirements'] or direction['expression'] is None:
                raise ValueError('Executable formulas need inspected sources and all prerequisites')
            if not direction['function_lineage']:
                raise ValueError('Executable formulas require function-level lineage')
            lineage_sources=set()
            for lineage in direction['function_lineage']:
                if set(lineage)!={'source_id','locator','original_formula','adaptation'} or lineage['locator'] not in passages:
                    raise ValueError('Function lineage must identify an inspected source location')
                source=passages[lineage['locator']]
                formula=lineage['original_formula']
                if (lineage['source_id']!=source['source_id'] or not isinstance(formula,str) or not formula.strip()
                        or formula not in (source.get('formula') or source.get('excerpt',''))
                        or not isinstance(lineage['adaptation'],str) or len(lineage['adaptation'].strip())<12):
                    raise ValueError('Lineage formula/source must match the inspected passage and explain specialization')
                lineage_sources.add(lineage['source_id'])
            if not lineage_sources <= set(direction['source_ids']):
                raise ValueError('Lineage sources must be included in direction citations')
            units = validate_observables(direction['observables'], packet, set(targets[direction['direction_id']]['evidence_ids']))
            validate_expression(direction['expression'], units)
        elif not direction['missing_requirements']:
            raise ValueError('Deferred designs must explain missing requirements')
    active = [d for d in value['directions'] if targets[d['direction_id']]['disposition'] == 'optimize' and d['status'] == 'executable']
    strategy = value['strategy']
    if set(strategy) != {'mode', 'aggregation', 'justification'} or len(str(strategy['justification'])) < 12:
        raise ValueError('Strategy needs mode, aggregation and scientific justification')
    if strategy['mode'] == 'common_descent':
        if strategy['aggregation'] is not None:
            raise ValueError('Common descent has no scalarization weights or aggregation')
    elif strategy['mode'] == 'scalar_potential':
        agg = strategy['aggregation']
        if not isinstance(agg, dict) or agg.get('op') not in ('single', 'maximum', 'lp_norm'):
            raise ValueError('Flat weighted sums are not an expert control strategy')
        if agg['op'] == 'lp_norm':
            if set(agg) != {'op', 'p'} or type(agg['p']) not in (int, float) or not 1 < agg['p'] <= 8:
                raise ValueError('lp_norm requires 1 < p <= 8')
        elif set(agg) != {'op'}:
            raise ValueError('Unexpected aggregation parameters')
        if agg['op'] == 'single' and len(active) > 1:
            raise ValueError('Single objective strategy cannot omit selected directions')
    else:
        raise ValueError('Unsupported controller; retain as design_only')
    return value


def compile_expert_spec(packet, report, biology, design, retrievals, dynamics, sources):
    dynamics=ModelDynamicsContext.model_validate(dynamics).model_dump()
    if dynamics['editable_atom_ids'] is not None and not set(dynamics['editable_atom_ids']) <= set(packet['representations']['prediction']['original_atom_ids']):
        raise ValueError('Declared editable atoms must belong to the bound molecule')
    biology = validate_biology(biology, report)
    design = validate_math(design, biology, packet, retrievals, sources)
    by_id = {d['direction_id']: d for d in design['directions']}
    blocked = [d['direction_id'] for d in biology['directions']
               if (d['required'] or d['disposition'] == 'constraint') and
               (d['disposition'] == 'deferred' or
                d['disposition'] in ('optimize', 'constraint') and by_id[d['direction_id']]['status'] != 'executable')]
    active = [d['direction_id'] for d in biology['directions'] if d['disposition'] == 'optimize'
              and by_id[d['direction_id']]['status'] == 'executable']
    if blocked or not active:
        return None, dict(status='design_only', blocked_directions=blocked,
                          reason='Required directions are unresolved or no executable optimization remains')
    spec = dict(kind='RewardSpec', schema_version='2.0.0', packet_id=packet['packet_id'], identity=packet['identity'],
                graph_signatures=packet['steering']['graph_signatures'],
                coordinate_hashes={v:s['coordinate_hash'] for v,s in packet['steering']['coordinate_snapshots'].items()},
                biology_plan=biology, mathematical_design=design, retrieval=retrievals, research_sources=sources,
                model_dynamics=dynamics, active_direction_ids=active,
                runtime_execution={'status':'requires_live_validation', 'live_gradient':'not_run'},
                required_validation=['units_and_domains', 'finite_difference', 'fixed_variables',
                                     'live_pullback', 'post_injection_direction', 'proposal_constraints'])
    spec['reward_id'] = 'rw_'+digest(spec)[:24]
    return spec, None


def validate_expert_spec(spec, packet, report=None):
    if spec.get('reward_id') != 'rw_'+digest({k:v for k,v in spec.items() if k != 'reward_id'})[:24]:
        raise ValueError('Expert reward digest mismatch')
    if (spec.get('packet_id') != packet['packet_id'] or spec.get('identity') != packet['identity']
            or spec.get('graph_signatures') != packet['steering']['graph_signatures']
            or spec.get('coordinate_hashes') != {v:s['coordinate_hash'] for v,s in packet['steering']['coordinate_snapshots'].items()}):
        raise ValueError('Expert reward state binding changed')
    if report is None:
        from molreader.localized_report import make_localized_report
        report = make_localized_report(packet)
    rebuilt, deferred = compile_expert_spec(packet, report, spec['biology_plan'], spec['mathematical_design'],
                                          spec['retrieval'], spec['model_dynamics'], spec['research_sources'])
    if deferred or rebuilt['reward_id'] != spec['reward_id']:
        raise ValueError('Expert reward is not an executable validated handoff')
    ModelDynamicsContext.model_validate(spec['model_dynamics'])
    return True
