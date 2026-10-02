"""Public, versioned expert handoffs. Mathematical summaries are not private reasoning."""
from typing import Literal
import re
from pydantic import Field, field_validator
from .config import StrictModel
from .contract_errors import ContractValidationError
from .expression_contracts import Expression
from .decision_workspace import selected_directions
from .audit_contracts import (FactorAssessment, ReferenceParameter, FunctionCandidate,
                              ShapeProbe, ArchitectureAudit, RepairClause, PredicateCoverage)
from .design_audit import validate_factor_assessment, validate_design_audit, validate_predicate_claims
from molsteer.molreader.measurement_refs import measurement_references
from molsteer.common import digest
from molsteer.molthinker.expressions import validate_expression, validate_observables


class BiologyDirection(StrictModel):
    direction_id: str = Field(min_length=1, max_length=100)
    rank: int = Field(ge=1, description='Value priority: lower ranks deserve earlier/stronger attention after raw trajectory, controllability, dependencies and tradeoffs are considered')
    value_assessment: dict = Field(default_factory=dict,
        description='Public rationale: raw trajectory references, persistence/retyping/late repair, plausible benefit, native-evolution risk, controllability, dependency and uncertainty; no invented intervention gains')
    finding_ids: list[str] = Field(description='Diagnostic finding IDs, or [] for a separate measured preservation/task direction')
    evidence_ids: list[str] = Field(min_length=1)
    mechanism: str = Field(min_length=12)
    evidence_class: Literal['observation', 'proxy', 'mechanistic_hypothesis', 'intervention_supported']
    optimization_direction: str = Field(min_length=8)
    repair_predicate: str = Field(min_length=12)
    preservation_conditions: list[str] = Field(description='IDs of explicit constraint directions; do not put unenforced prose here')
    chemical_state: str = Field(min_length=8)
    falsifier: str = Field(min_length=12)
    uncertainty: list[str]
    required: bool = Field(description='Scientifically necessary repair/preservation. If execution inputs are missing, required=true with deferred disposition retains it for mathematics.')
    disposition: Literal['optimize', 'constraint', 'monitor', 'deferred']
    priority_reason: str = Field(min_length=12)
    repair_clauses: list[RepairClause] = Field(default_factory=list, max_length=32,
        description='When audit_required, enumerate EVERY clause of each optimize/constraint repair predicate. Do not hide unsupported whole-clash/count/area guarantees in prose. Other dispositions may use [].')


class BiologyPlan(StrictModel):
    kind: Literal['BiologyPlan'] = 'BiologyPlan'
    schema_version: Literal['1.0'] = '1.0'
    outcome: str = Field(min_length=12)
    independent_measurement: str = Field(min_length=12)
    directions: list[BiologyDirection] = Field(min_length=1, max_length=32)
    summary: str = Field(min_length=12)
    factor_assessment: list[FactorAssessment] = Field(default_factory=list, max_length=8,
        description='When audit_required, assess every one of the eight factors exactly once; active factors require matching direction dispositions and no blocking prerequisites.')


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
    execution_scope: Literal['complete_goal_set', 'bounded_coordinate_pilot'] = 'complete_goal_set'

    @field_validator('editable_atom_ids')
    @classmethod
    def unique_nonnegative_atoms(cls,value):
        if value is not None and (not value or len(value)!=len(set(value)) or any(i<0 for i in value)):
            raise ValueError('Editable atom IDs must be nonempty, distinct and nonnegative')
        return value


class MathematicalDirection(StrictModel):
    direction_id: str
    status: Literal['executable', 'design_only'] = Field(description='Executable means the declared expression is runnable on bound coordinate copies with present prerequisites. It does not certify live guidance. Missing physical/oracle inputs require design_only or biology revision.')
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
    missing_requirements: list[str] = Field(description='Blocking requirements for current expression evaluation. MUST be [] for executable. Put future live Jacobian/continuation certification in gradient_path, independent_evaluation and unsupported_claims; do not erase missing physical inputs.')
    observables: list[dict]
    expression: Expression | None
    source_transform: Literal['retained', 'specialized', 'new_surrogate'] | None = None
    function_basis: list[FunctionCandidate] = Field(default_factory=list, max_length=12)
    reference_parameters: list[ReferenceParameter] = Field(default_factory=list, max_length=64)
    shape_probes: list[ShapeProbe] = Field(default_factory=list, max_length=12)
    predicate_coverage: list[PredicateCoverage] = Field(default_factory=list, max_length=32,
        description='When audit_required, map every biological repair_clauses ID to actual expression observables. Missing/unsupported clauses require design_only or request_biology_revision. No unstated host gates exist.')
    constraint_mode: Literal['absolute', 'native_nonincrease'] = 'absolute'


class MathematicalDesign(StrictModel):
    kind: Literal['MathematicalDesign'] = 'MathematicalDesign'
    schema_version: Literal['2.0'] = '2.0'
    directions: list[MathematicalDirection] = Field(min_length=1, max_length=32)
    strategy: dict
    conflict_assessment: str = Field(min_length=12)
    independent_evaluation: str = Field(min_length=12)
    design_audit: ArchitectureAudit | None = None


def validate_biology(plan, report, packet=None, *, require_audit=False):
    value = BiologyPlan.model_validate(plan).model_dump()
    findings = {f['finding_id']: f for f in report['findings']+report.get('raw_state_findings', [])}
    measured_evidence = {e['evidence_id'] for m in (packet or {}).get('observations', [])
                         for e in m.get('evidence', [])}
    measurement_ids = set(measurement_references(packet)) if packet else set()
    measured_evidence.update(measurement_ids)
    directions = value['directions']
    ids = [d['direction_id'] for d in directions]
    if len(ids) != len(set(ids)) or sorted(d['rank'] for d in directions) != list(range(1, len(ids)+1)):
        raise ValueError('Direction identities and ranks must be unique and contiguous')
    covered = set()
    constraints = {d['direction_id'] for d in directions if d['disposition'] == 'constraint'}
    for direction_index, direction in enumerate(directions):
        fs, evidence = set(direction['finding_ids']), set(direction['evidence_ids'])
        for field,allowed in [('finding_ids',set(findings)),('evidence_ids',set(report['evidence_index']) | measured_evidence)]:
            for index,ident in enumerate(direction[field]):
                if ident not in allowed:
                    raise ContractValidationError('Biological direction cites unknown diagnostic evidence',
                        ['directions',direction_index,field,index])
        related = ({e for f in fs for e in findings[f]['evidence_ids']} | measurement_ids) if fs else measured_evidence
        if not evidence <= related:
            raise ContractValidationError('Evidence does not belong to the direction findings',
                                          ['directions', direction_index, 'evidence_ids'])
        if not set(direction['preservation_conditions']) <= constraints:
            raise ValueError('Preservation conditions must reference explicit constraint directions')
        if direction['required'] and direction['disposition']=='monitor':
            raise ValueError('A required repair cannot have a monitoring-only disposition')
        clauses=direction['repair_clauses']
        if direction['disposition'] in ('optimize','constraint') and (clauses or require_audit):
            if not clauses or len({c['clause_id'] for c in clauses})!=len(clauses):
                raise ContractValidationError('Enumerate every independent repair clause with unique IDs',
                    ['directions',direction_index,'repair_clauses'])
            for clause_index,clause in enumerate(clauses):
                for index,ident in enumerate(clause['evidence_ids']):
                    if ident not in evidence:
                        raise ContractValidationError('Repair clauses must cite their direction bound evidence',
                            ['directions',direction_index,'repair_clauses',clause_index,'evidence_ids',index])
            validate_predicate_claims(direction,['directions',direction_index,'repair_clauses'])
        if not clauses:
            direction.pop('repair_clauses')
        if not direction['value_assessment']:
            direction.pop('value_assessment')
        covered |= fs
    if covered != set(findings):
        raise ValueError('Every diagnostic finding requires an explicit disposition')
    if value['factor_assessment'] or require_audit:
        if packet is None:
            raise ValueError('Factor assessment requires the bound StatePacket')
        validate_factor_assessment(value, packet, report, required=require_audit)
    else:
        # Do not alter content-addressed legacy handoffs when an audit is absent.
        value.pop('factor_assessment')
    return value


def _formula_layout(text):
    """Ignore LaTeX math whitespace only; preserve all tokens and outside prose."""
    return re.sub(r'\$[^$]*\$', lambda match: re.sub(r'\s+', '', match.group()), text)


def validate_math(design, biology, packet, retrievals, sources, *, report=None, require_audit=False, rank_decay=0.5):
    value = MathematicalDesign.model_validate(design).model_dump()
    from .priority import allocate_priorities
    value = allocate_priorities(value, biology, rank_decay)
    targets = {d['direction_id']: d for d in selected_directions(biology)}
    ids = [d['direction_id'] for d in value['directions']]
    if len(ids) != len(set(ids)) or set(ids) != set(targets):
        raise ValueError('Math must account for each selected direction without changing biological priorities')
    retrieved = {r['retrieval_id']: r for r in retrievals}
    all_passages = {}
    for direction_index, direction in enumerate(value['directions']):
        referenced = []
        passages = {}
        for rid in direction['retrieval_ids']:
            if rid not in retrieved or retrieved[rid]['direction_id'] != direction['direction_id']:
                raise ContractValidationError('Each direction requires its own actual local retrieval',
                    ['directions',direction_index,'retrieval_ids'])
            referenced.extend(x['source_id'] for x in retrieved[rid]['records'])
            passages.update({x['chunk_id']:x for x in retrieved[rid]['records']})
        passages.update({x['observation_id']:x for x in sources.values() if 'observation_id' in x})
        all_passages.update(passages)
        if not set(direction['source_ids']) <= set(referenced) | {x['source_id'] for x in sources.values()}:
            raise ValueError('Formula cites an unretrieved source')
        if direction['status'] == 'executable':
            if not direction['source_ids'] or direction['missing_requirements'] or direction['expression'] is None:
                field='source_ids' if not direction['source_ids'] else 'missing_requirements' if direction['missing_requirements'] else 'expression'
                raise ContractValidationError('Executable formulas need inspected sources and all prerequisites',
                    ['directions',direction_index,field])
            if not direction['function_lineage']:
                raise ValueError('Executable formulas require function-level lineage')
            lineage_sources=set()
            for lineage_index, lineage in enumerate(direction['function_lineage']):
                if set(lineage)!={'source_id','locator','original_formula','adaptation'} or lineage['locator'] not in passages:
                    raise ContractValidationError('Function lineage must identify an inspected source location',
                        ['directions',direction_index,'function_lineage',lineage_index])
                source=passages[lineage['locator']]
                formula=lineage['original_formula']
                field = ('source_id' if lineage['source_id']!=source['source_id'] else
                         'original_formula' if not isinstance(formula,str) or not formula.strip()
                         or _formula_layout(formula) not in _formula_layout(source.get('formula') or source.get('excerpt','')) else
                         'adaptation' if not isinstance(lineage['adaptation'],str) or len(lineage['adaptation'].strip())<12 else None)
                if field:
                    raise ContractValidationError('Lineage formula/source must match the inspected passage and explain specialization',
                        ['directions', direction_index, 'function_lineage', lineage_index, field])
                lineage_sources.add(lineage['source_id'])
            if not lineage_sources <= set(direction['source_ids']):
                raise ValueError('Lineage sources must be included in direction citations')
            try:
                units = validate_observables(direction['observables'], packet, set(targets[direction['direction_id']]['evidence_ids']))
            except ValueError as exc:
                raise ContractValidationError(str(exc),['directions',direction_index,'observables']) from None
            try:
                validate_expression(direction['expression'], units)
            except ValueError as exc:
                raise ContractValidationError(str(exc),['directions',direction_index,'expression']) from None
        elif not direction['missing_requirements']:
            raise ValueError('Deferred designs must explain missing requirements')
    active = [d for d in value['directions'] if targets[d['direction_id']]['disposition'] == 'optimize' and d['status'] == 'executable']
    strategy = value['strategy']
    if set(strategy) - {'mode', 'aggregation', 'justification', 'priority_weights', 'priority_basis'} or not {'mode','aggregation','justification'} <= set(strategy) or len(str(strategy['justification'])) < 12:
        raise ValueError('Strategy needs mode, aggregation and scientific justification')
    if strategy['mode'] == 'design_only':
        if strategy['aggregation'] is not None:
            raise ValueError('Unsupported controller; retain as design_only')
    elif strategy['mode'] == 'common_descent':
        if strategy['aggregation'] is not None:
            raise ValueError('Common descent has no scalarization weights or aggregation')
    elif strategy['mode'] == 'scalar_potential':
        agg = strategy['aggregation']
        if not isinstance(agg, dict) or agg.get('op') not in ('single', 'maximum', 'lp_norm', 'weighted_sum'):
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
    validate_design_audit(value, biology, packet, report or {'evidence_index': {}}, all_passages,
                          required=require_audit)
    if value['design_audit'] is None:
        value.pop('design_audit')
    for direction in value['directions']:
        for field in ('source_transform', 'function_basis', 'reference_parameters', 'shape_probes', 'predicate_coverage'):
            if not direction[field]:
                direction.pop(field)
        if direction['constraint_mode'] == 'absolute':
            direction.pop('constraint_mode')
    return value


def compile_expert_spec(packet, report, biology, design, retrievals, dynamics, sources):
    dynamics=ModelDynamicsContext.model_validate(dynamics).model_dump()
    if dynamics['editable_atom_ids'] is not None and not set(dynamics['editable_atom_ids']) <= set(packet['representations']['prediction']['original_atom_ids']):
        raise ValueError('Declared editable atoms must belong to the bound molecule')
    biology = validate_biology(biology, report, packet)
    design = validate_math(design, biology, packet, retrievals, sources, report=report)
    by_id = {d['direction_id']: d for d in design['directions']}
    blocked = [d['direction_id'] for d in biology['directions']
               if (d['required'] or d['disposition'] == 'constraint') and
               (d['disposition'] == 'deferred' or
                d['disposition'] in ('optimize', 'constraint') and by_id[d['direction_id']]['status'] != 'executable')]
    active = [d['direction_id'] for d in biology['directions'] if d['disposition'] == 'optimize'
              and by_id[d['direction_id']]['status'] == 'executable']
    dependencies={ident for direction in biology['directions']
                  if direction['disposition'] in ('optimize','constraint')
                  for ident in direction['preservation_conditions']}
    blocked_constraints=[d['direction_id'] for d in biology['directions']
                         if d['disposition']=='constraint' and by_id[d['direction_id']]['status']!='executable']
    partial=(dynamics['execution_scope']=='bounded_coordinate_pilot' and blocked
             and not blocked_constraints and not set(blocked)&dependencies)
    if (blocked and not partial) or not active or design['strategy']['mode'] == 'design_only':
        return None, dict(status='design_only', blocked_directions=blocked,
                          reason='Required directions or controller are unresolved, or no executable optimization remains')
    spec = dict(kind='RewardSpec', schema_version='2.0.0', packet_id=packet['packet_id'], identity=packet['identity'],
                graph_signatures=packet['steering']['graph_signatures'],
                coordinate_hashes={v:s['coordinate_hash'] for v,s in packet['steering']['coordinate_snapshots'].items()},
                biology_plan=biology, mathematical_design=design, retrieval=retrievals, research_sources=sources,
                model_dynamics=dynamics, active_direction_ids=active,
                runtime_execution={'status':'requires_live_validation', 'live_gradient':'not_run'},
                required_validation=['units_and_domains', 'finite_difference', 'fixed_variables',
                                     'live_pullback', 'post_injection_direction', 'proposal_constraints'])
    if partial:
        spec['partial_execution']={'scope':'bounded_coordinate_pilot','complete_goal_set_resolved':False,
            'unresolved_required_direction_ids':blocked,
            'statement':'Only executable coordinate goals are tested. Required deferred goals remain unresolved; no full molecular-quality repair is claimed.'}
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
