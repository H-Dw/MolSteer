"""Public scientific decision records, separate from private model reasoning."""
from typing import Literal, Any
from pydantic import Field
from .config import StrictModel

FACTORS = ('bond_geometry', 'angle_torsion_stereochemistry', 'intramolecular_stability',
           'steric_feasibility', 'target_contacts', 'target_surface_burial',
           'chemical_identity_functional_groups', 'terminal_task_utility')


class RepairClause(StrictModel):
    clause_id: str = Field(min_length=1, max_length=100)
    observable_kind: Literal['distance', 'receptor_distance', 'angle', 'dihedral',
        'signed_volume', 'mmff_strain', 'direction_alignment', 'anchor_offset',
        'whole_clash_screen', 'contact_count', 'contact_fraction', 'burial_sasa',
        'categorical_identity', 'terminal_oracle']
    evidence_ids: list[str] = Field(min_length=1)
    predicate: str = Field(min_length=12, description='One independently checkable clause of the repair predicate, with scope/target. List every conjunct, including preservation clauses. Unsupported whole-clash/count/area/category/oracle clauses need biology revision before execution.')


class PredicateCoverage(StrictModel):
    clause_id: str
    observable_ids: list[str] = Field(min_length=1, description='Actual observables implementing the entire declared clause in this direction expression. There are no automatic host contact-count, contact-fraction or burial/SASA gates.')
    implementation: str = Field(min_length=12)


class DraftReplacement(StrictModel):
    path: list[str | int] = Field(min_length=1,max_length=24,
        description='Existing JSON field/index path in the last proposed MathematicalDesign, e.g. [directions,0,expression]. Replacement only, no code or filesystem paths.')
    value: Any


class FactorAssessment(StrictModel):
    factor: Literal['bond_geometry', 'angle_torsion_stereochemistry', 'intramolecular_stability',
                    'steric_feasibility', 'target_contacts', 'target_surface_burial',
                    'chemical_identity_functional_groups', 'terminal_task_utility']
    disposition: Literal['optimize', 'constraint', 'monitor', 'deferred']
    direction_ids: list[str]
    evidence_ids: list[str]
    assessment: str = Field(min_length=12)
    missing_requirements: list[str] = Field(description='Blocking prerequisites in the declared scope. Must be [] for optimize/constraint; use monitor/deferred for unresolved prerequisites. Future live certification belongs in uncertainty/evaluation.')
    shortcut_test: str = Field(min_length=12)


class ReferenceParameter(StrictModel):
    origin: str = Field(min_length=5, description='Exact origin text of its expression constant')
    value: float
    unit: str
    role: Literal['reference', 'tolerance', 'normalization', 'physical_coefficient', 'operator_parameter']
    provenance: Literal['observed_reference', 'source_formula', 'calibration',
                        'declared_assumption', 'diagnostic_screening']
    evidence_ids: list[str]
    source_locators: list[str] = Field(description='Actual inspected chunk_id or fetched observation_id, never invented packet/biology path strings. Use evidence_ids for measured values; source_locators=[] is valid for an observed reference or declared assumption.')
    derivation: str = Field(min_length=12)


class FunctionCandidate(StrictModel):
    locator: str
    role: Literal['direct_repair', 'complementary_physics', 'preservation', 'terminal_utility', 'discrete_operator']
    decision: Literal['selected', 'rejected', 'deferred']
    reason: str = Field(min_length=12)
    missing_requirements: list[str]


class ShapeProbe(StrictModel):
    observable_values: dict[str, float]
    expected_zero: bool
    derivative_signs: dict[str, Literal['negative', 'zero', 'positive']]
    reason: str = Field(min_length=12)


class ArchitectureCandidate(StrictModel):
    name: str = Field(min_length=3)
    mathematical_form: str = Field(min_length=5)
    decision: Literal['selected', 'rejected', 'deferred']
    reason: str = Field(min_length=12)


class ArchitectureAudit(StrictModel):
    execution_scope: Literal['calibrated_repair', 'bounded_hypothesis_pilot']
    graph_policy: Literal['suspend_on_graph_change', 'graph_independent']
    architectures: list[ArchitectureCandidate] = Field(min_length=1, max_length=8)
    selection_reason: str = Field(min_length=12)
    unsupported_claims: list[str]
