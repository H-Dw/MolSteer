# MolThinker — Reward derivation

5i0b_A__5vef_M77 / ligand_002 / t_0.50

StatePacket: `sp_7c2bbc227fdd4e662ab161c2` · RewardSpec: `rw_4098beec3d554ffe2a657933`

Retrieved all 21 knowledge functions. Local coordinate rewards are conditional on the current chemical hypothesis. Live generator execution remains blocked; this trial checks numerical behavior on a coordinate copy.

## Derived reward terms

R = −Σ Eᵢ; Eᵢ = ½ wᵢ {max[(ℓᵢ−vᵢ)/sᵢ, 0]² + max[(vᵢ−uᵢ)/sᵢ, 0]²}.

The upper term is omitted for minimum-distance penalties. Views are evaluated separately. Weights are 1; distance scale is 1 Å and angle scale 10°. These are uncalibrated demonstration settings.

| View | Atoms | Observable | Bounds | Scale | Knowledge | Evidence |
|---|---|---|---|---|---|---|
| prediction | [1, 10] | 1-3 endpoint distance; coupled bond-length/angle proxy | 1.786254–3.077091 angstrom | 1.0 | G01, line 24 | ev_b426fc1a42649d46af8f |
| prediction | [1, 14] | MMFF reference bond window | 1.305900–1.596100 angstrom | 1.0 | G01, line 24 | ev_7f9b0b23003da23a9390 |
| state | [3] | ligand-protein minimum distance | ≥2.475000 angstrom | 1.0 | G02, line 25 | ev_dcbe941e2040b9b8d244 |
| state | [1] | ligand-protein minimum distance | ≥2.475000 angstrom | 1.0 | G02, line 25 | ev_66a39522a0aba6109d18 |

## Chemical hypotheses and evidence limits

- [1, 10]: order 0 p=0.439962; order 1 p=0.408524; order 2 p=0.139508.
- [1, 14]: order 1 p=0.341107; order 0 p=0.292530; order 2 p=0.212557.
- [10, 14]: order 1 p=0.590613; order 0 p=0.285790; order 2 p=0.073467.

The 1–3 distance couples bond lengths and angle. Its MMFF angle flag is supporting evidence, not an additional penalty. MMFF bond windows depend on current elements, formal charges, implicit hydrogens and bond orders; category changes require rederivation. Global strain is not local energy attribution.

## Retrieval and applicability decisions

| Function | Role | Decision | Reason |
|---|---|---|---|
| G01 · Flat-bottom interval penalty | coordinate_energy | conditional_offline | Localized measurements and bounds available; frozen hypothesis and explicit demo mobility required |
| P02 · Interfacial van der Waals potential | coordinate_energy | deferred | Protein coordinates exist; receptor force-field types do not |
| P01 · MMFF94 intramolecular energy | coordinate_energy | deferred | Global strain is evidence, but stable graph/protonation and parameter applicability are unverified |
| G02 · Pairwise minimum-distance penalty | coordinate_energy | conditional_offline | Localized measurements and bounds available; frozen hypothesis and explicit demo mobility required |
| G03 · Anchor quadratic potential | coordinate_energy | deferred | No anchor coordinates or mobility declaration |
| G04 · Directional agreement penalty | coordinate_energy | deferred | No target directions or prepared directional features |
| G05 · Stereochemical and planar geometry potential | coordinate_energy | deferred | No indicated stereo/plane defect with validated reference |
| G06 · Shape/volume occupancy potential | coordinate_energy | deferred | No selected reference shape or alignment |
| G07 · Local environment similarity kernel | coordinate_energy | deferred | No differentiable environment descriptor or reference library |
| P03 · Interfacial electrostatic potential | coordinate_energy | deferred | Formal charges cannot replace atomistic partial charges or a dielectric model |
| P04 · xTB force-norm loss | coordinate_energy | deferred | No xTB oracle, verified chemistry or oracle budget |
| P05 · Vina / torchvina total score | coordinate_energy_or_population_score | deferred | No prepared Vina inputs/backend or live suffix derivative |
| P06 · Differentiable SASA target potential | coordinate_energy | deferred | No atom-group SASA target or differentiable backend |
| P07 · ESP surface similarity potential | coordinate_energy | deferred | No aligned ESP reference or partial charges |
| S01 · Target-value Gaussian reward | population_weight | deferred | No target value/tolerance or live population |
| S02 · Softmax / tilted importance weights | population_weight | deferred | Affinity numbers alone do not define objectives, temperature or live particle population |
| S03 · On-target/off-target selectivity reward | population_score | deferred | No same-candidate matched off-target scores |
| S04 · SPSA gradient estimation | gradient_estimator | deferred | Estimator, not reward; missing reevaluable oracle, perturbable runtime state and budget |
| S05 · Pareto ranking and structural diversity | selection_strategy | deferred | No declared multiobjective population or diversity metric |
| S06 · Depth-calibrated batch integrity score | hyperparameter_strategy | deferred | Saved individual snapshots do not provide calibrated editing-depth batches |
| S07 · Three-population validity penalties | population_strategy | deferred | No target fragment and no three-population search adapter |

Unhandled diagnostic risks (such as valence contradictions, PAD tokens or disconnection) remain evidence. This implementation does not invent differentiable categorical repair rewards. SPSA is an estimator and cannot be enabled without a reevaluable oracle.

## Execution boundary

Only coordinate-copy derivatives are available. Saved tensors do not provide the generator Jacobian; active-slot masks are not editable masks. Three saved molecules do not establish a live particle population. Generator integration requires explicit editable degrees of freedom, budget and endpoint derivative mapping.

## Offline numerical validation

- Penalty: 0.03282699 → 5.6170305e-13; reward: -0.03282699 → -5.6170305e-13.
- Finite differences: True; maximum error 4.458e-10.
- Demo movable atoms: [1, 10, 14]; maximum displacement 0.124730 Å.
- Fixed atoms unchanged: True; original snapshot unchanged: True.
- New bond-window violations: 0; new protein clashes: 0.

| Atoms | Before | After | Unit |
|---|---|---|---|
| [1, 10] | 1.537599 | 1.786253 | angstrom |
| [1, 14] | 1.244057 | 1.335957 | angstrom |

This validates the formula and local energy descent only. Full generation was not rerun; improved final quality, binding or chemical stability is not established. Original generation files are unchanged.
