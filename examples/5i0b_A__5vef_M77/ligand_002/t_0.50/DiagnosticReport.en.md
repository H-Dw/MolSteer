# DiagnosticReport — 5i0b_A__5vef_M77 / ligand_002 / t_0.50

Assessment: **risks_observed**. 1 endpoint risk regions; 3 raw-state groups; 5 coverage root causes.

StatePacket: `sp_7c2bbc227fdd4e662ab161c2`. Atom IDs are original zero-based tensor slots.

## Predicted endpoint risks

### local_geometry · elevated

Location: [1, 10, 14]; scope: predicted_endpoint.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 1 | N | 0 | 1.0 |
| 10 | C | 0 | 2.0 |
| 14 | C | 0 | 2.0 |

- Atoms [1, 14], order 1: length **1.2441 Å**; MMFF reference 1.4510 Å, relative deviation -14.3%. Evidence: `ev_7f9b0b23003da23a9390`.
- Atoms [1, 14, 10], center 14: angle **67.669°**; MMFF reference 108.290°, deviation -40.621°. Evidence: `ev_bf05f4df733dbc0be334`.
- Atoms [1, 14, 10], center 14: angle **67.669°**; endpoint distance 1.5376 Å versus 1.7863–3.0771 Å. Evidence: `ev_b426fc1a42649d46af8f`.
- Support: Atoms [1, 10]: maximum category probability 0.440, top-two margin 0.031; not a defect probability. Evidence: `ev_fc58ca3ccc2aa0a99c8e`.
- Support: Atoms [1, 14]: maximum category probability 0.341, top-two margin 0.049; not a defect probability. Evidence: `ev_8a9968b3b1575efbacb1`.
- Support: Atoms [10, 14]: maximum category probability 0.591, top-two margin 0.305; not a defect probability. Evidence: `ev_e59b4afdaa1f48aee4f6`.
- Support: Molecule-level relaxation energy drop **111.704 kcal/mol**; screen threshold 20. This is not local energy attribution. Evidence: `ev_1a90667f9ee8002556c9`.
- Support: Molecule-level PoseBusters check `bond_angles` failed. Evidence: `ev_304545adc1275cd7579b`.

Local categorical alternatives (probabilities are not defect probabilities):

- Pair [1, 10]: declared order 0; 1.537599 Å. Alternatives: order 0: 0.440; order 1: 0.409; order 2: 0.140.
- Pair [1, 14]: declared order 1; 1.244057 Å. Alternatives: order 1: 0.341; order 0: 0.293; order 2: 0.213.
- Pair [10, 14]: declared order 1; 1.492470 Å. Alternatives: order 1: 0.591; order 0: 0.286; order 2: 0.073.

Relative to t=0.25: atom 1: O(0) → N(0); pair [1, 10]: order 1 → 0; pair [1, 14]: order 0 → 1. Chemical identity changed; this does not establish persistence of the same constraint.

Method limits: MMFF screening thresholds are >10% bond-length deviation and >30° angle deviation. These are uncalibrated screens; parameter applicability, especially for charged nitrogen environments, is not established.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: prediction + sdf; 13 source observations. Full card: `risk_880f5e87f7c6b287`.

## Current noisy state X_t

These observations describe X_t and do not establish final failure.

### raw_graph_validity · high

Location: [1, 2, 6, 9, 13, 14, 15, 19]; scope: observed_noisy_state.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 1 | N | 0 | 26.0 |
| 2 | O | 0 | 12.0 |
| 6 | C | 0 | 24.5 |
| 13 | N | 0 | 20.0 |
| 19 | None | 0 | 17.5 |

- Atoms [1], neutral N: declared bond-order sum **26**; conservative valence-demand lower bound 25, ordinary cap 3. Evidence: `ev_0bb3f60f9a8d9c0eb3ae`.
- Atoms [13], neutral N: declared bond-order sum **20**; conservative valence-demand lower bound 18, ordinary cap 3. Evidence: `ev_378fd072ce1ae279dea6`.
- Atoms [6], neutral C: declared bond-order sum **24.5**; conservative valence-demand lower bound 22, ordinary cap 4. Evidence: `ev_f8d8a3b2c196580a79c0`.
- Atoms [2], neutral O: declared bond-order sum **12**; conservative valence-demand lower bound 11, ordinary cap 2. Evidence: `ev_806e0a1a217fa3136b75`.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: state; 9 source observations. Full card: `risk_a1a7348bf672d3e6`.

### binding_interface_sterics · elevated

Location: [3]; scope: observed_noisy_state.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 3 | N | 1 | 14.0 |

- Atoms [3] to A:447:LEU:CD1: distance **1.9967 Å**, vdW ratio 0.6051; threshold 0.75. Evidence: `ev_dcbe941e2040b9b8d244`.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: state; 1 source observations. Full card: `risk_ec54c67fd53cfb36`.

### binding_interface_sterics · elevated

Location: [1]; scope: observed_noisy_state.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 1 | N | 0 | 26.0 |

- Atoms [1] to A:335:VAL:CG2: distance **2.4335 Å**, vdW ratio 0.7374; threshold 0.75. Evidence: `ev_66a39522a0aba6109d18`.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: state; 1 source observations. Full card: `risk_87e147bd63493bb5`.

## Derived-evidence appendix

All source observations and evidence IDs remain in DiagnosticReport.json and StatePacket.json.

- state.bond_lengths: 56 observations, not additional independent endpoint defects.

## Coverage limitations

- **Unresolved identities or incompatible declared bonds prevent complete graph-dependent evaluation**: 21 affected view/metric combinations, consolidated once.
- **Raw state lacks a valid topology reference; radius fallback is not bond-order specific and angle correctness is unassessed**: 2 affected view/metric combinations, consolidated once.
- **No nonbonded pairs remain after topology exclusions; absence of evaluated pairs is not a pass**: 1 affected view/metric combinations, consolidated once.
- **Receptor chemical types, explicit hydrogens and protonation unverified; interactions remain candidates and ProLIF is not prepared**: 12 affected view/metric combinations, consolidated once.
- **No provenance-verified Vina score; affinity-head predictions are not docking scores**: 3 affected view/metric combinations, consolidated once.

## Interpretation limits

- Prediction/SDF merge only after graph, identity and coordinate mapping checks; they are not independent experiments.
- Global relaxation strain and PoseBusters are molecular support, not local energy attribution.
- Routine properties without supplied targets remain in StatePacket. Missing coverage is not a pass.
- This report states risks and evidence only; no molecular edits or subsequent intervention recommendations.
