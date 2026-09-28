# DiagnosticReport — 5i0b_A__5vef_M77 / ligand_002 / t_0.75

Assessment: **risks_observed**. 1 endpoint risk regions; 1 raw-state groups; 4 coverage root causes.

StatePacket: `sp_ea21f649076801a7bbb9cad0`. Atom IDs are original zero-based tensor slots.

## Predicted endpoint risks

### local_geometry · elevated

Location: [1, 10, 14]; scope: predicted_endpoint.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 1 | N | 1 | 3.0 |
| 10 | N | 0 | 2.0 |
| 14 | N | -1 | 2.0 |

- Atoms [1, 14], order 2: length **1.0794 Å**; MMFF reference 1.4600 Å, relative deviation -26.1%. Evidence: `ev_6178e07afdcc7c01097e`.
- Atoms [10, 1, 14], center 1: angle **159.243°**; MMFF reference 119.500°, deviation +39.743°. Evidence: `ev_b54555bde86be45f8bab`.
- Atoms [1, 10], order 1: length **1.2968 Å**; MMFF reference 1.4600 Å, relative deviation -11.2%. Evidence: `ev_74f5a1f0f7cb682fd1fd`.
- Support: Atoms [1, 14]: PAINS / `azo_A(324)` substructure screening match. Evidence: `ev_66806c6ebe4d15e52a2b`.
- Support: Atoms [1, 14]: BRENK / `diazo_group` substructure screening match. Evidence: `ev_f8a253fd72d9555a82a9`.
- Support: Atoms [1, 10]: BRENK / `Oxygen-nitrogen_single_bond` substructure screening match. Evidence: `ev_cd7ce45a2564d58de779`.
- Support: Atoms [10]: maximum category probability 0.583, top-two margin 0.167; not a defect probability. Evidence: `ev_042cd49d297e543e2467`.
- Support: Molecule-level relaxation energy drop **115.441 kcal/mol**; screen threshold 20. This is not local energy attribution. Evidence: `ev_c5860b99303a6d8d8857`.

Local categorical alternatives (probabilities are not defect probabilities):

- Pair [1, 10]: declared order 1; 1.296818 Å. Alternatives: order 1: 0.972; order 2: 0.028; order 0: 0.000.
- Pair [1, 14]: declared order 2; 1.079426 Å. Alternatives: order 2: 0.999; order 1: 0.001; order 3: 0.000.
- Pair [10, 14]: declared order 0; 2.337697 Å. Alternatives: order 0: 1.000; order 1: 0.000; order 2: 0.000.

Relative to t=0.50: atom 1: N(0) → N(1); atom 10: C(0) → N(0); atom 14: C(0) → N(-1); pair [1, 10]: order 0 → 1; pair [1, 14]: order 1 → 2; pair [10, 14]: order 1 → 0. Chemical identity changed; this does not establish persistence of the same constraint.

Method limits: MMFF screening thresholds are >10% bond-length deviation and >30° angle deviation. These are uncalibrated screens; parameter applicability, especially for charged nitrogen environments, is not established.

Overlapping PAINS/BRENK matches are one screening concern, not independent defects. Catalog labels do not establish chemical impossibility or measured activity.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: prediction + sdf; 15 source observations. Full card: `risk_b8f1b2bd5579cb30`.

## Current noisy state X_t

These observations describe X_t and do not establish final failure.

### raw_graph_validity · high

Location: [0, 2, 3, 7, 9, 11, 12, 16, 17, 18, 19]; scope: observed_noisy_state.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 3 | None | 0 | 9.0 |
| 11 | O | 0 | 7.0 |
| 12 | C | 0 | 15.5 |
| 17 | N | 0 | 10.0 |
| 19 | O | 0 | 8.0 |

- Atoms [19], neutral O: declared bond-order sum **8**; conservative valence-demand lower bound 8, ordinary cap 2. Evidence: `ev_0988bd9f269cba9555d3`.
- Atoms [12], neutral C: declared bond-order sum **15.5**; conservative valence-demand lower bound 15, ordinary cap 4. Evidence: `ev_e4ef698861cb5e368112`.
- Atoms [11], neutral O: declared bond-order sum **7**; conservative valence-demand lower bound 7, ordinary cap 2. Evidence: `ev_8bb15c7cdd2546e08172`.
- Atoms [17], neutral N: declared bond-order sum **10**; conservative valence-demand lower bound 10, ordinary cap 3. Evidence: `ev_634dd7eb6f9780897bd6`.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: state; 12 source observations. Full card: `risk_472ff27779913d37`.

## Derived-evidence appendix

All source observations and evidence IDs remain in DiagnosticReport.json and StatePacket.json.

- state.bond_lengths: 35 observations, not additional independent endpoint defects.

## Coverage limitations

- **Unresolved identities or incompatible declared bonds prevent complete graph-dependent evaluation**: 22 affected view/metric combinations, consolidated once.
- **Raw state lacks a valid topology reference; radius fallback is not bond-order specific and angle correctness is unassessed**: 2 affected view/metric combinations, consolidated once.
- **Receptor chemical types, explicit hydrogens and protonation unverified; interactions remain candidates and ProLIF is not prepared**: 12 affected view/metric combinations, consolidated once.
- **No provenance-verified Vina score; affinity-head predictions are not docking scores**: 3 affected view/metric combinations, consolidated once.

## Interpretation limits

- Prediction/SDF merge only after graph, identity and coordinate mapping checks; they are not independent experiments.
- Global relaxation strain and PoseBusters are molecular support, not local energy attribution.
- Routine properties without supplied targets remain in StatePacket. Missing coverage is not a pass.
- This report states risks and evidence only; no molecular edits or subsequent intervention recommendations.
