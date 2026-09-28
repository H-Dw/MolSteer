# DiagnosticReport — 2pqw_A__2rhy_MLZ / ligand_001 / t_0.75

Assessment: **risks_observed**. 1 endpoint risk regions; 2 raw-state groups; 4 coverage root causes.

StatePacket: `sp_75ca8726467e4fcffe5f9d94`. Atom IDs are original zero-based tensor slots.

## Predicted endpoint risks

### structural_screening · contextual

Location: [2, 3, 8]; scope: predicted_endpoint.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 2 | C | 0 | 4.0 |
| 3 | N | 0 | 2.0 |
| 8 | N | 0 | 3.0 |

- Atoms [2, 8]: BRENK / `imine_1` substructure screening match. Evidence: `ev_5c322e469c8e8af683f2`.
- Atoms [2, 3, 8]: BRENK / `imine_2` substructure screening match. Evidence: `ev_1b7dcd8ca42617633597`.

Local categorical alternatives (probabilities are not defect probabilities):

- Pair [2, 3]: declared order 1; 1.351035 Å. Alternatives: order 1: 0.984; order 2: 0.015; order 0: 0.001.
- Pair [2, 8]: declared order 2; 1.321001 Å. Alternatives: order 2: 0.881; order 1: 0.119; order 0: 0.000.
- Pair [3, 8]: declared order 0; 2.363239 Å. Alternatives: order 0: 1.000; order 1: 0.000; order 2: 0.000.

Relative to t=0.50: atom 3: C(0) → N(0); atom 8: C(0) → N(0); pair [2, 8]: order 1 → 2. Chemical identity changed; this does not establish persistence of the same constraint.

Overlapping PAINS/BRENK matches are one screening concern, not independent defects. Catalog labels do not establish chemical impossibility or measured activity.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: prediction + sdf; 4 source observations. Full card: `risk_2d49cf0c4744da1b`.

## Current noisy state X_t

These observations describe X_t and do not establish final failure.

### raw_graph_validity · high

Location: [1, 3, 4, 6]; scope: observed_noisy_state.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 1 | C | 0 | 5.5 |
| 3 | N | None | 4.5 |
| 4 | N | 0 | 7.5 |
| 6 | C | 0 | 8.0 |

- Atoms [4], neutral N: declared bond-order sum **7.5**; conservative valence-demand lower bound 6, ordinary cap 3. Evidence: `ev_c946a03d800e41cd6cbd`.
- Atoms [6], neutral C: declared bond-order sum **8**; conservative valence-demand lower bound 8, ordinary cap 4. Evidence: `ev_72bbdf016ffec105c3f4`.
- Atoms [1], neutral C: declared bond-order sum **5.5**; conservative valence-demand lower bound 5, ordinary cap 4. Evidence: `ev_71f8cd2c2c495524d09a`.
- Atoms [3]: Formal charge unresolved because PAD is active. Evidence: `ev_f81c9df13302731b55ed`.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: state; 5 source observations. Full card: `risk_d7607ec53324f3ad`.

### intramolecular_sterics · elevated

Location: [7, 10]; scope: observed_noisy_state.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 7 | C | 0 | 4.0 |
| 10 | N | -1 | 7.0 |

- Atoms [7, 10]: distance **2.1364 Å**, vdW ratio 0.6474; threshold 0.7. Evidence: `ev_1428d0440c7228a2d113`.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: state; 1 source observations. Full card: `risk_4d99379b773e4217`.

## Derived-evidence appendix

All source observations and evidence IDs remain in DiagnosticReport.json and StatePacket.json.

- state.bond_lengths: 13 observations, not additional independent endpoint defects.

## Coverage limitations

- **Unresolved identities or incompatible declared bonds prevent complete graph-dependent evaluation**: 20 affected view/metric combinations, consolidated once.
- **Raw state lacks a valid topology reference; radius fallback is not bond-order specific and angle correctness is unassessed**: 2 affected view/metric combinations, consolidated once.
- **Receptor chemical types, explicit hydrogens and protonation unverified; interactions remain candidates and ProLIF is not prepared**: 12 affected view/metric combinations, consolidated once.
- **No provenance-verified Vina score; affinity-head predictions are not docking scores**: 3 affected view/metric combinations, consolidated once.

## Interpretation limits

- Prediction/SDF merge only after graph, identity and coordinate mapping checks; they are not independent experiments.
- Global relaxation strain and PoseBusters are molecular support, not local energy attribution.
- Routine properties without supplied targets remain in StatePacket. Missing coverage is not a pass.
- This report states risks and evidence only; no molecular edits or subsequent intervention recommendations.
