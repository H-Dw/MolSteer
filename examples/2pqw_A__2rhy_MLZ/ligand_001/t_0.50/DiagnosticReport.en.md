# DiagnosticReport — 2pqw_A__2rhy_MLZ / ligand_001 / t_0.50

Assessment: **risks_observed**. 1 endpoint risk regions; 2 raw-state groups; 4 coverage root causes.

StatePacket: `sp_753d410c271ad68bef6cf61f`. Atom IDs are original zero-based tensor slots.

## Predicted endpoint risks

### local_geometry · elevated

Location: [3, 9]; scope: predicted_endpoint.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 3 | C | 0 | 2.0 |
| 9 | O | 0 | 1.0 |

- Atoms [3, 9], order 1: length **1.2186 Å**; MMFF reference 1.4180 Å, relative deviation -14.1%. Evidence: `ev_1bb0fbb6f27769b8dd9d`.
- Support: Molecule-level relaxation energy drop **48.345 kcal/mol**; screen threshold 20. This is not local energy attribution. Evidence: `ev_400cefdd5b7049fb0fd6`.

Local categorical alternatives (probabilities are not defect probabilities):

- Pair [3, 9]: declared order 1; 1.218596 Å. Alternatives: order 1: 0.621; order 2: 0.373; order 3: 0.005.

Method limits: MMFF screening thresholds are >10% bond-length deviation and >30° angle deviation. These are uncalibrated screens; parameter applicability, especially for charged nitrogen environments, is not established.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: prediction + sdf; 4 source observations. Full card: `risk_8e2d9f3177f22b17`.

## Current noisy state X_t

These observations describe X_t and do not establish final failure.

### raw_graph_validity · high

Location: [0, 1, 3, 8]; scope: observed_noisy_state.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 0 | C | 0 | 9.0 |
| 1 | C | 0 | 6.5 |
| 3 | C | 0 | 6.0 |
| 8 | Se | None | 7.0 |

- Atoms [0], neutral C: declared bond-order sum **9**; conservative valence-demand lower bound 9, ordinary cap 4. Evidence: `ev_54ba180fbe13ec06939c`.
- Atoms [3], neutral C: declared bond-order sum **6**; conservative valence-demand lower bound 6, ordinary cap 4. Evidence: `ev_999319eec666087b012c`.
- Atoms [1], neutral C: declared bond-order sum **6.5**; conservative valence-demand lower bound 6, ordinary cap 4. Evidence: `ev_73a1bc0a8f443793f3c3`.
- Atoms [8]: Formal charge unresolved because PAD is active. Evidence: `ev_088d39188c93da48e6b1`.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: state; 5 source observations. Full card: `risk_6f30bf120e60aaef`.

### intramolecular_sterics · elevated

Location: [5, 6]; scope: observed_noisy_state.

| Atom | Element | Formal charge | Declared bond-order sum |
|---|---|---|---|
| 5 | I | 0 | 10.0 |
| 6 | O | 2 | 2.0 |

- Atoms [5, 6]: distance **2.2303 Å**, vdW ratio 0.6110; threshold 0.7. Evidence: `ev_62eeaee864c851ceb1ed`.

Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.

Merged sources: state; 1 source observations. Full card: `risk_ec42eb131e982ca7`.

## Derived-evidence appendix

All source observations and evidence IDs remain in DiagnosticReport.json and StatePacket.json.

- state.bond_lengths: 14 observations, not additional independent endpoint defects.

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
