# MolSteer results

Four bilingual DiagnosticReports were regenerated from enriched packets, retaining 516 metric observations and 189 evidence records. Duplicate sources remain grouped by local issue; noisy-state observations and coverage limits remain separate.

| Molecule | Stage | Main endpoint finding | Reports |
|---|---|---|---|
| 5i0b / ligand_002 | 0.50 | Angle [1,14,10] 67.669°; 1–14 distance 1.2441 Å; substantial category ambiguity | [English](5i0b_A__5vef_M77/ligand_002/t_0.50/DiagnosticReport.en.md) · [中文](5i0b_A__5vef_M77/ligand_002/t_0.50/DiagnosticReport.zh.md) |
| 5i0b / ligand_002 | 0.75 | Angle [10,1,14] 159.243°; 1–14 distance 1.0794 Å; changed elements, charges and angle-center role | [English](5i0b_A__5vef_M77/ligand_002/t_0.75/DiagnosticReport.en.md) · [中文](5i0b_A__5vef_M77/ligand_002/t_0.75/DiagnosticReport.zh.md) |
| 2pqw / ligand_001 | 0.50 | C3–O9 single bond 1.2186 Å, short relative to the MMFF reference | [English](2pqw_A__2rhy_MLZ/ligand_001/t_0.50/DiagnosticReport.en.md) · [中文](2pqw_A__2rhy_MLZ/ligand_001/t_0.50/DiagnosticReport.zh.md) |
| 2pqw / ligand_001 | 0.75 | Overlapping imine catalog matches consolidated into one screening concern | [English](2pqw_A__2rhy_MLZ/ligand_001/t_0.75/DiagnosticReport.en.md) · [中文](2pqw_A__2rhy_MLZ/ligand_001/t_0.75/DiagnosticReport.zh.md) |

## Reward demonstration

- [English derivation](5i0b_A__5vef_M77/ligand_002/t_0.50/RewardDerivation.en.md) · [中文推导](5i0b_A__5vef_M77/ligand_002/t_0.50/RewardDerivation.zh.md)
- [RewardSpec](5i0b_A__5vef_M77/ligand_002/t_0.50/RewardSpec.json) · [RetrievalTrace](5i0b_A__5vef_M77/ligand_002/t_0.50/RetrievalTrace.json)
- [StatePacket](5i0b_A__5vef_M77/ligand_002/t_0.50/StatePacket.json) · [ExecutionMonitor](5i0b_A__5vef_M77/ligand_002/t_0.50/ExecutionMonitor.json)

Two endpoint interval penalties use d(1,10) bounds 1.786254–3.077091 Å and the 1–14 MMFF reference ±10%, 1.3059–1.5961 Å. The same angle's MMFF flag is supporting evidence rather than an additional reward. Two raw-state protein-overlap penalties form a separate group and were not included in the endpoint trial.

Only atoms [1,10,14] were movable on the coordinate copy. Penalty decreased from 0.03282699 to 5.617×10⁻¹³; maximum finite-difference error was 4.458×10⁻¹⁰; maximum atom displacement was 0.124730 Å. No new evaluated bond-window violations or protein overlaps were introduced. The 1–14 single-bond probability is approximately 0.341; chemical identity and improved final quality remain unestablished.

[Artifact verification](verification.json), [36 passing tests](../validation/test_summary.json) and [environment changes](../validation/environment_changes.json) document validation and dependencies.
