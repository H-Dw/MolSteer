---
name: novel_conf_molreader-read-state
description: Measure molecular generation snapshots and assemble provenance-bound StatePackets, including geometry, chemistry, interface evidence and steering readiness. Use for feature extraction and evidence aggregation.
---

# MolReader State Evidence

Produce a StatePacket that locates current defects and preserves the evidence required for diagnosis and reward reasoning. Keep the original generation artifacts unchanged. Chinese instructions: [SKILL.zh-CN.md](SKILL.zh-CN.md).

## Establish representation and identity

Distinguish the current noisy state, the head endpoint estimate and the decoded molecule. Preserve original zero-based tensor-slot IDs, element identities, formal charges, bond orders and coordinate frames. Verify mapping before combining representations.

Confirm the generator's time direction from its actual metadata and sampling behavior. Do not assume larger time means more noise. Distinguish categorical probabilities, logits and sampled hard labels; sampled labels do not measure certainty.

When adapting unfamiliar inputs, consult the [evidence input contract](references/input-contract.md).

## Collect evidence and readiness

- Measure valence conflicts, connectivity, local geometry, severe overlaps, chemistry screening alerts and interface observations. Retain evaluated denominators, numerical references, units, assumptions and source identifiers.
- Attach category alternatives and chemically comparable stage changes to local evidence. Keep routine descriptors and affinity predictions available without inventing target ranges.
- Record coordinate availability, frame alignment, graph validity, formal-charge completeness, protonation validation, partial charges and receptor preparation separately.
- Record endpoint availability, live derivative access, editable degrees of freedom, masks, active particle population and evaluation budget only when supported by supplied evidence or explicit runtime declarations.
- Keep unknown objectives, anchors, shape references, off-target scores and intervention feedback unknown. An active-slot mask is not permission to move atoms; saved candidates do not establish a live weighted population.

## Preserve measurement meaning

Missing or unevaluable measurements are not zero or passed. Distinguish calculation success from molecular acceptability. MMFF parameter availability does not validate the chemical hypothesis. Formal charges do not substitute for partial charges; implicit hydrogens do not validate protonation.

Keep protein-coordinate overlap screens separate from prepared interaction typing. Shared prediction/SDF coordinates are not independent corroboration. Local relaxation on a documented copy does not modify the observed input state.

## Deliver and verify

Validate target, ligand, stage, original atom IDs, source hashes and coordinate provenance. Enrichment creates a new packet identity linked to its parent. Preserve all existing observations and evidence links.

Return the StatePacket, measured representations and consolidated coverage limitations. If diagnosis is requested, produce a separate risk-only DiagnosticReport. Reward design belongs to MolThinker.
