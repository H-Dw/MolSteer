# Outcome-aware MolThinker and execution

MolThinker can now consume the original intermediate diagnosis **and** provenance-bound matched native outcomes. Its output is an executable reward program with continuous physical proxies, explicit chemical trials and counterevidence-driven revisions. The implementation is a deterministic policy; no external language-model call is claimed.

## Evidence and attribution

`molthinker/outcomes.py` verifies the enriched StatePacket, comparison sidecar identity/hash, molecular subject, original diagnosis lineage and dense native-reference replay. The diagnosis remains unchanged. Intermediate atom identities are compared with native terminal identities; obsolete graph-bound constraints are retained as historical evidence rather than active targets. Native terminal force and buried-polar evidence remain available to prioritize local hypotheses.

Native replay retains current tensors, self-conditioning, all saved RNG states, batch layout, precision and the exact integration grid. Measurements are collected at every remaining step. Unsupported or disconnected chemical states are explicitly unavailable, never zero-energy observations.

The implemented experimental composition is

\[
R_t=w_A\,2\tanh[(pK_d-pK_d^0(t))/2]
+w_S\log\frac{1+S^0(t)/5}{1+S/5}
+w_I\frac{I-I^0(t)}2
+w_D\frac{D^0(t)-D}2
-w_P(P-P^0(t)).
\]

Superscript 0 denotes the matched unguided prediction at the same time. Initial weights are `(1, 1, 0.5, 0.25, 1)`. Scales are explicit experimental settings, not calibrated physical equivalences. Pure native self-correction receives zero incremental credit. Subtracting a fixed baseline alone does **not** change the derivative; replacing obsolete targets and introducing new terms changes the search. The affinity term also uses a baseline-centered saturation.

Knowledge rows P01, G04, P06, P05 and S05 supply mechanisms for strain, directional interactions, exposed area, empirical scoring and multi-objective comparison. They do not establish the validity of this particular composition or its weights.

## Continuous objectives

- `mmff_bridge.py`: MMFF94s energy minus a frozen same-graph local minimum. Heavy atoms stay fixed while hydrogens relax. The heavy-atom force is exposed through a first-order PyTorch autograd bridge. Failed parameterization/minimization or a materially lower-than-reference energy causes an explicit unavailable result. This is local strain, not a global conformational free energy.
- `interface_energy.py`: donor/acceptor typing, smooth distance and orientation factors, and saturation per ligand polar atom. Protein donor hydrogens are explicit; ligand donor direction and protein acceptor direction use heavy-neighbor approximations. Smooth SASA estimates buried polar area unsatisfied by these direct contacts. The desolvation proxy is `sum(buried_area / 20 Å² * (1-satisfaction))`; it is **not** a solvation free energy.
- `outcome_reward.py`: graph-dependent measurements are recomputed after a graph change. Live head derivatives pass through the model. Component gradients are logged; components opposing strain improvement can be projected onto its non-worsening half-space. This is a structure-prioritized, first-order search direction, not the unmodified scalar reward gradient or a guarantee about the next/final state.

MMFF already includes bonded and intramolecular nonbonded terms. The old geometry energy is not added again. Hard chemistry/clash checks remain separate. Missing water, receptor ensembles, calibrated affinity uncertainty and pH-dependent populations remain explicit limitations.

## Active chemical hypotheses

`graph_search.py` enumerates mapped tautomers, charge normalization/reionization, atom substitutions, formal-charge changes and bond-order alternatives. It preserves original atom slots and validates tensor round trips, connectivity and chemistry. Aromatic and Kekule representations are tried with preference for fewer unnecessary categorical changes. Unrepresentable hydrogen identities are rejected and logged. Candidate families are interleaved to prevent substitutions from excluding microstates.

Each trial edits only the target molecule's selected categorical **current-state** tensors, then calls the live model with unchanged self-conditioning. The actual predicted endpoint must change graph and pass chemistry, geometry, displacement, reward-gain and independent-score tests. Rejected trials do not mutate sampling tensors or consume sampler randomness. A frozen marginal model-ranking cost is a regularizer, not a calibrated graph probability. Acceptance changes current categorical tensors; subsequent self-conditioning is updated only by the native sampler.

The delivered 100-step experiment checks steps 60, 75 and 85, at most 16 candidates each, six edited slots and three commits. These are configurable experiment settings, not model-independent universal times. Other grids must supply appropriate step indices. Atom insertion/deletion and an equilibrium microstate population are not implemented. Zero accepted candidates is a valid search outcome and must be reported without claiming a new molecule.

## Counterevidence and revision

`oracles.py` performs Vina score-only evaluation with prepared receptor hashes and fixed heavy-atom poses. Initialization and evaluation preserve Python, NumPy and Torch RNG. Vina supplies no invented derivative. It is independent of the live head and continuous objective, but **not held out** once used for selection.

The execution loop measures actual head/strain gradient cosine and component norms. Three consecutive cosine values below −0.25, or head improvement above 0.03 accompanied by Vina deterioration above 0.15 kcal/mol or strain deterioration above 2 kcal/mol, can trigger `revise_from_evidence`. A child program halves affinity weight and increases strain weight; Vina counterevidence also increases interface weights. Every revision carries parent identity, evidence, reasons and unchanged η. This is a bounded heuristic policy whose terminal benefit must be tested, not assumed.

The supplied experiment observed that this automatic revision policy underperformed its fixed continuous counterpart. Retain that negative result. Do not promote a revised program solely because its trigger fired or its own weighted reward increased. Compare original objective values, independent scores, local residuals and the matched native outcome; retain nondominated alternatives and route counterevidence for further objective revision.

## Interfaces and runtime

`python -m molsteer think` defaults to outcome-aware creativity when the StatePacket contains outcome evidence. It then requires `--native-reference` and `--base-program`; evidence cannot be silently ignored. Existing selection and legacy execution remain available. `make_reward` and `run_suffix` dispatch `evaluator: outcome_aware` to the new implementation. Core chemical/physical primitives are reusable; the delivered suffix loop is verified on the FLOWR.ROOT adapter, not every diffusion implementation.

The runtime saves active program, lineage, frozen force-field references, conflict streak, categorical commits and cumulative displacement alongside native tensors, SC and RNG. Restarts must preserve these fields. External scoring and rejected probes are isolated from sampling. Intermediate and final structures are emitted by the native model head; no final coordinate minimization is substituted for inference.

## Reproduction

Use the existing remote environment and a fresh output directory. The bounded scripts reuse the prior authentic runtime and prepared comparison evidence.

```bash
cd /data1/dhuang/flowr_root
export PYTHONPATH=MolSteer/src
.venv/bin/python MolSteer/scripts/build_outcome_reference.py \
  --config output/molsteer_monitor_20260923/continuations/adaptive_same_budget/execution.json \
  --evaluation /data1/dhuang/flowr_root/output/molsteer_independent_evaluation_20260923 \
  --output /path/reference-output
.venv/bin/python MolSteer/scripts/validate_outcome_guidance.py \
  --root /path/reference-output --evaluation output/molsteer_independent_evaluation_20260923
.venv/bin/python MolSteer/scripts/audit_outcome_guidance.py --root /path/reference-output
.venv/bin/python -m unittest discover -s MolSteer/tests
```

To produce an agent-consumable program directly, pass `--packet StatePacket.with_outcomes.json --report DiagnosticReport.json --knowledge MolSteer/knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md --native-reference /path/reference-output/native_reference.json --base-program active_RewardProgram.json --output /path/RewardProgram.json` to `.venv/bin/python -m molsteer think`. Resolve input paths to their existing provenance-bound files. English and Chinese derivations are written beside the program.

`evaluate_guidance_independently.py --cases-json` accepts explicit saved-stage paths. Prepare its receptor subdirectory using the existing preparation workflow. Assessment includes original-pose Vina/Vinardo, separately optimized copies, ProLIF, continuous force-field evidence, descriptors and PoseBusters. Vinardo is related to Vina; their agreement is not independent experimental confirmation. Compare saved-SDF assessments with saved-SDF assessments; live tensor strain uses a frozen reference and should not be mixed with separately minimized SDF references.

See [environment](ENVIRONMENT.md) and [outcome evidence schema](OUTCOME_EVIDENCE.en.md).

## Primary implementation references

- [RDKit force-field energy and gradient API](https://rdkit.org/docs/source/rdkit.ForceField.rdForceField.html)
- [RDKit tautomer enumeration API](https://www.rdkit.org/docs/cppapi/MolStandardize_2Tautomer_8h_source.html)
- [AutoDock Vina Python API](https://autodock-vina.readthedocs.io/en/latest/vina.html)
