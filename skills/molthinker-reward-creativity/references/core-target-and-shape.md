# Core-target discovery and mathematical shape audit

Use this reference before proposing a reward equation. A diagnostic flag is a measurement, not an optimization objective. The number of flags, retrieved functions or available gradients must not determine the number of objectives.

## Discover the smallest sufficient target set

1. State the user outcome and the independent final measurement that would show success. Bind the present molecule, generation state, chemical hypothesis and receptor preparation.
2. Cluster alerts that share atoms, a physical mechanism or a direct-measurement/proxy relationship. For each cluster, distinguish an observed defect from a proposed cause. Preserve competing graph, protonation, pose and model-error explanations.
3. Compare the matched native continuation when it exists. A transient noisy-state clash that disappears without intervention is not automatically a repair target. An endpoint defect that persists, worsens, or predicts an independently measured final failure deserves stronger consideration. Mark missing comparison as unknown.
4. For each candidate core target, write a **repair predicate** in the representation where it can be measured, a **preservation predicate**, an editable intervention path, and a falsifier. Ask whether improving the proposed observable can leave the named defect unchanged or worsen a more direct measurement.
5. Retain the smallest set of independent targets needed to distinguish successful repair from these shortcuts. Classify every other signal as a constraint, validity gate, mechanism modifier, terminal selector, counterevidence or deferred observation. Do not make a screening threshold a hard physical boundary without validation.

Rank candidates by evidence and decision gates, not an arbitrary weighted score: validity of reference and chemistry; persistence or credible impact; causal specificity; controllability with the actual live derivative; preservation feasibility; independent evaluation. An objective that fails a required gate is deferred even if it is easy to differentiate.

## Derive shape from the acceptable set

For a retained observable `z(X,G)` define the evidenced acceptable set `T(G)`, its uncertainty, and a scale with physical meaning. Choose a loss whose zero set, symmetry and derivative fit that set:

- A two-sided interval has zero cost throughout a justified tolerance. A one-sided exclusion has no attraction beyond its safe boundary. Neither is a universal substitute for molecular mechanics or a directional interaction.
- Angular and torsional variables require circular or signed geometry. Handedness requires a stereochemical predicate; a generic distance cannot certify it.
- Ambiguous contact partners call for an assignment or soft matching mechanism with an explicit partner set. Choosing one nearest atom merely because it triggered an alert can steer to the wrong pose.
- Coupled bond, angle and torsion strain may require a validated graph-dependent physical energy or a local force decomposition. Adding separate proxy penalties can count one deformation repeatedly.
- Graph or microstate changes require branch-specific references and acceptance logic. A frozen categorical preference is not a coordinate gradient.

Select curvature and smoothing from a justified response law. Check the loss and its first derivative below, at and above the target; its Hessian or local conditioning near active boundaries; singularities and flat regions; rigid-motion and atom-permutation invariance; and what happens as the graph, receptor pose or state view changes. A broad zero-gradient window that includes an unresolved defect is invalid. A loss that keeps attracting after the actual repair predicate passes may damage the native trajectory.

## Derive composition from compatibility

First form region objectives by eliminating duplicated measurements. Then compare their gradients in the *same live editable coordinates*, after the actual generator Jacobian, mask and constraint projection. Record norms and pairwise directions; a saved endpoint gradient alone is insufficient. Select an architecture according to the relationship:

- One sufficient repair target: use its localized potential with independent preservation checks. Do not manufacture extra objectives for visual balance.
- Several compatible and scientifically compensable targets: a dimensionless aggregate may be valid if its marginal sensitivities and zero set preserve the intended tradeoff. Derive it for this case; no aggregate is the default.
- Simultaneous targets that must each pass: use a set-distance, worst-violation, constrained or other noncompensating construction whose zero set is the intersection of acceptable sets. Verify whether any target becomes gradient-inactive while still failing.
- Conflicting gradients or noncompensable preservation: solve a constrained common-descent or priority problem in the feasible tangent space, or decline guidance. Report the optimization problem and solver evidence; scalar weights do not resolve infeasibility.
- Different graph hypotheses: compare discrete branches by matched feasible continuation, and optimize coordinates within each branch. Do not average incompatible gradients.

An aggregation operator is a new mathematical design unless the retrieved knowledge entry actually supports that operator in the same role. In particular, population softmax weights are not evidence for applying log-sum-exp to deficits within one molecule. Record the knowledge lineage of **local function shapes** separately from the derivation and validation of the **composition rule**.

## Reject a mathematically plausible but ineffective reward

For each candidate architecture, check (a) the exact zero or optimal set against the repair and preservation predicates; (b) component sensitivities, including active and inactive regimes; (c) a feasible descent direction in the live editable space; (d) predicted displacement after masking, projection, clipping and the actual injection sign; (e) new clashes, strain and graph changes; and (f) matched native continuation with independent terminal outcomes. If the reward decreases but the named defect or outcome does not improve, revise the observable or mechanism before tuning its weight. Keep guidance strength separate from the reward definition.

Deliver the selected core targets and rejected alternatives, a function-shape derivation with source and assumptions, candidate compositions and their sensitivity/feasibility checks, a bounded execution contract, and explicit `not_run` results for unavailable validations. An unsupported design is a hypothesis, not an executable RewardProgram.
