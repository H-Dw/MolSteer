# Knowledge-Guided Function Composition

Use this reference when MolThinker must create a reward that is not copied from one knowledge entry. The purpose is to preserve the reported function mechanisms while composing them into an objective matched to the diagnosed state.

## 1. Build a function-basis ledger

For each optimization goal, retrieve functions by mechanism rather than by keyword alone. Record:

| Field | Required meaning |
| --- | --- |
| Knowledge entry | Source identifier, location and original equation |
| Physical question | What observable failure or desired relation the function measures |
| Preconditions | Graph, atom typing, protonation, receptor preparation, reference or population requirements |
| Native role | Coordinate potential, categorical preference, constraint, terminal score, oracle, selector or estimator |
| Specialization | Bound atoms, receptor patch, target set, scale, direction and support |
| Composition role | Direct repair, coupled-liability control, preservation, task deficit or counterevidence |
| Exclusion test | Evidence that would make the function inapplicable or redundant |

A goal is not adequately represented merely because one function mentions the same property. Seek the smallest set that represents the direct repair mechanism, its main shortcut, and the required validity boundary. Stop retrieving when additional entries neither change the formula nor add an independent falsification route.

## 2. Specialize functions into normalized deficits

Let a knowledge-derived observable be $z_k(X,G)$ with an evidenced acceptable target set $T_k$ and scale $s_k$. Define

\[
F_k(X,G)=\phi_k\!\left(
\frac{\operatorname{dist}(z_k(X,G),T_k)}{s_k}
\right)\ge 0,
\]

where $F_k=0$ means the target is satisfied. Preserve one-sided behavior for minimum distances, collision penetration and directional goals. Use periodic or signed residuals for torsion, chirality and orientation where required. Do not convert an unavailable target into a zero residual.

When several measurements describe one causal region, first remove duplicate views and then choose a region operator matching the failure semantics:

- smooth maximum or robust upper tail for a single dominant local failure;
- normalized mean for independent mild deviations;
- assignment or soft matching for ambiguous interaction partners;
- a force-field or decomposed physical energy only when graph, typing and microstate prerequisites hold.

This produces region deficits $F_r$. Whole-molecule task utilities may become deficits only when a target, matched reference or non-regression margin makes their zero set meaningful.

## 3. Derive a composition only when several targets require it

When the selected core-target set contains several independent deficits, first decide whether they are compensable, jointly mandatory, ordered by a validated priority, or incompatible in the actual editable space. A single sufficient target needs no cross-target aggregator. Derive any aggregator instead of selecting a remembered final equation.

First state the desired response law and acceptable joint set. Decide whether the controller should repair one cause, satisfy the intersection of several target sets, permit a measured tradeoff, enforce priority, or solve for a feasible common direction. Derive a composition whose zero set and sensitivities have that behavior. A maximum, norm, constrained solve or scalar sum is a candidate only when its assumptions fit. Keep graph or microstate operations separate unless a verified estimator connects them to the same derivative path. The provided function table may justify the constituent local potentials while offering no citation for the new aggregation operator.

The generated composition must answer:

- which component receives marginal pressure in each relevant regime and whether a failed mandatory target can be compensated;
- whether any retained component can receive exactly zero pressure before its repair predicate passes;
- how behavior changes when two deficits exchange rank or approach their targets;
- which parameters control curvature, smoothing, priority and graph ranking, and how their scales were obtained;
- which knowledge entries justify each *local function* and which composition rules are newly derived.

Derive the component sensitivities symbolically or automatically and compare them with the intended response law. Reject the composition if these differ or if a named repair can remain failed at its optimum. Use a hierarchy only when the mechanism structure requires one. Do not place hard constraints, missing measurements or unrelated raw scores in a common soft set.

## 4. Add knowledge-derived components by role

A composed reward can draw on several knowledge families without becoming a flat sum:

- local interval, exclusion, orientation, torsion or stereochemical functions define repair residuals;
- molecular mechanics or residual-force functions cover coupled intramolecular strain when their prerequisites hold;
- typed contact, directional interaction, electrostatic or exposure functions define interface deficits when receptor and microstate evidence supports them;
- anchor, pocket occupancy or movement functions preserve justified context rather than forcing all atoms toward an early pose;
- affinity or docking functions become task deficits, terminal selectors or independent oracles according to their derivative availability;
- categorical preferences, explicit enumeration or population weighting operate outside continuous coordinate gradients unless a verified estimator connects them.

Keep severe new clashes, sanitization, disconnected graphs, fixed atoms and mandatory identities as rejection conditions. Their satisfaction must not be purchasable by improvements elsewhere.

## 5. Reject unreliable composition patterns

Reject or revise a candidate when any of these holds:

- raw observables with incompatible units are added directly;
- multiple alerts for one defect create multiple weights;
- a staged controller removes an unresolved objective from the scalar gradient;
- a stage unlocks at equality with a native baseline rather than an evidenced positive margin;
- a broad flat-bottom makes the named defect gradient-free while the defect remains physically relevant;
- a model head is optimized without a matched reference and independent counterevidence;
- a black-box score, categorical argmax or graph cost is described as differentiable when no derivative estimator exists;
- graph-dependent references are rebound without checking the original repair predicate;
- internal weights are increased to compensate for insufficient external guidance strength, or $\eta$ is increased to compensate for an invalid objective.

## 6. Calibrate objective allocation and guidance strength separately

For every component, measure the coordinate gradient before masking, after masking and after projection. Compare the proposed injected displacement with the matched native update and record clipping and path-budget consumption. Diagnose:

- semantic domination: one normalized deficit is scientifically over-prioritized;
- numerical domination: one derivative scale overwhelms the others despite comparable priority;
- execution attenuation: masking, projection or generator Jacobians remove most of a valid gradient;
- native conflict: the native update repeatedly erases or reverses the intervention.

Revise $F_k$, $s_k$ or $w_k$ for semantic or numerical imbalance. Revise editable support, projection or the external strength $\eta$ for execution attenuation. Select $\eta$ against a bounded post-projection guidance-to-native displacement regime and trust-region behavior; do not infer it from reward magnitude alone.

## 7. Required derivation record

The final derivation must show:

1. the diagnosed causal regions and declared optimization goals;
2. the retrieved function basis and applicability decisions;
3. every specialized $F_k$ and its normalization origin;
4. the members and exclusions of $A$;
5. the generated aggregation mechanism, its sensitivity behavior and rejected alternatives;
6. derivative allocation across components and the separate $\eta$ calibration;
7. hard constraints, graph-change semantics and independent falsification measurements.

If these items cannot be supported, retain the design as a hypothesis and do not label it executable.
