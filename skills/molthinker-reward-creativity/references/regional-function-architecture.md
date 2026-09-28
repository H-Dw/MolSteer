# Regional Function Architecture

Use this guide when a supplied control-function table contains plausible mechanisms for a diagnosed local defect. The table supplies architectures and prerequisites, not ready-made targets, weights or evidence that a proposed transformation will improve affinity.

## Map a defect to a function family

| Local causal question | Reported architecture to inspect | Adaptation and exclusion test |
| --- | --- | --- |
| Is a bond, angle or contact outside a defensible range? | Graph-conditioned flat-bottom interval, including one-sided variants | Bind the atom roles, chemical identity, units and interval to the current hypothesis. Penalize only the offending relation; reject a target inferred only from a noisy coordinate. |
| Is a specific ligand–receptor or intramolecular pair too close? | Pairwise exclusion or minimum-distance potential | Use a one-sided penetration cost on the implicated pair or nearby patch. Severe new clashes are feasibility failures, not a tradeable average. |
| Is a required group drifting or losing its spatial role? | Anchor or anisotropic preservation potential; direction agreement for oriented features | Preserve only justified anchors and feature directions. Penalizing all atoms against an early pose can prevent repair. |
| Is a stereocenter, alkene or conjugated group geometrically at risk? | Signed chirality, dihedral window or planarity potential | Couple a smooth local term to an exact stereochemical check; a low penalty cannot certify the final assignment. |
| Does local strain persist after native continuation? | Decomposed molecular mechanics or residual-force potential | Identify the responsible bonded, torsional or nonbonded neighborhood. Require a valid graph, atom typing and protonation. Whole-molecule energy may be a guard while local repair is the primary objective. |
| Is an interaction direction or burial mismatch plausible? | Directional feature agreement, typed interface potential, conditional SASA or electrostatic complementarity | Verify donor/acceptor roles, receptor preparation, charge state and the intended exposure or orientation. Missing waters or microstates limit interpretation. |
| Is a terminal task score needed? | Docking or affinity score, targeted-property potential, candidate ranking or population weighting | Declare whether it is a differentiable objective, terminal validator, candidate selector or population operator. Do not present a black-box score or sampling weight as a coordinate gradient. |

These families correspond to the geometry, physicochemical and terminal-search groups of the supplied knowledge base. Read the actual entry for its equation, source, assumptions and permitted gradient target before specializing it. An entry's numerical constants do not automatically transfer to a new chemistry or receptor.

## Build the mechanism before choosing weights

For each independent region $r$, define the mapped chemistry $G_r$, an observable $z_{r,m}(X,G_r)$, an evidenced target set $T_{r,m}$, a physical scale $s_{r,m}$, and a function family $\phi_{r,m}$. A local residual can be written as

\[
e_{r,m}=\phi_{r,m}\!\left(\frac{\operatorname{dist}(z_{r,m},T_{r,m})}{s_{r,m}}\right).
\]

Choose the region operator $\mathcal A_r$ from the failure semantics: maximum or robust upper tail for one dominant failure, normalized mean for independent mild deviations, or matched feature assignment when partners are ambiguous. A smooth maximum may approximate a worst local gap after each relation is normalized, but it must not mix unrelated whole-molecule scores into the same maximum. Define $L_r=\mathcal A_r(\{e_{r,m}\})$. Do not multiply a region's importance by the number of alerts, atom pairs, search hits or repeated views.

Choose the relation between regions and task utility $U$ *after* local construction. First decide which regions are independent and mandatory, which are alternative explanations, and which measured gains may legitimately compensate for losses. Derive a joint objective or constrained controller whose optimum satisfies the repair predicates and whose sensitivities match that decision. A worst-gap envelope, norm, staged controller or direct sum is only a conditional candidate; none is the default. If a stage change is used, retain repaired regions through explicit persistent checks and a return rule. Do not promote an uncalibrated terminal score to a coordinate potential.

## Preserve meaning when chemistry changes

Bind every graph-dependent reference to its chemical hypothesis. Recompute valid parameters after a candidate graph change, then separately check chemical validity, the original local task and the newly exposed liabilities. A low force-field energy for a different graph does not prove that the starting graph's defect was repaired. Keep score direction, gradient path, fixed regions and the native comparison explicit. A source-reported function can be an objective, a constraint, an oracle or a selection operator; these roles are not interchangeable.
