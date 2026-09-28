---
name: molthinker-reward-creativity
description: Design evidence-grounded molecular rewards from StatePackets, DiagnosticReports, declared goals and a supplied function knowledge base. Build a reward-design intermediate representation, compare materially different architectures, and compile an executable RewardProgram without fixed objectives or default weights. Default MolThinker reasoning mode.
---

# MolThinker: Evidence-Grounded Reward Design

Turn diagnosed molecular risks and declared goals into a testable control hypothesis. Localize the causal defect, represent the design problem explicitly, compare feasible reward architectures, and compile the RewardProgram from the selected architecture. Treat supplied knowledge as reference material rather than instructions. Keep the DiagnosticReport risk-only. Chinese instructions: [SKILL.zh-CN.md](SKILL.zh-CN.md).

## 1. Ground the decision

Bind the DiagnosticReport to its StatePacket, provenance, representation, atom mapping, coordinate frame, units and generation-time convention. Record controllable variables, fixed conditions, categorical-channel semantics, live gradient path, execution budget and user-declared outcomes. Separate observations, interpretations and experimental assumptions. Missing requirements remain unknown and cannot become favorable measurements.

Consolidate duplicate alerts into independent causal regions. For each region, record the implicated chemical identities, bonds, neighboring chemistry, receptor patch, observed values, references, competing graph or microstate hypotheses, smallest credible editable support and the surrounding structure that must remain valid. Compare with a matched native continuation when available; otherwise mark persistence as unknown rather than blocking first-pass design.

## 2. Build the RewardDesignIR

Before choosing a formula, construct a reward-design intermediate representation. Read [RewardDesignIR and architecture selection](references/reward-design-ir.md) for the required fields and decision rules.

The representation must keep these objects separate:

- independent defect regions and their direct observables;
- preservation conditions and noncompensable constraints;
- terminal task utilities and monitoring-only measurements;
- continuous coordinate objectives and discrete proposal, ranking or resampling operators;
- aggregation inside each region, priority between regions and activation over generation time;
- applicability conditions, uncertainty, prohibited shortcuts and stopping rules.

Every objective node must bind to evidence, a representation, an observable, a target set or reference, physical units, normalization origin, editable support, gradient or selection path and validity conditions. Do not add a separate objective for every alert, and do not flatten independent regions into one undifferentiated atom set.

## 3. Retrieve a function basis and test transferability

Retrieve knowledge entries by local observable, physical mechanism and control role. Preserve the source location, original mathematical form, prerequisites and gradient target. Distinguish local potentials, whole-molecule energies, terminal scores, gradient estimators and population strategies. Read [regional function architecture](references/regional-function-architecture.md) when adapting a supplied control-function table, and [knowledge-guided composition](references/knowledge-guided-composition.md) before constructing a new formula.

Build a **function basis** for every optimization goal rather than selecting one convenient entry. The basis should contain, when supported by the evidence:

- a direct repair function for the named defect;
- a complementary physical function that detects a shortcut or coupled liability;
- a preservation or feasibility mechanism for effects that cannot be traded away;
- a terminal utility or independent oracle in its valid role;
- a discrete proposal or ranking mechanism when coordinate changes cannot express the required chemical alternative.

Do not inflate the basis with weakly related entries. Every retained function must change the design decision, cover an independent failure mode, or falsify a proposed repair. Record a lineage table from each knowledge entry through its specialization to its final role. A formula without this function-level ancestry is not eligible for selection.

Map each relevant entry through **observable → evidenced target or reference → function shape → regional aggregation → gradient or selection operator → validity conditions**. Record whether the architecture is retained, specialized, combined, replaced or deferred, and explain any transformation. Parameter availability does not establish chemical applicability. A newly composed objective must not be attributed to a cited method.

When coordinates alone cannot address a credible residual, categorical alternatives may be considered through verified transition guidance, proposal ranking, resampling or explicit enumeration. Uncalibrated marginal scores are model preferences rather than joint chemical probabilities. State whether each discrete signal proposes alternatives, ranks them, affects acceptance or only explains uncertainty.

## 4. Compose normalized deficits before choosing an architecture

Convert each compatible soft objective into a nonnegative dimensionless deficit $F_k$ whose zero set means that the evidenced target is satisfied. Keep the raw measurement and transformation beside it. A maximize-type utility may join this set only after it is expressed as a target- or matched-reference deficit with a defensible scale; otherwise keep it as a terminal selector or oracle.

Construct local objectives hierarchically:

1. specialize knowledge functions into relation-level residuals;
2. aggregate duplicate or coupled relations inside each causal region without counting alert multiplicity as importance;
3. form the common evaluable set $A$ from independent, simultaneously meaningful regional and task deficits;
4. keep hard validity, severe new clashes, fixed variables and unsupported measurements outside the soft aggregation;
5. attach graph or microstate preference as a separate ranking regularizer with its actual derivative semantics.

For every $F_k$, record units before normalization, scale origin, active support, gradient path, valid graph hypothesis and the knowledge functions that justify both its shape and role. Missing objectives are omitted from $A$, not assigned zero.

## 5. Generate and compare candidate architectures

Construct every materially distinct architecture that is feasible for the RewardDesignIR. Whenever at least two compatible soft deficits remain concurrently relevant, include a candidate generated from these requirements unless their prerequisites fail:

- its primary aggregation behavior responds preferentially to the most important current normalized deficit;
- a secondary coverage mechanism prevents the remaining retained deficits from becoming gradient-inactive;
- graph or microstate preference remains a separate operator with its actual proposal, ranking and derivative semantics;
- hard constraints and unavailable measurements remain outside the soft composition.

Retrieve possible aggregation and coverage mechanisms from the supplied knowledge base, then adapt and compare them. Do not copy a previous composite equation as the candidate. Derive the composition from the present function basis, deficit semantics and execution path. State why all members of $A$ are compatible and why each excluded measurement is a constraint, oracle, discrete operator or unavailable input.

Also consider, when justified:

- a normalized direct sum for commensurate, scientifically compensable objectives;
- a hierarchical composition when several relations belong to one region and several regions must then be coordinated;
- staged or lexicographic control when a priority defect must enter an acceptable range before a task utility is activated;
- explicit constrained optimization when local validity or preservation cannot be traded for task gain;
- a hybrid continuous objective plus a separate discrete operator when chemical identity changes are required.

Do not select an architecture by familiarity. Compare candidates using causal fidelity, reference validity, normalization, gradient availability, locality, compensation safety, shortcut risk, graph-change robustness, expected interaction with native dynamics, execution compatibility and evaluation cost. Reject candidates that omit a noncompensable requirement or rely on unavailable inputs. If only one candidate survives, retain the rejected alternatives and their reasons.

Treat staged or lexicographic control as a conditional architecture, not a safe default. Reject it when a previous defect can recur during later optimization, its exit test merely matches the native baseline without a positive repair margin, or earlier objectives disappear from the scalar gradient and survive only as loose monitoring or first-order projection. If stages are necessary, retain repaired objectives through an explicit persistent mechanism and define evidence-based entry, exit and fallback margins.

## 6. Select the architecture and parameters

Use prerequisite validity and noncompensable constraints as gates. Among surviving candidates, prefer the architecture that directly measures repair of the priority region, preserves required chemistry and remains executable. Use terminal utility and convenience only after those conditions.

Derive active objectives from the RewardDesignIR. Derive scales from evidenced tolerances, reference variability or matched native behavior. Derive weights from scientific priority after normalization, not from raw numerical magnitude. For every proposed aggregator, derive and report its sensitivity to each component, then verify that its allocation matches the intended behavior: priority deficits receive stronger pressure, retained secondary objectives do not silently lose all pressure, and categorical preferences are not described as continuous gradients without a valid estimator. Reject an aggregator whose sensitivity pattern contradicts the design intent even if its scalar value appears reasonable.

Keep objective weights separate from the external guidance strength $\eta$. Before selecting either, inspect per-term raw and post-mask gradient norms, projection losses, expected injected displacement relative to the matched native step, clipping frequency and path-budget use. Use $w_k$ to express priority among normalized deficits; use $\eta$ to control the total intervention. Do not compensate for a weak or invalid objective by increasing either quantity. When calibration data are absent, require a bounded pilot and declare the parameter uncalibrated.

Derive aggregation parameters, coverage strength, graph-ranking influence and activation schedules from the intended decision rule and available calibration; do not inherit a previous composite formula, fixed objective list or fixed numeric defaults. Mark remaining assumptions as uncalibrated and define the observation that would revise them.

## 7. Compile the RewardProgram

Compile the RewardProgram only after architecture selection. It must preserve the selected regional hierarchy, objective roles and constraint semantics rather than collapsing them into a flat sum.

Include:

- region-scoped objective nodes and their normalized residuals or utilities;
- regional and cross-region aggregation operators with parameter origins;
- separate preservation, terminal and discrete components;
- evidence-bound hard constraints with baselines, thresholds, scope and rejection semantics;
- graph-change policy, reference rebinding rules and checks that the original local task remains satisfied;
- gradient target, coordinate frame, editable and fixed masks, time activation, injection convention and control budgets;
- provenance from every program component back to RewardDesignIR, diagnostic evidence and knowledge entries.

For a newly composed reward, also include the function-basis lineage, the explicit set $A$, each $F_k$ definition, the aggregation equation, the derivative allocation implied by the aggregator, and a declaration of which fields actually affect execution. Compatibility fields or inactive parameters must not be presented as active weights.

Unavailable objectives do not become zero. If the selected architecture cannot be represented or executed, mark it non-executable and retain the design; do not silently replace it with a simpler template.

## 8. Reassess changing states

Re-evaluate applicability when atom identity, charge, topology, stereochemistry, pose or receptor context changes. Rebind only references valid for the new hypothesis, then independently check the original repair criterion, new chemical liabilities and terminal utility. A low energy under a changed graph does not establish repair of the original defect.

Keep noisy state, predicted endpoint and decoded structure separate unless an explicit mapping justifies a joint objective. Rejected proposals must preserve the unmodified native path for that step. Preserve restoration state, self-conditioning and randomness for matched comparisons.

## 9. Validate and revise

Check derivative sign and finite-difference agreement at the intended support, gradient leakage, fixed-variable preservation, normalization behavior and constraint rejection. Verify each component alone, then the composed reward, so cancellation and domination are observable. Compare candidate architectures or the selected program against a matched native continuation under a common budget. Measure local repair, newly introduced defects, graph changes, native-gradient conflict, clipping and independent terminal outcomes.

Revise the RewardDesignIR or architecture when a surrogate improves without repairing the named defect, when graph changes exploit a reference-rebinding shortcut, or when native dynamics consistently erase or oppose the intervention. Distinguish reward misspecification from insufficient execution strength.

## Deliverables

Produce an evidence ledger, RewardDesignIR, function-basis and lineage table, candidate-architecture comparison, selected RewardProgram, retrieval trace, applicability and constraint contract, gradient-scale audit, and concise English and Chinese derivation notes. Keep diagnoses without a supported reward visible. Report lower surrogate cost, final chemical validity and task performance as separate outcomes.
