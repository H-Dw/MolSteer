# RewardDesignIR and Architecture Selection

Use this reference after grounding a StatePacket and DiagnosticReport and before writing a reward formula. RewardDesignIR preserves the scientific meaning of the design problem so that architecture choice and RewardProgram compilation remain auditable.

## 1. Required objects

| Object | Required content | Decision purpose |
| --- | --- | --- |
| `identity` | Target, sample, stage, StatePacket identity, DiagnosticReport identity and source digests | Prevent cross-state evidence mixing |
| `control_context` | Time convention, remaining horizon, controlled representation, continuous and categorical variables, editable and fixed support, live derivative path, native update semantics and budgets | Determine which operations are executable |
| `regions` | One record per independent causal defect or design region | Preserve locality and prevent unrelated defects from being averaged together |
| `objective_nodes` | Direct repair observables with targets, residuals, scales, roles and validity conditions | Define what reward improvement means |
| `preservation_conditions` | Required identities, interactions, pose or scaffold properties that must not regress | Prevent score-improving shortcuts |
| `hard_constraints` | Noncompensable validity and safety conditions with baselines, thresholds, scope and rejection semantics | Keep infeasible proposals outside the tradeoff |
| `terminal_utilities` | User-declared task outcomes, direction, target or interval, evaluator and uncertainty | Keep task performance separate from local repair |
| `discrete_operators` | Candidate source, edited channels, ranking or resampling rule, trust policy and acceptance checks | Represent identity changes without fictitious coordinate gradients |
| `architecture_candidates` | Feasible compositions and rejected alternatives | Make architecture selection comparative rather than habitual |
| `selected_architecture` | Selection gates, decision rule, assumptions and expected failure modes | Explain why this structure was chosen |
| `execution_contract` | Gradient target, coordinate frame, masks, time activation, step and path budgets, rejection rollback and restoration state | Preserve mathematical meaning during execution |
| `feedback_contract` | Monitored signals, revision triggers and stop conditions | Close the loop between generation outcomes and reward revision |
| `provenance` | Evidence, knowledge and declared-goal links for every design element | Support audit and later revision |

## 2. Region records

Each region represents one causal question. Keep separate regions separate even when they overlap in atom support.

A region record contains:

- a stable region identifier and priority class;
- the evidence view: noisy state, predicted endpoint, decoded structure or an explicitly mapped combination;
- atom, bond, stereochemical and receptor identities with local neighbors;
- current graph or microstate hypothesis and credible alternatives;
- direct observables, measured values, units, references and uncertainty;
- persistence under matched native continuation when available;
- severity relative to an interpretable tolerance and confidence in local attribution;
- editable support, fixed boundary and receptor patch;
- a repair predicate stating what must become true;
- preservation predicates stating what must remain true;
- prohibited shortcuts that could improve a surrogate without repairing the region;
- conditions that suspend or retire the region.

Alerts, proxies and whole-molecule support belong under the same region when they describe one cause. They do not become independent objective nodes unless they measure an independent failure.

## 3. Objective nodes

An objective node records:

| Field | Meaning |
| --- | --- |
| Role | Local repair, preservation, terminal utility or monitoring only |
| Observable | A measurable function of the controlled representation |
| Target | An evidenced point, interval, set, ordering or direction |
| Residual | Distance from the target with explicit sign and zero set |
| Scale | Physical tolerance, reference variability or matched native scale |
| Function family | Interval, exclusion, orientation, stereochemical, energy, similarity or task utility architecture |
| Support | Atoms, bonds, pose variables or categorical channels allowed to influence the node |
| Operator | Coordinate gradient, categorical proposal, ranking, resampling, terminal evaluation or hard rejection |
| Validity | Chemical, geometric, preparation and model assumptions required for interpretation |
| Coverage | Available, partial, unavailable or conditional, with reason |
| Provenance | Diagnostic evidence, goal declaration and knowledge entry |

Define a normalized local residual as

\[
e_{r,m}=\phi_{r,m}\!\left(\frac{\operatorname{dist}(z_{r,m},T_{r,m})}{s_{r,m}}\right),
\]

where the zero set of \(\phi\) matches the evidenced acceptable set. Keep the raw observable beside its normalized residual.

Use this preference order for scale origins:

1. validated task-specific tolerance or target distribution;
2. a matched reference ensemble or matched native continuation;
3. a physically interpretable literature or force-field tolerance whose applicability is checked;
4. a declared experimental scale with an explicit revision trigger.

Do not infer scale from the number of alerts, number of atom pairs or current residual magnitude alone. If initial gradient diagnostics are available, use them to detect numerical domination and revise the parameterization without erasing the physical meaning of the scale.

## 4. Candidate architecture families

Generate all families whose prerequisites are satisfied. Hard constraints remain outside every soft aggregation.

### Normalized direct sum

Retrieve a linear aggregation architecture only when objectives are independently meaningful, normalized and scientifically compensable. Reject it when a severe local failure could be bought away by many small improvements. Derive its actual expression from the active objective nodes rather than retaining a generic template.

### Dominant-gap aggregation with persistent secondary progress

Use when the largest normalized deficit should dominate while secondary objectives must continue to improve. Retrieve a primary operator with the required dominance behavior and, when needed, a distinct coverage operator that prevents premature gradient starvation. The common set contains only evaluable, compatible soft objectives; it excludes missing measurements, hard constraints and unrelated terminal scores. Derive the final mathematical form from the retrieved operators and current objective semantics, then inspect its component sensitivities instead of copying a stored composite formula.

### Staged or lexicographic control

First minimize the highest-priority regional loss until its repair predicate and non-regression margin pass. Then activate the next region or terminal utility while retaining the repaired region as a constraint. Define stage-entry, stage-exit, regression and fallback conditions from observables rather than elapsed steps alone.

### Explicit constrained objective

Use when task utility is meaningful but local validity, severe collision, fixed identity or preservation requirements cannot be compensated. State whether constraints are enforced by projection, proposal rejection, acceptance filtering or a separate feasibility procedure.

### Hybrid continuous and discrete control

Keep coordinate-differentiable loss separate from a discrete operator. Retrieve a categorical preference or ranking mechanism only for the role supported by its source and available scores. Frozen marginal scores provide relative model preference, not a calibrated chemical probability. State how alternatives are proposed, when the operator ranks them, which chemical checks gate acceptance and whether any discrete signal affects a continuous derivative. If no alternative-generation or acceptance mechanism exists, retain the discrete term as monitoring-only.

## 5. Candidate comparison

For every candidate, record a pass, conditional pass or failure with evidence for each criterion:

1. causal fidelity to each priority region;
2. validity and provenance of targets and references;
3. objective independence and absence of duplicate weighting;
4. physical normalization and parameter identifiability;
5. gradient or selection-path availability;
6. localization and expected gradient leakage;
7. compensation safety and hard-constraint coverage;
8. shortcut and reward-hacking exposure;
9. behavior under graph, charge, stereochemistry or pose changes;
10. expected alignment or conflict with native dynamics;
11. execution compatibility, budget and evaluation cost;
12. independent measurements that can falsify the hypothesis.

Apply selection gates in this order: required inputs and validity, noncompensable constraints, direct repair fidelity, graph-change robustness, execution feasibility, then terminal utility and efficiency. Do not collapse these gates into an arbitrary meta-score. Record why the selected candidate is preferable and what evidence would reverse the decision.

## 6. RewardProgram compile contract

The RewardProgram is compiled from the selected RewardDesignIR. It contains no objective or parameter that lacks an intermediate-representation origin.

Preserve these sections:

- identity and selected architecture;
- regional objectives with raw observables, normalized residuals and repair predicates;
- region-level and cross-region aggregations;
- preservation conditions and terminal utilities;
- hard constraints with reference baselines and rejection behavior;
- discrete operators and their proposal, ranking and acceptance roles;
- graph-transition and reference-rebinding policy;
- time activation and any state-dependent stage transitions;
- gradient or selection target, masks and control budgets;
- expected failure modes, monitoring signals, revision triggers and stop conditions;
- complete evidence and knowledge provenance.

Compilation must preserve roles. A constraint cannot become a soft penalty merely because that form is easier to execute. A terminal evaluator cannot become a coordinate gradient without a valid derivative path. Independent regions cannot be merged solely because they share atoms. When the selected architecture is unsupported by the available execution mechanism, report the incompatibility and preserve the intended design for later support.

## 7. Outcome-driven revision

Use matched native behavior and independent final measurements to distinguish four cases:

- the objective and named defect both improve: retain the hypothesis and continue calibration;
- the objective improves but the defect does not: revise the observable, target, support or shortcut controls;
- the gradient is locally correct but native dynamics erase it: revise activation, injection or budget before redefining the objective;
- chemical validity, preservation or terminal outcomes regress: reject the candidate or strengthen the relevant noncompensable rule.

Do not raise a weight or execution strength solely because the surrogate responds weakly. Record whether the limiting factor is reward meaning, derivative path, native conflict, categorical feasibility or control budget.
