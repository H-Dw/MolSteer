# Decision contract

`ConflictAwareControlDecision` is a compact, auditable handoff. It records decisions, evidence and observations, not private raw chain-of-thought. A consumer MUST reject the handoff if required fields are absent or status is not authorized.

## Required fields

- `schema_version`: contract version string.
- `decision_id`: stable identifier.
- `status`: `proposed`, `defer`, `reject`, or `authorized`.
- `execution_authorized`: boolean; MUST be false unless all mandatory checks pass.
- `state_binding`: StatePacket and DiagnosticReport IDs/hashes, generation step, state view, representation, atom mapping and units.
- `control_scope`: editable coordinates/mask, fixed variables, branch IDs, time direction, model injection convention, step and cumulative budgets.
- `evidence_ledger`: observations, causal/mechanistic classification, uncertainty and provenance references.
- `objective_ledger`: normalized objective definitions, directions, targets/tolerances, scales, aggregation groups and differentiability.
- `constraints`: predicates, tolerances, enforcement scope, fail-closed behavior and rollback state.
- `branch_decisions`: per-branch applicability, continuous solve status, feasibility and matched-continuation results.
- `conflict_solution`: common-coordinate/metric definition, gradients, conflict diagnostics, QP status, weights, direction and residuals, or an explicit no-descent status.
- `local_potential`: formula and validity window; selected weights are not silently differentiated through.
- `controller`: branch selection, recomputation/gating, schedule, injection and budget policy.
- `continuation_evaluation`: matched state/RNG protocol, horizon, budget, evaluator, sample count, paired outcomes, uncertainty and limitations; use `not_run` or `unavailable` when absent.
- `retrieval_trace`: actual tool observations with query, corpus/source IDs, versions, locators and excerpts; distinguish zero-hit/unavailable/error.
- `checks`: named checks with `status` (`passed`, `failed`, `not_run`, `unavailable`) and evidence.
- `limitations`: missing evidence, untested assumptions, coverage and claims not supported.
- `next_action`: precise consumer action (execute, gather evidence, restore baseline, or defer).

## Status semantics

`authorized` is permitted only when `execution_authorized` is true and no required check is failed, unavailable or not run. `proposed` is a design suggestion and MUST NOT be executed. `defer` means required evidence or capability is missing. `reject` means a hard constraint, binding, or validity condition failed. `unknown` values MUST be represented explicitly rather than converted to a favorable value.

## Evidence semantics

Each claim SHOULD cite one or more `evidence_ledger` or `retrieval_trace` entries. A tool plan is not a tool observation. A web snippet is not equivalent to a verified source passage. A single matched continuation supports only the recorded run and conditions.
