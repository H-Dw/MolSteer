# MolThinker dual experts

See [the current residual-first scalar contract](RESIDUAL_NATIVE_SCALAR.md). Budget, proposal-rejection, segmented-control and common-descent sections below document historical adapters, not the current MolThinker → FLOWR.ROOT path.

The shipped configuration enables `dual_expert`; old configurations without a
`thinker` section retain `single`. Explicit offline mode still runs the historical
deterministic algorithms. The public Reader → Thinker → Executor → Monitor graph
is unchanged. See [the full Chinese interface guide](DUAL_EXPERTS.zh-CN.md).

Biology produces a complete `BiologyPlan` with mechanism-based grouping, ranked
directions, repair predicates, evidence, falsifiers and uncertainty. Every diagnostic
finding has an optimize/constraint/monitor/deferred disposition. Preservation
conditions reference explicit constraint directions. Mathematical feasibility does
not determine biological importance.

Mathematics retrieves local knowledge for every selected direction and records
original formulas, exact passage locators, assumptions, specialization, sensitivities
and alternatives in `MathematicalDesign`. Executable submissions must match a design
that passed the actual `test_mathematical_design` tool. Unsupported designs remain
`design_only`. Biology reconsideration is bounded to two rounds by default.

Each expert can call the independent Researcher. Search/fetch providers support
Markdown, Europe PMC, arXiv and host-injected Web clients. Research begins locally,
distinguishes search snippets from inspected passages, caches identical requests,
and returns evidence without activating objectives. Default budgets are four requests
per planning pass, six tool rounds and three searches per request. External errors,
zero hits, abstract-only coverage and unanswered questions remain explicit.

Configure `thinker.experts.biology`, `.mathematics`, and `.researcher` as model-profile
names; omitted roles inherit MolThinker's existing profile. Injected models use
`molthinker.<role>` keys. `AgentRuntime` accepts `model_dynamics`, `fetch_fn` and
`research_providers`; an inference adapter may supply `describe_dynamics()`.
Unknown generator assumptions are never inferred from an endpoint snapshot.

RewardSpec 2.0.0 separates local observables, bounded unit-checked JSON expressions,
constraints and controllers. It supports distances, angles, anchors, directions,
dihedrals, signed volumes and applicability-checked first-order MMFF strain. Local
formulas must be nonnegative dimensionless deficits. Constraint violations must be
zero. Cross-direction fixed weighted sums are rejected; justified scalar potentials
and a masked Euclidean common-descent controller are supported. General constrained
QP solvers, arbitrary code and new categorical editing operators remain unsupported.

Live component gradients are pulled back through the actual generator into the same
editable variable before comparison. Solver convergence, directional derivatives,
post-clipping/injection directions and same-time nonlinear proposal constraints are
checked. Graph changes invalidate the binding. Probe calls restore native coordinates,
conditioning and random state. The FLOWR bridge and monitored runtime preserve the
controller, rather than replacing it with a scalar reporting score. Live preflight is
mandatory for version-2 programs; coordinate-copy validation is not that preflight.

Run `python -X utf8 -m pytest tests -q` for regression tests and
`python -X utf8 scripts/run_dual_expert_smoke.py` for a real-LLM smoke run. Missing
credentials produce an explicit skipped artifact. The smoke records whether both
experts actually used Researcher. GPU continuation and terminal molecular efficacy
are separate, unrun evaluations unless explicitly executed. Audit snapshots retain
artifacts and concise mathematical summaries, not provider-private reasoning or keys.
