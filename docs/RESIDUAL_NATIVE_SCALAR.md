# Residual needs and full native scalar injection

Implemented 2026-10-03 using offline tests, synthetic differentiable models and saved experiment records. No new paid API calls or molecular generation. [中文说明](RESIDUAL_NATIVE_SCALAR.zh-CN.md).

## Evidence and expert decisions

The current DiagnosticReport keeps its original state semantics. A separate paginated `residual_needs` index starts from explicit raw-final remainders and traces current precursors across every configured node. Cards retain current chemistry, relation applicability, measurements, references, deviations, recurrence, natural clearance, missing observations, preservation references and competing explanations. Chemistry transitions do not erase current-type defects. Absent relations, missing measurements and resolution are distinct; a missing final stays unknown.

`inspect_raw_comparison` pages this index; `read_raw_reference` supplies exact sources. Host prepares diagnosis and temporal indices before the Reader loop. No fixed three-tool read sequence is required. `reader.measurement_stage_paths` optionally maps node IDs to saved input stages. Existing calculators run on bound copies through `measure_evidence_gap`; separately saved MeasurementSupplements preserve source/coordinate/configuration hashes and can be read by field. Missing inputs are explicit. Original packet, report and data remain unchanged; future supplements cannot serve as current numerical reward evidence.

Biology compares terminal remainder, current intervention mechanism and newly covered independent need before selecting a minimal sufficient set. Value records include counterevidence, preservation, uncertainty and pairwise rank reasons. Native-resolved defects require additional-value arguments; necessary unsupported goals stay deferred. No objective count is prescribed.

Mathematics researches independent needs, distinguishes retained/specialized/reconstructed source parts, determines target sets, derives local response and constructs one scalar reward. Rank coefficients affect the actual composition. Preservation is a scalar term or independent evaluation. Copy probes record component values, scales, weights, gradient norms, cosines and directional contributions; an optional live adapter probe measures pullbacks while restoring RNG/state. Copy, live and terminal evidence remain separate.

## Workspace and usage

Host owns IDs, versions, full/computational hashes, dependencies, historical artifacts and validation cache. Construction saves once and returns receipts; assembled directions reference one authoritative design. Read/patch/test/submit by candidate reference. Contracts support overview, sections and explicit full reads; repeat reads return receipts. Pure narrative changes reuse numerical results while final validation still binds the exact submitted artifact.

Default tool groups depend on actual candidate/evidence/validation state. Research can reopen without discarding the candidate, and the complete registry retains legitimate older calls. After `runtime.no_progress_actions` (default 3) unchanged tool actions, one advisory appears in the next ordinary request. Repeated reads/saves, wording changes, group switches and candidate oscillations do not count as progress. No extra model call or submission gate is added.

`api_usage` stores module, phase, provider-reported usage/cost when present, and serialized message/tool-definition bytes. Missing usage is unknown. Byte counts are neither token estimates nor exact HTTP payload sizes. Provider tool-call pairs and private reasoning signatures remain intact in ephemeral conversation history.

## Native execution and migration

Every remaining step performs the same native forward, computes the scalar pullback into current coordinates, executes the native update and adds `guidance_weight * adapter.inject(gradient, dt)`. FLOWR's injection is `dt * gradient`. Reward and external weight stay fixed; declared current-chemistry rules interpret parameters. All active target coordinates participate unless explicitly fixed; padding and other batch members receive no injected delta.

The direct path has no displacement caps, path budgets, windows, extra step limits, projections, proposal rejection, backtracking, graph-change suspension or automatic reweighting. Independent Monitor/graph review defaults off and is not called. Executor uses deterministic loading and computation, with no separate model approval round. Zero gradients remain valid; undefined required components, nonfinite calculations or tensor mismatches fail with diagnostics and a recoverable checkpoint. Path length is telemetry only.

Use `runtime.guidance_weight` in Agent config and `guidance_weight` in execution JSON. Old monitor strength, budget strength and reward-weight wrappers do not supply a second multiplier. Legacy controller fields are readable but logged as inactive. Historical scalar handoffs preserve scientific binding; common-descent and proposal-constraint designs explicitly require scalar redesign. The 5i0b sweep script now holds the reward fixed and varies only the external weight.

## Verification scope

The full offline suite passed 438 tests in 295 seconds. An additional historical scalar-contract migration test then passed together with the existing version-binding test (2 passed). Migration ignores obsolete execution metadata only, preserving scientific and computational bindings. The full suite emitted one PyTorch warning from a historical discrete-gradient utility outside the direct injection path. `git diff --check` passed.

Synthetic tests cover direct numerical step equivalence including high weights, exact zero-weight native behavior, zero gradients, fixed variables, full coordinate pullback, batch isolation, categorical randomness, self-conditioning, restart and true failure recovery. Scripted model tests verify actual prompt/tool routing, reference submission, numerical caching, usage accounting and absence of Executor/Monitor API calls. These are not measurements of GLM reasoning quality or new molecular outcomes.

Re-indexing the saved `5i0b_glm_10_100_20261002_2030/attempt_08` comparison found 171 tracks across 6 nodes: 64 naturally cleared, 105 terminal relations absent, and 2 terminal defects with observation gaps (both structural alerts). 78 tracks contain chemistry transitions. Source SHA256: `3f13449577063633f5c72eef4d61eca2f3bcdacf18320117223f2c832ee979ea`. Relation absence is not successful repair, and structural alerts are not automatically coordinate-controllable. No new affinity, synthesis or terminal-quality improvement is claimed.
