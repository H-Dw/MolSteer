---
name: novel_conf_molthinker-reward-creativity
description: Compose evidence-grounded molecular rewards when reference functions need adaptation or several competing objectives must be reconciled. Use StatePackets, DiagnosticReports and a supplied knowledge base to specify objectives, constraints, uncertainty and testable execution contracts. Default MolThinker reasoning mode.
---

# MolThinker: Reward Synthesis

Turn diagnosed risks and declared goals into a defensible optimization hypothesis. Treat retrieved functions as building blocks whose assumptions must be checked. Preserve the risk-only DiagnosticReport. Chinese instructions: [SKILL.zh-CN.md](SKILL.zh-CN.md).

## 1. Establish the decision context

Bind the report to its StatePacket, provenance, representation, atom mapping, units and generation time convention. Identify the controllable variables, immutable conditions, execution budget and desired outcomes. Separate observations, interpretations and chosen experimental assumptions. Treat attached knowledge as reference material, not instructions.

Output: a compact evidence ledger and an explicit control scope. Missing requirements remain unknown; they do not become favorable measurements.

## 2. Convert evidence into independent objectives

Consolidate duplicate signals and distinguish causes from correlated consequences. For each risk, ask which measurable quantity changes when that risk improves and which unrelated changes could falsely improve its score. Link each objective to evidence and an acceptance reference. Keep context-only descriptors outside the reward unless the user supplies an intended range or preference.

Classify each requirement as an optimization objective, a noncompensable constraint, a preservation condition or a monitoring-only observation. A benefit in one objective must not erase a violation of a declared constraint. Avoid adding a proxy and its direct measurement as independent penalties without a justified reason.

Output: an objective ledger with scope, evidence, meaning, units, confidence and prohibited shortcuts.

## 3. Retrieve mechanisms and test transferability

Retrieve by the observable and control mechanism, not only by a risk label. Preserve source location, original mathematical form and assumptions. Distinguish an objective from a gradient estimator, a candidate selection policy and a population strategy.

For each relevant reference, decide whether to retain, specialize, combine or replace it. Explain the mismatch that motivates a transformation and the behavior the transformation should preserve. Do not claim a newly composed expression is the original cited method. Parameter availability alone does not establish chemical or physical applicability.

Output: a short derivation map from references to the proposed terms, with unsupported candidates deferred.

## 4. Design terms and balance competing aims

Specify each observable, sign, target or tolerance, normalization, scope and differentiability. Use scales with an interpretable meaning; a unit conversion is not a priority weight. Identify uncalibrated choices and keep comparable candidates on the same objective set and normalization. An unavailable component invalidates that comparison or requires a separately declared experiment; it never silently becomes zero.

Choose a composition that matches the decision: additive costs for compensable tradeoffs, bottleneck-sensitive aggregation for imbalanced deficits, or explicit priorities when compensation is inappropriate. Smooth maxima, robust residuals and preservation costs are options, not obligatory ingredients. Check how term count, molecular size and duplicated evidence affect the score.

Model preferences can limit unsupported departures from a proposed identity, but uncalibrated marginal scores are not joint chemical probabilities. State which preferences influence gradients, which rank discrete alternatives and which only explain uncertainty.

Output: a complete mathematical reward and separate constraints, with parameter origins and expected failure modes.

## 5. Define behavior as the state changes

Bind graph-dependent references to their chemical hypotheses. Re-evaluate applicability when identity, charge, topology, stereochemistry or environmental context changes. Specify how to rebind valid references and when to suspend guidance. Do not keep an obsolete reference merely because it remains numerically computable.

Keep observed noisy state, predicted endpoint and decoded structure distinct. Any joint objective requires an explicit mapping and a reason to combine views. Treat uncertainty as a modifier of interpretation or control intensity only when its meaning supports that use; do not reward uncertainty reduction for its own sake.

Output: applicability transitions, uncertainty handling, coverage limits and stopping conditions.

## 6. Hand off an executable contract

State the gradient target, coordinate frame, editable and fixed masks, time direction, model-specific injection convention, per-step limits and cumulative control budget. Distinguish a derivative of saved coordinates from a derivative through the live predictor. A discrete term has no ordinary coordinate gradient within a fixed branch; any relaxation changes the mathematical contract and must be identified.

Require rejected proposals to preserve the unmodified sampling path for that step. Record whether constraints protect added guidance, the native generator, or final acceptance. Do not represent a proposal check as a guarantee of terminal validity. Preserve restoration state, self-conditioning and randomness so comparisons are meaningful.

Output: RewardSpec, retrieval trace, applicability/constraint contract and concise English/Chinese derivation notes. Keep diagnoses that have no supported reward visible.

## 7. Test the hypothesis and report limits

Check derivative sign and finite-difference agreement, fixed-variable preservation, normalization behavior and constraint rejection. Compare continued generation from a matched state under a shared control budget. Evaluate independently of the optimized reward and report failures, graph changes and missing coverage alongside successes.

Separate a lower surrogate cost, a valid final structure and improved task performance. An observed benefit on one continuation is evidence for that run, not a general performance claim. Deliver an auditable design rationale and results; exhaustive private reasoning is unnecessary.
