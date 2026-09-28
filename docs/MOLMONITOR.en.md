# MolMonitor: adaptive strength and reward revision routing

MolMonitor supplies fast per-step control decisions to MolExecutor and persistent-event summaries to MolThinker. Revision events produce a RewardRevisionRequest, retrieved reasoning context and a complete resumable runtime checkpoint. Creating a request does not mean a reasoning agent has already revised the reward.

## Evidence and transfer boundaries

| Primary source | Relevant idea | Transfer decision |
|---|---|---|
| [ASCED, CVPR 2025](https://arxiv.org/abs/2503.16218) | Local temporal anomalies and robust thresholds | Observe molecular geometry; a rapid change alone is not broken chemistry. |
| [APG, ICLR 2025](https://arxiv.org/abs/2410.02416) | Direction decomposition and norm control | Bound effective injection; do not assume image CFG projection has the right molecular direction. |
| [Understanding and Improving Training-free Loss-based Diffusion Guidance, NeurIPS 2024](https://arxiv.org/abs/2403.12404) | Misaligned gradients and adaptive steps | Its noise-norm/gradient-square scaling is parameterization-specific; FLOWR uses measured proposals and displacement budgets instead. |
| [Variational Control, ICML 2025](https://proceedings.mlr.press/v267/pandey25a.html) | Trajectory control under a terminal cost | Short lookahead is a promising extension; DTM is not implemented here. |
| [Dynamic CFG via Online Feedback, ICLR 2026](https://openreview.net/forum?id=z9YC9bvfUL) | Online evaluation of candidate scales | Use bounded probe search; no claim of possessing that work's trained latent evaluators. |
| [Training-free Multi-objective 3D Molecule Generation, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c8ff6807d1f362bb22b4f0be7b66e9ca-Abstract-Conference.html) | Conflicting property objectives | Persistent conflicts should revise objectives, not merely increase injection. Its quantum-property results do not validate binding affinity. |

ASCED Eq. 4 discusses weighted score differences. In audited commit `7501e05ec6f45f075ccfabaff6b687e04af18d2d`, [detection.py](https://github.com/YuCao16/ASCED/blob/7501e05ec6f45f075ccfabaff6b687e04af18d2d/src/detection.py) smooths adjacent input differences and applies median + 3×1.4826×MAD before spatial filtering. [diffusion_util.py](https://github.com/YuCao16/ASCED/blob/7501e05ec6f45f075ccfabaff6b687e04af18d2d/src/diffusion_util.py) supplies predicted pred_xstart sequences. This molecular adaptation neither reproduces ASCED nor copies image-size thresholds or its noise correction.

## Observables and reference

Keep two controls distinct: the complete unguided trajectory from the original checkpoint, and the current guided branch's same-step native-next proposal without additional injection. The latter already includes earlier guidance effects.

Observe endpoint pair distances and bond/angle residuals normalized against the current graph's MMFF references. Each graph-dependent feature binds atom identity, charge, local orders and reference values; compare it across time only when its meaning is unchanged. Valid graph changes are permitted. A temporarily invalid noisy X_t is not automatically a terminal defect.

For normalized feature f, use `v=(f_next-f_previous)/Δt`. The native-envelope threshold is

`L=max(rate_floor, 3*abs(native_rate), median(abs(native_rates))+4*1.4826*MAD(abs(native_rates)))`.

The robust window uses up to four native intervals ending at the comparison time. Default rate_floor=2 normalized units per unit progress. These are uncalibrated control assumptions, not population-derived false-alarm guarantees. A rate alarm blocks guidance only when corroborated by worsening same-graph geometry; severe chemical/geometry/clash failures are checked independently.

The three checkpoints 0.50, 0.75 and 1.00 provide only two interval-average slopes and can hide spikes or cancellations. `build_monitor_reference.py` replays the authentic native suffix at 0.01 resolution, retaining 51 endpoint frames plus the coarse summary and proving unchanged final state, conditioning and RNG. Guided runs save full runtime state every 0.05 and immediately on revision. Sparse-only references do not permit automatic strength escalation. Progress 1 is distinguished from the final head's model time 1−1e−4.

## Largest tested effective control

The controller maximizes the actual displacement norm among tested feasible proposals:

`delta(eta)=BudgetClip(eta*adapter.inject(gradient,dt))`;
`selected = argmax ||delta(eta)|| over tested eta satisfying the checks`.

The finite set covers growth/shrinkage around the previous coefficient, a lower bound and a budget-saturation bound, with at most nine probes. Equivalent clipped proposals are deduplicated, and equal effective proposals prefer a smaller nominal coefficient. Defaults are initial eta 30, range 0.01–10000 and growth factor 4. Per-step and cumulative budgets are explicit experiment limits.

All probes share the same native-next state, time and conditioning and consume no sampler randomness. Graph transitions invalidate assumptions of monotone feasibility, so this is not a global optimum or a binary-search proof. Rejected candidates do not mutate the sampler. A same-time reward gain cannot guarantee final improvement.

Every five steps from progress 0.75, an MMFF94s relaxation sentinel evaluates a molecule copy. It is not differentiated and never edits the submitted pose. Same-graph results can be compared with the native reference. A changed graph uses strain relative to its own local minimum, normalized per heavy atom; values above the declared default of 1 kcal/mol/heavy atom, or an unavailable check, request quality review. This provisional screen is not a universal validity threshold or a cross-graph force-field total-energy ranking. Nonconvergence is not a pass.

## Routing and revision packet

Feasible control goes to MolExecutor. No feasible tested control causes a native-only step and a lower search center. Early unavailable chemistry also uses native-only progress. Exhausted displacement budget stops added guidance without misclassifying the budget as a faulty reward.

Persistent proposal failure, late unavailable evaluation, objective/quality conflict or adverse changed-graph quality review goes to MolThinker. Defaults: three-step persistence, five-step event cooldown, at most two revisions. Checkpoints are saved after the completed native transition and any accepted proposal, preserving an internally consistent state. Strain alarms persist until their next five-step refresh; three-step persistence does not imply three independent force-field evaluations.

A revision packet contains:

- Evidence/program identity, target, model/receptor/checkpoint bindings, units and time convention.
- Local atom identity, charge and coordinates; observed geometry, reference/tolerance, native and observed rates, and threshold provenance.
- The branch's native-next and tested counterfactual candidates, distinct from the original native trajectory.
- Tried nominal/effective strengths, actual displacements, gains and individual rejection causes.
- Separate affinity heads and independent quality/coverage, including comparable strain.
- Recent event history, persistence, graph changes and remaining control budget.
- Observed facts separated from hypotheses about excessive steps, missing terms, conflicting directions or unsuitable scales.
- Constraints and measurement definitions that revisions must preserve, plus a validation contract.

The reasoning context retains positive-relevance knowledge hits with formulas and provenance, plus a summary and digest of the complete reward, avoiding duplicated historical retrieval tables. It contains no fabricated language-model call. A response binds request_id and parent_program_id and supplies a declarative program, rationale and validation plan. The executor's `--revision-response` and `--resume` options preserve path use/conditioning/RNG; a revised reward requires a live-gradient preflight. An explicit `stop_guidance` response keeps the current program and budget, disables further injection and observes the native tail without a gradient preflight. Stopping is not a claim of rescue. Guard thresholds and reference normalization cannot silently change.

## Architecture and use

`features.py` extracts observations; `reference.py` handles native envelopes; `controller.py` owns search and persistent routing state; `runtime.py` performs transactional execution; `feedback.py` forms requests; `molthinker/feedback.py` prepares reasoning context and validates responses. The generic Executor selects this path when its configuration contains `monitor`. The previous execution path remains available.

The controller is independent of network architecture. The runtime requires adapter methods for prediction, endpoint conversion, native stepping, injection units and full checkpointing. Integration has been validated only with FLOWR.ROOT. Existing PyTorch, RDKit and NumPy suffice; no dependency was added.

Final molecules still require independent Reader assessment. One matched trajectory, heuristic thresholds and local proposal checks cannot guarantee globally broken-free generation or calibrated affinity improvement.

## Invocation

Add `monitor.reference`, `monitor.knowledge_path` and optional `monitor.policy` to the existing Executor JSON configuration. The reference must bind the original complete runtime. Keep the per-step and cumulative displacement limits in the existing `budget` field. A minimal policy is `{"initial_eta":30.0,"max_eta":10000.0}`.

```bash
cd /data1/dhuang/flowr_root
PYTHONPATH=MolSteer/src .venv/bin/python -m molsteer.molexecutor.runner \
  --config /path/execution.json --output /path/fresh-output
PYTHONPATH=MolSteer/src .venv/bin/python -m molsteer.molexecutor.runner \
  --config /path/execution.json --resume /path/resume_revision.pt \
  --revision-response /path/response.json --output /path/fresh-revision-output
```
