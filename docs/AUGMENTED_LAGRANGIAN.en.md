# Augmented-Lagrangian execution and dual-state restart

MolExecutor accepts `evaluator: augmented_lagrangian` for a task utility subject to registered differentiable inequality constraints. For residuals (h_j(X,G)\ge 0), the evaluator maximizes

\[
R(X,G)=U(X,G)-\sum_j\left[\lambda_j h_j(X,G)+\frac{\kappa_j}{2}h_j(X,G)^2\right].
\]

Each constraint declares its normalization, positive repair margin, initial multiplier, quadratic penalty, multiplier ceiling, satisfaction tolerance and persistence rule. A violation updates `lambda_j <- min(lambda_max, lambda_j + kappa_j h_j)`. A multiplier can decay only after the constraint remains within tolerance for the declared number of accepted frames. Task gains therefore cannot numerically cancel chemical validity, connectivity, new severe clashes, receptor immobility or injection budgets; those remain proposal guards.

Constraint evaluators use a registry. The delivered evaluators cover localized graph-conditioned MMFF geometry, local MMFF relaxation strain relative to a same-time native target, and displacement outside the diagnosed support. New flow/diffusion integrations can register another scalar residual without changing the dual controller. The editable mask is independently declared as the defect halo or all ligand atoms.

`guidance_state.augmented_lagrangian` stores the controller schema, RewardProgram identity, a hash of the complete constraint definition, every multiplier, satisfaction counters, update count, last updated integration step and bounded update history. The surrounding runtime checkpoint already stores current state, self-conditioning, RNG, path budget and model context. Restore rejects a changed program, constraint definition or constraint ID set. Repeating the same update step is idempotent.

The FLOWR suffix engine evaluates proposals at the same next time and self-conditioning state as the native branch. It updates duals once from the selected endpoint, writes a checkpoint every five integration steps and records component gradient norms, constraint residuals, multiplier transitions, clipping and independent oracle guards. Missing native scalar measurements may be linearly interpolated only between two valid bracketing frames within a declared maximum gap; the trace records the interpolation mode and bracket.

The verified 5i0b / ligand_002 test uses the exact t=0.50 runtime and eta 100. It passes the live finite-difference screen, all 73 repository tests, and an exact t=0.75 restart comparison over 2,143 checkpoint leaves. The uninterrupted and resumed final state, head prediction, world prediction and SDF are byte-identical. This verifies execution and restart fidelity for the tested FLOWR adapter; it does not calibrate the chosen constraint scales for other molecules.

No additional package is required. Use a fresh output directory and an execution JSON whose RewardProgram declares the evaluator, control trajectory, native references, constraints and guidance budget.
