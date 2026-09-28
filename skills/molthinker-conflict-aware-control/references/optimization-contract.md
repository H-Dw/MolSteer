# Optimization contract

This reference fixes the minimum mathematical contract for conflict-aware guidance. It is descriptive until a runtime adapter validates every convention.

## Common coordinates

Let `z` be the actual editable control vector for the current branch and state. Let `M` encode the editable mask and equality-constraint tangent basis, and let `G` be a declared positive-definite metric in that reduced coordinate system. Every objective derivative MUST be represented as `g_i = d f_i / d z` after the live mapping/Jacobian, then restricted to the same feasible tangent coordinates. A derivative in endpoint coordinates, a stale branch, or a different normalization is not comparable.

For an equality-tangent space, define the metric projection explicitly. If `B` has columns spanning the feasible tangent in ambient coordinates, one valid reduced representation is `q = B^T G B` and `a_i = B^T g_i`; inner products are `a_i^T q^{-1} a_j`, norms use the same metric, and the physical direction is `B q^{-1} a_i`. An implementation MAY use another equivalent representation but MUST record it and verify positive definiteness and finite-difference agreement.

Inequality constraints produce a feasible tangent cone. The local common-descent problem MUST account for active inequalities and may not replace the cone with an unconstrained subspace without recording that approximation.

## Normalized objective and common-descent QP

For normalized objectives `f_i(z)` and a proposed direction `d`, one reference problem is:

```text
minimize      1/2 ||d||_G^2
subject to    a_i^T d <= -m_i       for each active soft objective i
              d in K(z)             (feasible tangent cone)
              ||d||_G <= d_max
```

Here `m_i >= 0` is an explicitly declared first-order margin, not an arbitrary weight. When a feasible solution exists, a dual/KKT solution can provide nonnegative objective multipliers `alpha_i` normalized to `sum(alpha_i)=1` for reporting the compromise local potential `Phi = sum(alpha_i f_i)`. Report solver status, primal residuals, dual residuals, complementarity, active set, metric and tolerances. If weights are obtained from a different primal/dual problem, state that exact problem instead.

A negative pairwise cosine in the shared metric is a conflict signal:

```text
cos(i,j) = <a_i, a_j>_(G^{-1}) / (||a_i||_(G^{-1}) ||a_j||_(G^{-1}))
```

It is not a proof that no joint descent exists. Conversely, nonnegative pairwise cosines do not prove feasibility under constraints. If no feasible solution satisfies the declared margins, report `no_feasible_common_descent` or the more specific status (`stationary`, `constraint_infeasible`, `derivative_unavailable`, `numerical_failure`). Do not choose a convex combination merely because it produces a finite vector.

Hard constraints are predicates or explicit feasible sets and MUST not be represented only by a soft objective. A linearized proposal requires a post-update nonlinear predicate check.

## Derivative and scale checks

For each editable coordinate or controlled direction, compare the declared derivative with a symmetric finite difference at a documented epsilon, respecting branch identity and constraints. Record failures near kinks or stochastic predictors rather than hiding them. Scales `s_i > 0` are part of the objective definition and remain fixed for a comparison. A runtime that changes scales or metric during a step MUST record the controller event and recompute diagnostics.

## Controller boundary

The runtime adapter MUST define whether a direction is added, subtracted, projected, clipped or used to alter a model input. The optimization sign does not determine the injection sign. Record time direction, model input, mask, state snapshot, RNG stream and restoration operation.
