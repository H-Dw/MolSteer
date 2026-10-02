"""Dimensionless minimum-norm multiobjective weights in one editable space."""
import numpy as np


def priority_descent(gradients, editable_mask, weights, *, max_iterations=2000, tolerance=1e-10):
    """Project a rank-weighted preferred descent onto the common non-ascent cone.

    Minimize ||d + sum(w_i g_i)/sum(w_i)||² subject to g_i.d <= 0.
    Hildreth dual coordinate updates need only the small gradient Gram matrix.
    Scaling gradients before a minimum-norm simplex solve would instead often
    give a *lower*-weighted orthogonal goal more movement, reversing preference.
    """
    raw = np.asarray(gradients, dtype=float)
    mask = np.asarray(editable_mask)
    weights = np.asarray(weights, dtype=float)
    if (raw.ndim < 2 or mask.shape != raw.shape[1:] or not np.isin(mask, [0, 1]).all()
            or not np.isfinite(raw).all() or weights.shape != (len(raw),)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('Priority descent requires finite gradients, a matching mask and positive weights')
    g = (raw * mask).reshape(len(raw), -1)
    preferred = -(weights @ g) / weights.sum()
    norms = np.linalg.norm(g, axis=1)
    normals = np.divide(g, norms[:, None], out=np.zeros_like(g), where=norms[:, None] > 0)
    gram = normals @ normals.T
    residual = normals @ preferred
    multipliers = np.zeros(len(g))
    limit = tolerance * max(1., np.linalg.norm(preferred))
    converged = False
    for iteration in range(max_iterations):
        movement = 0.
        for i in range(len(g)):
            change = max(0., multipliers[i] + residual[i]) - multipliers[i]
            multipliers[i] += change
            residual -= change * gram[:, i]
            movement = max(movement, abs(change))
        if movement <= limit and residual.max(initial=0.) <= limit:
            converged = True
            break
    direction = preferred - multipliers @ normals
    derivatives = g @ direction
    status = ('numerical_failure' if not converged else
              'pareto_stationary_or_inactive' if np.linalg.norm(direction) <= limit else 'candidate_descent')
    return dict(projected_direction=direction.reshape(raw.shape[1:]).tolist(),
        preferred_direction=preferred.reshape(raw.shape[1:]).tolist(), weights=weights.tolist(),
        directional_derivatives=derivatives.tolist(), status=status, converged=converged,
        iterations=iteration+1, allocation='Rank-weighted descent projected onto current common non-ascent directions',
        constraint_scope='First-order objective preservation and editable mask; finite proposals still require evaluation')


def conflict_weights(gradients, editable_mask, *, scales=None, max_iterations=2000, tolerance=1e-10):
    """Solve min_{w>=0, sum(w)=1} ||sum_i w_i projected(g_i / scale_i)||².

    Gradients must already be pulled back to the same runtime variable, frame,
    and metric. The mask protects fixed coordinates; it is not a substitute for
    a general nonlinear constraint Jacobian. This function never changes a
    generator's injection sign, strength or hard feasibility constraints.
    """
    raw = np.asarray(gradients, dtype=float)
    mask = np.asarray(editable_mask)
    if raw.ndim < 2 or not raw.shape[0] or not np.isfinite(raw).all():
        raise ValueError('Expected finite gradients with objective and editable-space axes')
    if mask.shape != raw.shape[1:] or not np.isin(mask, [0, 1]).all():
        raise ValueError('Mask must match the full editable space and contain only 0/1')
    n = raw.shape[0]
    scales = np.ones(n) if scales is None else np.asarray(scales, dtype=float)
    if scales.shape != (n,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Each objective requires a finite positive normalization scale')
    if max_iterations < 1 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError('Invalid optimization budget')
    g = (raw * mask).reshape(n, -1) / scales[:, None]
    gram = g @ g.T
    w = np.full(n, 1. / n)
    gap = 0.
    for iteration in range(max_iterations):
        derivative = gram @ w
        vertex = np.zeros(n)
        vertex[np.argmin(derivative)] = 1.
        direction = vertex - w
        gap = float(-direction @ derivative)
        if gap <= tolerance:
            break
        curvature = float(direction @ gram @ direction)
        step = min(1., gap / curvature) if curvature > 0 else 1.
        w += step * direction
    combined = w @ g
    derivative = gram @ w
    gap = max(0., float(w @ derivative - derivative.min()))
    norm = float(np.linalg.norm(combined))
    directional = -g @ combined
    denom = np.sqrt(np.diag(gram))
    products = np.outer(denom, denom)
    cosine = np.divide(gram, products, out=np.zeros_like(gram), where=products > 0)
    # A zero gradient is inactive; zero cosine does not establish independence.
    cosine_values = [[float(cosine[i, j]) if products[i, j] > 0 else None for j in range(n)] for i in range(n)]
    converged = gap <= tolerance
    status = ('numerical_failure' if not converged else
              'pareto_stationary_or_inactive' if norm <= np.sqrt(tolerance) else
              'no_common_descent' if np.any(directional > tolerance) else 'candidate_descent')
    return dict(weights=w.tolist(), projected_direction=(-combined).reshape(raw.shape[1:]).tolist(),
                cosine=cosine_values, inactive_objectives=np.flatnonzero(denom <= tolerance).tolist(),
                status=status,
                directional_derivatives=directional.tolist(), converged=converged,
                dual_gap=gap, iterations=iteration + 1,
                constraint_scope='editable mask only; other hard constraints require proposal checks',
                controller_strength_included=False)
