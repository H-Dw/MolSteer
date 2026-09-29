"""Dimensionless minimum-norm multiobjective weights in one editable space."""
import numpy as np


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
