"""Small model-neutral contracts. A model adapter owns its sampling convention."""
from dataclasses import dataclass
from typing import Protocol, Callable
import torch


@dataclass(frozen=True)
class GuidanceBudget:
    strength: float = 1.0
    max_step_angstrom: float = 0.02
    max_path_angstrom: float = 0.5

    def __post_init__(self):
        import math
        if any(not math.isfinite(v) for v in vars(self).values()) or self.strength<0 or self.max_step_angstrom<=0 or self.max_path_angstrom<=0:
            raise ValueError('Strength must be finite and nonnegative; displacement limits must be finite and positive')


class SamplingAdapter(Protocol):
    """Endpoint gradients are pulled back through predict; inject owns sign/scale."""
    def predict(self, coordinates: torch.Tensor): ...
    def inject(self, gradient: torch.Tensor, step_size: float) -> torch.Tensor: ...


@dataclass
class CallbackAdapter:
    """Bridge for external samplers without subclassing their model.

    pullback_multiplier must be explicitly supplied in the sampler's units.
    For a forward flow velocity use positive dt. For a DDPM reverse mean,
    use the appropriate covariance, never infer a sign from a time label.
    """
    endpoint: Callable
    pullback_multiplier: Callable

    def predict(self, coordinates):
        return self.endpoint(coordinates)

    def inject(self, gradient, step_size):
        factor = self.pullback_multiplier(step_size)
        return factor * gradient


def bounded_displacement(delta, editable_mask, path_used, budget, coord_scale=1.0):
    """Clip per-atom displacement and total injected path, in world Angstrom."""
    if coord_scale <= 0 or not torch.isfinite(delta).all():
        raise ValueError('Nonfinite guidance or invalid coordinate scale')
    if editable_mask.shape != delta.shape[:-1] or path_used.shape != editable_mask.shape:
        raise ValueError('Explicit editable mask/path shape mismatch')
    delta = delta * editable_mask.unsqueeze(-1)
    norms = torch.linalg.vector_norm(delta, dim=-1) * coord_scale
    cap = torch.minimum(torch.full_like(norms, budget.max_step_angstrom),
                        (budget.max_path_angstrom-path_used).clamp(min=0))
    delta = delta * (cap / norms.clamp(min=1e-12)).clamp(max=1).unsqueeze(-1)
    return delta


def endpoint_gradient(adapter, coordinates, reward):
    """No BPTT: each generation step supplies fresh detached coordinates."""
    x = coordinates.detach().requires_grad_(True)
    endpoint = adapter.predict(x)
    value = reward(endpoint)
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError('Reward must be a finite scalar')
    gradient, = torch.autograd.grad(value, x, allow_unused=False)
    if not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite reward gradient')
    return value.detach(), gradient.detach(), endpoint


def guided_step(adapter, coordinates, reward, native_step, step_size, editable_mask,
                path_used, budget, validator, coord_scale=1.0):
    """Inject guidance into an arbitrary flow/diffusion sampler's native step.

    native_step owns randomness and returns the unmodified next coordinates.
    validator(candidate, native) is mandatory and supplies model/task constraints.
    Rejection returns native exactly. Fixed atoms are protected against *guidance*;
    preserving them in the native process is the caller's inpainting responsibility.
    """
    value,gradient,_=endpoint_gradient(adapter,coordinates,reward)
    with torch.no_grad():
        native=native_step(coordinates.detach())
        delta=bounded_displacement(budget.strength*adapter.inject(gradient,step_size),
            editable_mask,path_used,budget,coord_scale)
        candidate=native+delta
        accepted=bool(validator(candidate,native))
        if not accepted:delta=torch.zeros_like(delta)
        path=path_used+delta.norm(dim=-1)*coord_scale
    return native+delta,path,dict(reward=float(value),accepted=accepted,gradient_norm=float(gradient.norm()))
