"""Diagnostics separate native dynamics, injection, clipping and local retention."""
import torch


def vector_cosine(a, b):
    a,b=a.detach().reshape(-1),b.detach().reshape(-1)
    denominator=float(a.norm()*b.norm())
    return float(torch.dot(a,b))/denominator if denominator>1e-16 else None


def step_comparison(native_delta, gradient, raw_delta, bounded_delta, scale=1.):
    native_delta,gradient,raw_delta,bounded_delta=[x.detach() for x in [native_delta,gradient,raw_delta,bounded_delta]]
    n=float(native_delta.norm())
    return dict(native_step_l2_angstrom=n*scale,
        native_step_max_angstrom=float(native_delta.norm(dim=-1).max())*scale,
        cosine_native_gradient=vector_cosine(native_delta,gradient),
        gradient_dot_native_step=float((native_delta*gradient).sum()),
        raw_guidance_l2_angstrom=float(raw_delta.norm())*scale,
        bounded_guidance_l2_angstrom=float(bounded_delta.norm())*scale,
        raw_guidance_to_native_ratio=float(raw_delta.norm())/n if n>1e-16 else None,
        bounded_guidance_to_native_ratio=float(bounded_delta.norm())/n if n>1e-16 else None,
        clipped_atom_count=int((raw_delta.norm(dim=-1)>bounded_delta.norm(dim=-1)+1e-9).sum()))


def linear_flow_retention(injection, endpoint_response, next_dt, time):
    """One counterfactual native coordinate step with common randomness/SC.

    For linear continuous FLOWR only: delta_next = (1-a)*delta + a*delta_head.
    Categorical updates and subsequent self-conditioning divergence are outside
    this local estimate. It is not final trajectory attribution.
    """
    coefficient=next_dt/(1-time)
    propagated=(1-coefficient)*injection+coefficient*endpoint_response
    norm=float(injection.norm())
    if norm<=1e-16:return {}
    return dict(local_retention_norm_ratio=float(propagated.norm())/norm,
        local_retention_signed_ratio=float((propagated*injection).sum())/(norm*norm),
        endpoint_response_to_injection_ratio=float(endpoint_response.norm())/norm,
        flow_endpoint_coefficient=coefficient)
