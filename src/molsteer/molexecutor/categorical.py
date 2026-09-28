"""Deterministic distribution proposals followed by the model's native transition."""
import math
import torch


def temper_prediction(predicted, conditioning, index, temperature, confidence_ceiling=.95):
    """Temper uncertain slots only; consume no RNG and keep SC consistent.

    This changes the categorical proposal kernel, not a calibrated posterior.
    The native integrator still samples the next state. Scores must come from a
    fresh model call on that state, never from the modified proposal's old heads.
    """
    if not math.isfinite(temperature) or temperature<=0 or not 0<confidence_ceiling<=1:
        raise ValueError('Invalid categorical proposal policy')
    if temperature==1:
        return predicted,conditioning
    out=dict(predicted);cond=dict(conditioning)
    for key in ('atomics','charges','bonds'):
        probabilities=predicted[key][index]
        if not torch.isfinite(probabilities).all() or (probabilities<0).any():
            raise ValueError('Invalid categorical probabilities')
        proposal=probabilities.pow(1/temperature)
        proposal=proposal/proposal.sum(-1,keepdim=True).clamp(min=1e-30)
        editable=probabilities.max(-1).values<confidence_ceiling
        if key=='bonds':
            editable=editable & editable.T
            editable=editable & ~torch.eye(len(editable),device=editable.device,dtype=torch.bool)
        proposal=torch.where(editable.unsqueeze(-1),proposal,probabilities)
        out[key]=predicted[key].clone();out[key][index]=proposal
        if key in cond:
            cond[key]=conditioning[key].clone();cond[key][index]=proposal
    return out,cond
