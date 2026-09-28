"""First-order probability guidance before native categorical sampling.

This changes endpoint categorical proposals, not hard-token derivatives and not
the exact CTMC Doob transform. The score must explicitly accept soft categories.
"""
import math
import torch


def motif_score(probabilities, hypotheses, temperature=0.25):
    """Alternative mapped templates; mean log support is not a graph likelihood.

    A hypothesis has assignments {feature, slots, category} and optional utility.
    Bond assignments are undirected. The caller owns chemical validity, source
    evidence and mapping of the FULL candidate, including attachment chemistry.
    """
    if not hypotheses or temperature<=0:raise ValueError('Nonempty alternatives and positive temperature required')
    scores=[]
    for hypothesis in hypotheses:
        terms=[]
        for term in hypothesis['assignments']:
            key=term['feature'];slot=tuple(term['slots']);category=term['category']
            value=probabilities[key][slot+(category,)]
            if key=='bonds':
                if len(slot)!=2 or slot[0]==slot[1]:raise ValueError('Distinct undirected bond endpoints required')
                value=.5*(value+probabilities[key][(slot[1],slot[0],category)])
            terms.append(value.clamp_min(1e-8).log())
        if not terms:raise ValueError('Empty chemical hypothesis')
        scores.append(torch.stack(terms).mean()+float(hypothesis.get('utility',0.)))
    values=torch.stack(scores)/temperature
    return temperature*(torch.logsumexp(values,0)-math.log(len(scores)))


def guided_probabilities(probabilities,score,editable,*,strength=1.,max_kl=.02,max_log_change=1.):
    """Exponentiated probability-gradient step with shared KL backtracking.

    Inputs describe one molecule. Noneditable slots, symmetric bonds, diagonal,
    exact zeros and all input tensors are preserved. No random draws occur.
    """
    if not all(math.isfinite(v) for v in (strength,max_kl,max_log_change)) or strength<0 or max_kl<=0 or max_log_change<=0:
        raise ValueError('Invalid discrete trust region')
    leaves={k:v.detach().clone().requires_grad_(True) for k,v in probabilities.items()}
    for key,p in leaves.items():
        if not torch.isfinite(p).all() or (p<0).any() or not torch.allclose(p.sum(-1),torch.ones_like(p[...,0]),atol=1e-5,rtol=0):
            raise ValueError('Categorical input is not a probability simplex')
        if editable[key].shape!=p.shape[:-1]:raise ValueError('Explicit editable mask mismatch')
        if key=='bonds' and (not torch.allclose(p,p.transpose(0,1),atol=1e-6,rtol=0) or not torch.equal(editable[key],editable[key].T)):
            raise ValueError('Bonds and bond masks must be symmetric')
    with torch.enable_grad():
        value=score(leaves)
        if value.ndim or not torch.isfinite(value):raise ValueError('Finite scalar soft-category score required')
        raw=torch.autograd.grad(value,tuple(leaves.values()),allow_unused=True)
    gradients={k:(torch.zeros_like(leaves[k]) if g is None else g.detach()) for k,g in zip(leaves,raw)}
    masks={k:v.clone().bool() for k,v in editable.items()}
    if 'bonds' in masks:masks['bonds'].fill_diagonal_(False)
    for key,g in gradients.items():
        if not torch.isfinite(g).all():raise ValueError('Nonfinite categorical derivative')
        if key=='bonds':g=.5*(g+g.transpose(0,1))
        p=leaves[key].detach();g=g-(g*p).sum(-1,keepdim=True)
        gradients[key]=g*masks[key].unsqueeze(-1)
    if strength==0:return {k:v.detach().clone() for k,v in probabilities.items()},dict(accepted=False,reason='zero_strength',kl_max=0.)
    for bt in range(16):
        candidate={};kl_max=0.;delta_max=0.
        for key,p0 in probabilities.items():
            delta=(strength*.5**bt*gradients[key]).clamp(-max_log_change,max_log_change)
            log_p=torch.where(p0>0,p0.clamp_min(1e-30).log(),torch.full_like(p0,-torch.inf))
            q=torch.softmax(log_p+delta,-1)
            q=torch.where(masks[key].unsqueeze(-1),q,p0)
            kl=(q*(q.clamp_min(1e-30).log()-p0.clamp_min(1e-30).log())).sum(-1)
            kl_max=max(kl_max,float(kl.max()));delta_max=max(delta_max,float(delta.abs().max()))
            candidate[key]=q.detach()
        with torch.no_grad():after=score(candidate)
        if kl_max<=max_kl and torch.isfinite(after) and float(after-value)>1e-8:
            return candidate,dict(accepted=True,backtracks=bt,score_before=float(value.detach()),score_after=float(after),
                kl_max=kl_max,max_log_change=delta_max,gradient_norms={k:float(v.norm()) for k,v in gradients.items()},
                derivative='soft categorical score with exponentiated probability-gradient update',
                guarantee='local surrogate ascent only; native sampled graph and later heads require fresh evaluation')
    return {k:v.detach().clone() for k,v in probabilities.items()},dict(accepted=False,reason='no_trusted_surrogate_gain',kl_max=0.)


def apply_to_prediction(predicted,conditioning,index,probabilities):
    """Update one batch row's proposal and matching SC, leaving coordinates intact."""
    out=dict(predicted);cond=dict(conditioning)
    for key,q in probabilities.items():
        if key not in ('atomics','bonds','charges'):raise ValueError('Unsupported categorical modality')
        if q.shape!=predicted[key][index].shape:raise ValueError('Slot shape changed')
        out[key]=predicted[key].clone();out[key][index]=q
        if key in cond:cond[key]=conditioning[key].clone();cond[key][index]=q
    # Existing affinity entries describe the ORIGINAL forward, never the guided graph.
    return out,cond
