"""Numerical screen of the complete endpoint-to-current-state derivative."""
import torch
from molsteer.molexecutor.program import evaluate_with_state


def check_live_gradient(adapter, reward, epsilons=(.002,.005,.01), sample_count=3):
    if hasattr(reward,'control_gradient'):
        return check_expert_live_gradient(adapter,reward,epsilons,sample_count)
    x=adapter.curr['coords'].detach().requires_grad_(True)
    pred,_=adapter.predict(coordinates=x)
    value,_=evaluate_with_state(reward,adapter,adapter.endpoint(pred),x)
    gradient,=torch.autograd.grad(value,x)
    magnitudes=gradient[adapter.index].detach().abs()
    editable=adapter.config.get('editable_atom_ids')
    if editable is not None:
        mask=torch.zeros(magnitudes.shape[0],dtype=torch.bool,device=magnitudes.device)
        mask[editable]=True
        magnitudes=magnitudes.masked_fill(~mask[:,None],-1)
    ids=magnitudes.reshape(-1).topk(min(sample_count,3*len(editable) if editable is not None else magnitudes.numel())).indices.tolist()
    samples=[]
    for eps in epsilons:
        for index in ids:
            atom,axis=divmod(index,3)
            d=torch.zeros_like(x);d[adapter.index,atom,axis]=eps
            with torch.no_grad():
                pp,_=adapter.predict(coordinates=x+d);pm,_=adapter.predict(coordinates=x-d)
                ep,em=adapter.endpoint(pp),adapter.endpoint(pm)
                vp,_=evaluate_with_state(reward,adapter,ep,x+d)
                vm,_=evaluate_with_state(reward,adapter,em,x-d)
            fd=float((vp-vm)/(2*eps));ag=float(gradient[adapter.index,atom,axis])
            samples.append(dict(epsilon=eps,atom_id=atom,axis=axis,autograd=ag,finite_difference=fd,
                relative_error=abs(fd-ag)/max(abs(fd),abs(ag),1e-8),same_discrete_graph=reward.graph(ep)==reward.graph(em)))
    # Coarser perturbations reduce float32 subtractive cancellation. Preserve all
    # scales in the record; graph crossings cannot qualify as a derivative check.
    stable=[s for s in samples if s['epsilon']==max(epsilons)]
    passed=all(s['same_discrete_graph'] and s['autograd']*s['finite_difference']>0 and s['relative_error']<.15 for s in stable)
    return dict(passed=passed,criterion='At largest declared epsilon: same graph, matching sign, relative error < 0.15; float32 screen, not an analytic guarantee',
                precision=adapter.precision,samples=samples)


def check_expert_live_gradient(adapter,reward,epsilons,sample_count):
    """Check every component in the shared variable, including constraints and zero derivatives."""
    x=adapter.curr['coords'].detach().requires_grad_(True)
    from molsteer.molexecutor.expert_control import probe_predict
    pred,_=probe_predict(adapter,coordinates=x)
    def parts(prediction,coordinates):
        return reward.components(adapter.endpoint(prediction),
            adapter.world_state_coordinates(coordinates) if reward.uses_state_view else None)
    values=parts(pred,x);checks={}
    editable=adapter.config.get('editable_atom_ids')
    if not editable: raise ValueError('Expert preflight requires editable atom IDs')
    for key,value in values.items():
        gradient,=torch.autograd.grad(value,x,retain_graph=True)
        ids=gradient[adapter.index,editable].detach().abs().reshape(-1).topk(min(sample_count,3*len(editable))).indices.tolist()
        samples=[]
        for eps in epsilons:
            for idx in ids:
                local,axis=divmod(idx,3);atom=editable[local]
                d=torch.zeros_like(x);d[adapter.index,atom,axis]=eps
                try:
                    with torch.no_grad():
                        pp,_=probe_predict(adapter,coordinates=x+d);pm,_=probe_predict(adapter,coordinates=x-d)
                        vp,vm=parts(pp,x+d)[key],parts(pm,x-d)[key]
                    fd=float((vp-vm)/(2*eps));ag=float(gradient[adapter.index,atom,axis])
                    passed=abs(fd-ag)<=1e-6+.15*max(abs(fd),abs(ag))
                    samples.append(dict(epsilon=eps,atom_id=atom,axis=axis,autograd=ag,finite_difference=fd,passed=passed))
                except ValueError:
                    samples.append(dict(epsilon=eps,atom_id=atom,axis=axis,passed=False,reason='domain_or_graph_boundary'))
        checks[key]=dict(passed=all(s['passed'] for s in samples if s['epsilon']==max(epsilons)),samples=samples)
    return dict(passed=all(c['passed'] for c in checks.values()),components=checks,
                precision=adapter.precision,scope='Every local function pulled back through the live adapter; no terminal efficacy claim')
