"""Numerical screen of the complete endpoint-to-current-state derivative."""
import torch


def check_live_gradient(adapter, reward, epsilons=(.002,.005,.01), sample_count=3):
    x=adapter.curr['coords'].detach().requires_grad_(True)
    pred,_=adapter.predict(coordinates=x)
    value,_=reward.evaluate(adapter.endpoint(pred))
    gradient,=torch.autograd.grad(value,x)
    ids=gradient[adapter.index].detach().abs().reshape(-1).topk(sample_count).indices.tolist()
    samples=[]
    for eps in epsilons:
        for index in ids:
            atom,axis=divmod(index,3)
            d=torch.zeros_like(x);d[adapter.index,atom,axis]=eps
            with torch.no_grad():
                pp,_=adapter.predict(coordinates=x+d);pm,_=adapter.predict(coordinates=x-d)
                ep,em=adapter.endpoint(pp),adapter.endpoint(pm)
                vp,_=reward.evaluate(ep);vm,_=reward.evaluate(em)
            fd=float((vp-vm)/(2*eps));ag=float(gradient[adapter.index,atom,axis])
            samples.append(dict(epsilon=eps,atom_id=atom,axis=axis,autograd=ag,finite_difference=fd,
                relative_error=abs(fd-ag)/max(abs(fd),abs(ag),1e-8),same_discrete_graph=reward.graph(ep)==reward.graph(em)))
    # Coarser perturbations reduce float32 subtractive cancellation. Preserve all
    # scales in the record; graph crossings cannot qualify as a derivative check.
    stable=[s for s in samples if s['epsilon']==max(epsilons)]
    passed=all(s['same_discrete_graph'] and s['autograd']*s['finite_difference']>0 and s['relative_error']<.15 for s in stable)
    return dict(passed=passed,criterion='At largest declared epsilon: same graph, matching sign, relative error < 0.15; float32 screen, not an analytic guarantee',
                precision=adapter.precision,samples=samples)
