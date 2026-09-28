"""Native categorical-gradient audit and two declared coordinate-policy ablations."""
import argparse,copy,json,statistics
from pathlib import Path
import torch
from rdkit import Chem
from molreader.io import load_config,parse_pdb
from molsteer.common import digest,write_json,file_hash
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import make_reward
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molexecutor.discrete_gradient import motif_score,guided_probabilities,apply_to_prediction
from run_exact_continuations import unequal


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prior',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    prior=Path(a.prior).resolve();out=Path(a.output).resolve()
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True)
    config=json.loads((prior/'continuations/outcome_continuous/execution.json').read_text())
    adapter=FlowrRootAdapter(config);cp=torch.load(config['resume_checkpoint'],weights_only=True,map_location='cpu');adapter.restore(cp)
    state=dict(adapter.curr);state['coords']=state['coords'].detach().requires_grad_(True)
    for key in ('atomics','bonds','charges'):state[key]=state[key].detach().requires_grad_(True)
    predicted,cond=adapter.predict(state=state);pkd=predicted['affinity']['pkd'][adapter.index].sum()
    keys=('atomics','bonds','charges')
    in_grad=torch.autograd.grad(pkd,tuple(state[k] for k in keys),allow_unused=True,retain_graph=True)
    out_grad=torch.autograd.grad(pkd,tuple(predicted[k] for k in keys),allow_unused=True,retain_graph=True)
    probe=dict(current_category_gradient={k:None if g is None else float(g.norm()) for k,g in zip(keys,in_grad)},
        head_to_output_probability_gradient={k:None if g is None else float(g.norm()) for k,g in zip(keys,out_grad)},
        interpretation='None means disconnected from this head, not a measured near-zero chemical effect')
    vocab=load_config();native_sdf=json.loads((prior/'native_reference.json').read_text())['native_final_sdf']['path']
    m=Chem.MolFromMolFile(native_sdf,removeHs=True)
    # Mechanism-only deaza hypothesis, not a literature-validated potency claim.
    candidate=Chem.RWMol(m);atom=candidate.GetAtomWithIdx(15);atom.SetAtomicNum(6);atom.SetFormalCharge(0);atom.SetNumExplicitHs(0);atom.SetNoImplicit(False)
    candidate=candidate.GetMol();Chem.SanitizeMol(candidate)
    hypothesis=dict(assignments=[dict(feature='atomics',slots=[15],category=vocab['atomic_tokens'].index('C')),
        dict(feature='charges',slots=[15],category=vocab['charge_tokens'].index(0)),
        dict(feature='bonds',slots=[12,15],category=vocab['bond_orders'].index(2.))])
    p={k:predicted[k][adapter.index].detach() for k in keys}
    masks={k:torch.zeros(v.shape[:-1],device=v.device,dtype=torch.bool) for k,v in p.items()}
    masks['atomics'][15]=True;masks['charges'][15]=True;masks['bonds'][12,15]=True;masks['bonds'][15,12]=True
    score=lambda values:motif_score(values,[hypothesis])
    q,detail=guided_probabilities(p,score,masks,strength=1.,max_kl=.02,max_log_change=1.)
    probe['soft_guidance']=detail
    probe['hypothesis']=dict(native_smiles=Chem.MolToSmiles(m),candidate_smiles=Chem.MolToSmiles(candidate),
        changed_atom=15,claim='Mechanism-only N-to-C hypothesis at a buried polar site. No predicted activity improvement is asserted.',template=hypothesis)
    probe['probability_changes']=[dict(feature=k,slots=t['slots'],category=t['category'],before=float(p[k][tuple(t['slots'])+(t['category'],)]),
        after=float(q[k][tuple(t['slots'])+(t['category'],)])) for t in hypothesis['assignments'] for k in [t['feature']]]
    # Differential check within the simplex, including an undirected bond.
    x={k:v.double().requires_grad_(True) for k,v in p.items()};g=torch.autograd.grad(score(x),tuple(x.values()))
    fd=[]
    for term in hypothesis['assignments']:
        k=term['feature'];idx=tuple(term['slots']);c=term['category'];other=int(p[k][idx].argmax())
        if c==other:other=(c+1)%p[k].shape[-1]
        eps=min(1e-5,float(p[k][idx+(other,)])/4,float(p[k][idx+(c,)])/4)
        if eps<1e-12:continue
        direction=torch.zeros_like(x[k]);direction[idx+(c,)]=1.;direction[idx+(other,)]=-1.
        if k=='bonds':direction[idx[::-1]+(c,)]=1.;direction[idx[::-1]+(other,)]=-1.
        plus=dict(x);minus=dict(x);plus[k]=x[k]+eps*direction;minus[k]=x[k]-eps*direction
        numeric=float(((score(plus)-score(minus))/(2*eps)).detach());analytic=float((g[keys.index(k)]*direction).sum().detach())
        fd.append(dict(feature=k,epsilon=eps,analytic=analytic,numeric=numeric,relative_error=abs(analytic-numeric)/max(abs(analytic),abs(numeric),1e-10)))
    probe['finite_differences']=fd
    # Matched one-step test uses the real native categorical integrator and SC.
    adapter.restore(cp)
    with torch.no_grad():original,condition=adapter.predict()
    adapter.native_step(original,condition,adapter.grid[51]-adapter.grid[50]);native=adapter.checkpoint()
    adapter.restore(cp)
    modified,condition=apply_to_prediction(original,condition,adapter.index,q)
    adapter.native_step(modified,condition,adapter.grid[51]-adapter.grid[50]);guided=adapter.checkpoint()
    probe['native_step']=dict(rng_identical=not unequal(native['rng'],guided['rng']),
        other_batch_unchanged=all(torch.equal(native['curr'][k][:2],guided['curr'][k][:2]) for k in ('coords',)+keys),
        categorical_changed_slots={k:int((native['curr'][k][2].argmax(-1)!=guided['curr'][k][2].argmax(-1)).sum()) for k in keys},
        guided_sc_matches=all(torch.equal(guided['cond'][k][2],q[k].cpu()) for k in keys if k in guided['cond']))
    probe['passed']=detail['accepted'] and len(fd)>=2 and all(v['relative_error']<1e-4 for v in fd) and probe['native_step']['rng_identical'] and probe['native_step']['other_batch_unchanged'] and probe['native_step']['guided_sc_matches']
    write_json(out/'DiscreteGradientAudit.json',probe)
    if not probe['passed']:raise ValueError('Discrete guidance mechanism audit failed')
    receptor,_=parse_pdb(config['receptor']);pt=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=pt.GetRvdw(atom['atomic_number'])
    baseline=torch.load(Path(config['saved_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    controls=json.loads(Path(config['control_trajectory']).read_text());base=json.loads((prior/'programs/outcome_continuous.json').read_text());runs=[]
    for label,wS,wI,wD in [('no_projection',1.,.5,.25),('rebalanced_no_projection',.25,1.,.5)]:
        spec=copy.deepcopy(base);spec['parent_program_id']=spec.pop('program_id');spec['gradient_policy']['project_conflicting_components']=False
        spec['outcome_weights'].update(strain=wS,contact=wI,desolvation=wD);spec['weights']=[1.,wS,wI,wD]
        spec['experiment_note']='Predeclared coordinate-only attribution ablation; no categorical medicinal guidance or source evidence activation'
        spec['program_id']='rp_'+digest(spec)[:24];path=out/'programs'/f'{label}.json';write_json(path,spec)
        cfg=copy.deepcopy(config);cfg.update(output=str(out/'continuations'/label),reward_programs=dict(creativity=str(path)))
        adapter.config=cfg;adapter.restore(cp);reward=make_reward(spec,baseline,receptor,vocab,controls[spec['affinity_head']])
        write_json(Path(cfg['output'])/'execution.json',cfg)
        result=run_suffix(adapter,reward,Path(cfg['output'])/'creativity',GuidanceBudget(**cfg['budget']),'creativity')
        runs.append(dict(label=label,summary=result,config=cfg));write_json(out/'runs.json',runs)
    write_json(out/'provenance.json',dict(native_runtime_sha256=file_hash(config['resume_checkpoint']),
        reference_directory=str(prior),sampler_seed_sweep=False,
        implementation={str(p.relative_to(Path(__file__).resolve().parents[1])):file_hash(p) for p in (Path(__file__).resolve().parents[1]/'src').rglob('*.py')}))


if __name__=='__main__':main()
