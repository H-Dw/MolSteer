"""Verify restart, untouched batch rows, RNG and input provenance on real runs."""
import argparse,json
from pathlib import Path
import torch
from rdkit import Chem
from molreader.io import parse_pdb,load_config
from molsteer.common import write_json,file_hash
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import make_reward
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.interfaces import GuidanceBudget
from run_exact_continuations import unequal


def load(path):return torch.load(path,weights_only=True,map_location='cpu')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();runs=json.loads((root/'runs.json').read_text())
    control=next(r for r in runs if r['label']=='control')
    native=load(Path(control['directory'])/'unguided/resume_final.pt');checks={}
    for r in runs:
        final=load(Path(r['directory'])/r['arm']/'resume_final.pt')
        def others(t):
            if torch.is_tensor(t) and t.ndim and t.shape[0]==3:return t[:2]
            if isinstance(t,dict):return {k:others(v) for k,v in t.items()}
            if isinstance(t,(list,tuple)):return [others(v) for v in t]
            return t
        checks[r['label']]=dict(untouched_current=not unequal(others(native['curr']),others(final['curr'])),
            untouched_self_condition=not unequal(others(native['cond']),others(final['cond'])),
            identical_rng=not unequal(native['rng'],final['rng']))
    r=next(r for r in runs if r['label']=='outcome_adaptive');cfg=r['config']
    adapter=FlowrRootAdapter(cfg);adapter.restore(load(Path(r['directory'])/'creativity/resume_t_0.75.pt'))
    spec=json.loads(Path(cfg['reward_programs']['creativity']).read_text())
    baseline=load(Path(cfg['saved_stage'])/'world_prediction.pt')
    baseline={k:v[0].to(adapter.device) for k,v in baseline.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(cfg['receptor']);pt=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=pt.GetRvdw(atom['atomic_number'])
    controls=json.loads(Path(cfg['control_trajectory']).read_text())
    reward=make_reward(spec,baseline,receptor,load_config(),controls[spec['affinity_head']])
    out=root/'restart_audit'
    if out.exists():raise ValueError('Restart audit output already exists')
    run_suffix(adapter,reward,out,GuidanceBudget(**cfg['budget']),'creativity')
    expected=load(Path(r['directory'])/'creativity/resume_final.pt');actual=adapter.checkpoint()
    difference={k:unequal(expected[k],actual[k]) for k in ('curr','cond','times','rng','guidance_state')}
    provenance=json.loads((root/'provenance.json').read_text())
    sources={k:file_hash(Path(__file__).resolve().parents[1]/k)==v for k,v in provenance['source_hashes'].items()}
    result=dict(batch_checks=checks,restart_t075=dict(all_exact=not any(difference.values()),differences=difference),
        source_unchanged=all(sources.values()),changed_sources=[k for k,v in sources.items() if not v],
        native_runtime_unchanged=file_hash(provenance['origin_runtime'])==provenance['origin_runtime_sha256'])
    result['all_passed']=all(all(c.values()) for c in checks.values()) and result['restart_t075']['all_exact'] and result['source_unchanged'] and result['native_runtime_unchanged']
    write_json(root/'validation/runtime_audit.json',result)
    print(json.dumps(result),flush=True)
    if not result['all_passed']:raise ValueError('Runtime audit failed')


if __name__=='__main__':main()
