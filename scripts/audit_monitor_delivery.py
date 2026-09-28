"""Recheck final delivered runtime against saved physical continuations."""
import argparse
import json
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import load_config,parse_pdb
from molsteer.common import write_json,file_hash
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import make_reward
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molexecutor.engine import run_suffix
from run_exact_continuations import unequal


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();records=json.loads((root/'all_runs.json').read_text());checks=[]
    RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
    adapter=None
    for label,start,final in [('adaptive_same_budget','resume_t_0.75.pt','resume_final.pt'),('stress_monitored','resume_t_0.80.pt','resume_revision.pt')]:
        record=next(r for r in records if r['label']==label);cfg=record['config'].copy()
        original=Path(record['directory'])/'creativity';cfg['resume_checkpoint']=str(original/start)
        if adapter is None:adapter=FlowrRootAdapter(cfg)
        adapter.config=cfg;adapter.restore(torch.load(original/start,weights_only=True,map_location='cpu'))
        baseline=torch.load(Path(cfg['reward_reference_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
        baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
        receptor,_=parse_pdb(cfg['receptor']);pt=Chem.GetPeriodicTable()
        for atom in receptor:atom['vdw_radius']=pt.GetRvdw(atom['atomic_number'])
        spec=json.loads(Path(cfg['reward_programs']['creativity']).read_text())
        controls=json.loads(Path(cfg['control_trajectory']).read_text())
        reward=make_reward(spec,baseline,receptor,load_config(),controls[spec['affinity_head']])
        directory=root/'delivery_audit'/label
        summary=run_suffix(adapter,reward,directory,GuidanceBudget(**cfg['budget']),'creativity')
        expected=torch.load(original/final,weights_only=True,map_location='cpu');actual=adapter.checkpoint()
        diff={k:unequal(expected[k],actual[k]) for k in ['curr','cond','rng','times']}
        diff['path_used']=unequal(expected['guidance_state']['path_used'],actual['guidance_state']['path_used'])
        check=dict(label=label,physical_state_exact=not any(diff.values()),differences=diff,summary=summary)
        checks.append(check)
        if any(diff.values()):raise ValueError('Delivered runtime changed the established physical continuation')
        pending=actual['guidance_state'].get('pending_request')
        if pending:
            write_json(root/'final_revision_request.json',pending)
            context=json.loads((directory/'feedback'/(pending['request_id']+'.context.json')).read_text())
            write_json(root/'final_revision_context.json',context)
    write_json(root/'delivery_audit.json',checks)
    source_root=Path(__file__).resolve().parents[1]
    write_json(root/'delivered_source_hashes.json',{str(path.relative_to(source_root)):file_hash(path) for folder in ['src/molsteer','scripts'] for path in (source_root/folder).rglob('*.py')})


if __name__=='__main__':main()
