"""One captured random state; exact-control checks precede guidance branches."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,RDLogger
from molreader.io import parse_pdb,load_config
from molsteer.common import write_json
from molsteer.molexecutor.flowr import FlowrRootAdapter,snapshot_rng
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molexecutor.program import MolecularReward
from molsteer.molmonitor.live_gradient import check_live_gradient


def unequal(a,b,path=''):
    if torch.is_tensor(a):
        same=torch.is_tensor(b) and a.shape==b.shape and (torch.equal(a,b) or
            (a.is_floating_point() and b.is_floating_point() and torch.allclose(a,b,rtol=0,atol=0,equal_nan=True)))
        return [] if same else [path]
    if isinstance(a,dict):
        return ([path+':keys'] if set(a)!=set(b) else [])+[p for k in a.keys()&b.keys() for p in unequal(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,(list,tuple)):
        return ([path+':length'] if len(a)!=len(b) else [])+[p for i,(x,y) in enumerate(zip(a,b)) for p in unequal(x,y,path+'/'+str(i))]
    return [] if a==b else [path]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root).resolve();index=root/'runs.json'
    records=json.loads(index.read_text()) if index.exists() else []
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    variants=[('eta_0',0.,False),('eta_1',1.,False),('eta_10',10.,False),('eta_100',100.,False),('eta_300',300.,False),('zero_initial_sc',0.,True)]
    for stage in ['t_0.50','t_0.25']:
        if all(any(r['stage']==stage and r['label']==label for r in records) for label,_,_ in variants):continue
        config=json.loads((root/'configurations'/f'{stage}.json').read_text())
        adapter=FlowrRootAdapter(config)
        cp=torch.load(config['resume_checkpoint'],weights_only=True,map_location='cpu')
        adapter.restore(cp)
        original_root=Path(config['saved_stage']).parent
        suffix=Path(config['target_id'])/f"ligand_{config['ligand_index']:03d}"
        native_final=torch.load(original_root/'final/runtime.pt',weights_only=True,map_location='cpu')
        baseline=torch.load(Path(config['reward_reference_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
        baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
        receptor,_=parse_pdb(config['receptor']);table=Chem.GetPeriodicTable()
        for atom in receptor:atom['vdw_radius']=table.GetRvdw(atom['atomic_number'])
        spec=json.loads(Path(config['reward_programs']['creativity']).read_text())
        reward=MolecularReward(spec,baseline,receptor,load_config())
        # The initial prediction and a one-time SC ablation use exactly the same RNG.
        with torch.no_grad():pfull,_=adapter.predict()
        initial={k:float((pfull[k][adapter.index:adapter.index+1].cpu()-torch.load(Path(config['saved_stage'])/'structure_affinity_prediction.pt',weights_only=True)[k]).abs().max())
                 for k in ['coords','atomics','charges','bonds','mask']}
        zero={k:torch.zeros_like(v) for k,v in adapter.cond.items()}
        adapter.restore(cp)
        with torch.no_grad():pzero,_=adapter.predict(cond=zero)
        head_ablation={k:float((pfull[k][adapter.index]-pzero[k][adapter.index]).abs().max()) for k in ['coords','atomics','charges','bonds']}
        write_json(root/'validation'/f'{stage}_initial_restore.json',dict(max_abs=initial,all_exact=all(v==0 for v in initial.values()),
            zero_initial_self_condition_head_max_abs=head_ablation,checkpoint_format=cp['format']))
        if any(v!=0 for v in initial.values()):raise ValueError('Initial head does not reproduce the captured head')
        adapter.restore(cp)
        unavailable=[]
        # Early noisy endpoints may lack the chemical graph required by the SAME
        # reward. Screen the first applicable point on the untouched native path.
        while True:
            with torch.no_grad():probe,conditioning=adapter.predict()
            try:reward.evaluate(adapter.endpoint(probe))
            except ValueError as exc:
                unavailable.append(dict(step=adapter.step_index,reason=str(exc)))
                if adapter.step_index>=adapter.args.integration_steps-1:raise ValueError('No applicable gradient preflight point')
                adapter.native_step(probe,conditioning,adapter.grid[adapter.step_index+1]-adapter.grid[adapter.step_index])
                continue
            preflight=check_live_gradient(adapter,reward)
            preflight.update(evaluated_step=adapter.step_index,initial_unavailable=unavailable,
                scope='First applicable endpoint on unchanged native prefix; each actual step still enforces reward applicability')
            break
        write_json(root/'validation'/f'{stage}_gradient_preflight.json',preflight)
        if not preflight['passed']:raise ValueError('Finite-difference preflight failed at '+stage)
        for label,strength,clear_sc in variants:
            if any(r['stage']==stage and r['label']==label for r in records):continue
            adapter.restore(cp)
            if clear_sc:adapter.cond={k:torch.zeros_like(v) for k,v in adapter.cond.items()}
            arm='creativity' if strength else 'unguided'
            directory=root/'continuations'/stage/label
            variant_config=dict(config,output=str(directory),arms=[arm],
                budget=dict(strength=strength or 1.,max_step_angstrom=.02,max_path_angstrom=.5),
                sweep_variant=dict(label=label,guidance_enabled=bool(strength),
                    effective_strength=strength,clear_initial_self_condition=clear_sc))
            write_json(directory/'execution.json',variant_config)
            summary=run_suffix(adapter,reward,directory/arm,GuidanceBudget(strength=strength or 1.,max_step_angstrom=.02,max_path_angstrom=.5),arm)
            if label=='eta_0':
                generated=torch.load(directory/arm/'resume_final.pt',weights_only=True,map_location='cpu')
                diff={k:unequal(native_final[k],generated[k]) for k in ['curr','cond','prior','times','rng']}
                for filename in ['state.pt','structure_affinity_prediction.pt','world_prediction.pt']:
                    original=torch.load(original_root/'final'/filename,weights_only=True,map_location='cpu')
                    replay=torch.load(directory/arm/suffix/'final'/filename,weights_only=True,map_location='cpu')
                    diff[filename]=unequal(original,replay)
                write_json(root/'validation'/f'{stage}_unguided_exact.json',dict(different_paths=diff,all_exact=all(not v for v in diff.values())))
                if any(diff.values()):raise ValueError('Unguided final differs from original live trajectory')
            record=dict(stage=stage,label=label,strength=strength,clear_initial_self_condition=clear_sc,
                directory=str(directory),arm=arm,config=variant_config,summary=summary)
            records.append(record);write_json(index,records)
            print('EXACT_RUN '+json.dumps({k:v for k,v in record.items() if k!='config'}),flush=True)
        del adapter,reward
        torch.cuda.empty_cache()


if __name__=='__main__':main()
