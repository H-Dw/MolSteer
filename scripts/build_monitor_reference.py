"""Replay one authentic unguided suffix and record dense endpoint observations."""
import argparse
import json
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import load_config,parse_pdb
from molsteer.common import write_json,file_hash
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import make_reward
from molsteer.molmonitor.features import snapshot
from molsteer.molmonitor.reference import ReferenceTrajectory
from run_exact_continuations import unequal


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());cfg.pop('monitor',None);cfg.pop('categorical_proposal',None)
    destination=Path(a.output)
    if destination.exists():raise ValueError('Reference cannot overwrite a previous experiment')
    adapter=FlowrRootAdapter(cfg);cp=torch.load(cfg['resume_checkpoint'],weights_only=True,map_location='cpu');adapter.restore(cp)
    if cp.get('format')!='flowr_root_live_runtime' or cp['step_index']!=50:raise ValueError('Requires authentic t=0.50 runtime')
    receptor,_=parse_pdb(cfg['receptor']);pt=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=pt.GetRvdw(atom['atomic_number'])
    baseline=torch.load(Path(cfg['reward_reference_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    spec=json.loads(Path(cfg['reward_programs']['creativity']).read_text())
    controls=json.loads(Path(cfg['control_trajectory']).read_text())
    reward=make_reward(spec,baseline,receptor,load_config(),controls[spec['affinity_head']])
    RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
    frames=[]
    while True:
        step=adapter.step_index;t=step/adapter.args.integration_steps
        times=adapter.model._update_times(adapter.times,-1e-4) if t==1. else adapter.times
        with torch.no_grad():pred,cond=adapter.predict(times=times)
        frame=snapshot(adapter.endpoint(pred),reward,t,strain=step%5==0)
        frame['model_time']=float(times[0][0]);frames.append(frame)
        if t==1.:break
        adapter.native_step(pred,cond,adapter.grid[step+1]-adapter.grid[step])
    native=torch.load(Path(cfg['saved_stage']).parent/'final/runtime.pt',weights_only=True,map_location='cpu')
    live=adapter.checkpoint();differences={k:unequal(native[k],live[k]) for k in ['curr','cond','rng','times']}
    if any(differences.values()):raise ValueError('Dense measurement changed native generation')
    bundle=dict(kind='NativeGeometryReference',frames=frames,normalized_progress='FLOWR forward time, 0 -> 1',
        final_model_time_note='final head evaluated at t=1-1e-4; progress remains 1',
        bindings=dict(origin_runtime=cfg['resume_checkpoint'],origin_runtime_sha256=file_hash(cfg['resume_checkpoint']),
            model_checkpoint=cfg['checkpoint'],model_checkpoint_sha256=cp['model_checkpoint_sha256'],
            receptor=cfg['receptor'],target_id=cfg['target_id'],ligand_index=cfg['ligand_index']),
        normalization={k:spec[k] for k in ['bond_tolerance_fraction','angle_tolerance_degrees','protein_vdw_ratio','intra_vdw_ratio']},
        fidelity=dict(all_exact=True,different_paths=differences),
        limitations=['one matched trajectory, not a calibrated population distribution',
            'graph-dependent references compared only when chemical identity and reference match'])
    bundle['coarse_050_075_100']=ReferenceTrajectory(bundle).coarse_summary()
    write_json(destination,bundle)
    print(json.dumps(dict(frames=len(frames),all_exact=True,coarse=bundle['coarse_050_075_100']),indent=2))


if __name__=='__main__':main()
