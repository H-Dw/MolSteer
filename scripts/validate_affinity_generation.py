"""Paired ablations from one authentic checkpoint, with no seed sweep."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import parse_pdb,load_config
from molsteer.common import write_json,file_hash
from molsteer.molthinker.affinity import build_affinity_program
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import MolecularReward,make_reward
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molmonitor.live_gradient import check_live_gradient
from run_exact_continuations import unequal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.output).resolve()
    if (root/'runs.json').exists():raise ValueError('Use a fresh validation directory')
    root.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.base_config).read_text())
    config.update(output=str(root),guidance_interval=[.5,1.],record_tensor_trace=True)
    if config['start_step']!=50:raise ValueError('This experiment is restricted to the declared t=0.50 checkpoint')
    adapter=FlowrRootAdapter(config)
    cp=torch.load(config['resume_checkpoint'],map_location='cpu',weights_only=True)
    adapter.restore(cp)
    if cp.get('format')!='flowr_root_live_runtime':raise ValueError('Authentic runtime capture required')
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    spec=json.loads(Path(config['reward_programs']['creativity']).read_text())
    baseline=torch.load(Path(config['reward_reference_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(config['receptor']);table=Chem.GetPeriodicTable()
    for a in receptor:a['vdw_radius']=table.GetRvdw(a['atomic_number'])
    vocab=load_config();records=[]
    reasoning=root/'reasoning'/'t_0.50';reasoning.mkdir(parents=True,exist_ok=True)
    old_reasoning=Path(config['reward_programs']['creativity']).parent
    for source in old_reasoning.iterdir():
        if source.is_file():shutil.copy2(source,reasoning/source.name)
    intent=dict(primary_affinity='pkd',direction='maximize',diagnostic_time=.5,
        control_interval=[.5,1.],allow_substantial_identity_changes=True,atom_slots=20,
        other_heads='Report pic50, pki and pec50 separately; no composite mean',
        evidence_policy='Only the supplied t=0.50 report defines the diagnosis; live checks only test runtime feasibility')
    write_json(root/'DesignIntent.json',intent)
    legacy=MolecularReward(spec,baseline,receptor,vocab)
    controls={key:{} for key in ['pic50','pki','pkd','pec50']}
    adapter.restore(cp)
    while adapter.step_index<adapter.args.integration_steps:
        with torch.no_grad():prediction,cond=adapter.predict()
        t=float(adapter.times[0][0])
        for key in controls:controls[key][str(t)]=float(prediction['affinity'][key][adapter.index])
        adapter.native_step(prediction,cond,adapter.grid[adapter.step_index+1]-adapter.grid[adapter.step_index])
    times=adapter.model._update_times(adapter.times,-1e-4)
    with torch.no_grad():prediction,_=adapter.predict(times=times)
    for key in controls:controls[key][str(float(times[0][0]))]=float(prediction['affinity'][key][adapter.index])
    write_json(root/'control_trajectory.json',controls)
    designs=[('control',None,0.,.02,.5,None),('legacy',None,300.,.02,.5,None),
        ('affinity_retained',dict(retain_initial=True,geometry_scope='diagnosed'),300.,.02,.5,None),
        ('affinity_released',dict(geometry_scope='diagnosed'),300.,.02,.5,None),
        ('affinity_global',dict(geometry_scope='all'),300.,.02,.5,None),
        ('affinity_global_large',dict(geometry_scope='all'),300.,.04,2.,None),
        ('affinity_global_large_a3',dict(geometry_scope='all',affinity_weight=3.),300.,.04,2.,None),
        ('branch_cool',dict(geometry_scope='all',affinity_weight=3.),300.,.04,2.,.7),
        ('branch_warm',dict(geometry_scope='all',affinity_weight=3.),300.,.04,2.,1.5),
        ('branch_hot',dict(geometry_scope='all',affinity_weight=3.),300.,.04,2.,2.)]
    write_json(root/'planned_ablations.json',[dict(label=d[0],composition=d[1],eta=d[2],step=d[3],path=d[4],temperature=d[5]) for d in designs])
    seen=set()
    for label,composition,eta,step_cap,path_cap,temperature in designs:
        program=build_affinity_program(spec,intent,**composition) if composition else spec
        program_path=root/'programs'/f'{label}.json';write_json(program_path,program)
        reward=make_reward(program,baseline,receptor,vocab,controls.get(program.get('affinity_head')))
        adapter.restore(cp)
        if composition and program['program_id'] not in seen:
            if hasattr(reward,'set_time'):reward.set_time(float(adapter.times[0][0]))
            preflight=check_live_gradient(adapter,reward)
            write_json(root/'validation'/f'{label}_gradient.json',preflight)
            if not preflight['passed']:raise ValueError('Live finite-difference validation failed: '+label)
            seen.add(program['program_id'])
        adapter.restore(cp)
        arm='creativity' if eta else 'unguided';directory=root/'continuations'/label
        variant=copy.deepcopy(config)
        variant.update(output=str(directory),arms=[arm],control_trajectory=str(root/'control_trajectory.json'),
            reward_programs=dict(creativity=str(program_path)),
            budget=dict(strength=eta or 1.,max_step_angstrom=step_cap,max_path_angstrom=path_cap))
        if temperature:
            variant['categorical_proposal']=dict(start=.5,end=.7,temperature=temperature,confidence_ceiling=.95)
        adapter.config=variant
        write_json(directory/'execution.json',variant)
        summary=run_suffix(adapter,reward,directory/arm,GuidanceBudget(**variant['budget']),arm)
        if label=='control':
            generated=torch.load(directory/arm/'resume_final.pt',weights_only=True,map_location='cpu')
            native=torch.load(Path(config['saved_stage']).parent/'final/runtime.pt',weights_only=True,map_location='cpu')
            diffs={key:unequal(native[key],generated[key]) for key in ['curr','cond','prior','times','rng']}
            suffix=Path(config['target_id'])/f"ligand_{config['ligand_index']:03d}"/'final'
            for name in ['state.pt','structure_affinity_prediction.pt','world_prediction.pt']:
                a=torch.load(Path(config['saved_stage']).parent/'final'/name,weights_only=True,map_location='cpu')
                b=torch.load(directory/arm/suffix/name,weights_only=True,map_location='cpu')
                diffs[name]=unequal(a,b)
            write_json(root/'validation'/'control_exact.json',dict(all_exact=not any(diffs.values()),different_paths=diffs))
            if any(diffs.values()):raise ValueError('Unguided control no longer matches native generation')
        records.append(dict(stage='t_0.50',label=label,strength=eta,clear_initial_self_condition=False,
            directory=str(directory),arm=arm,config=variant,summary=summary))
        write_json(root/'runs.json',records)
        print('AFFINITY_RUN '+label+' '+json.dumps(summary),flush=True)
    source_root=Path(__file__).resolve().parents[1]
    write_json(root/'provenance.json',dict(checkpoint=config['resume_checkpoint'],
        checkpoint_sha256=file_hash(config['resume_checkpoint']),rng_policy='Restore the same captured RNG for every complete suffix; no seed sweep',
        diagnosis_stage='t_0.50',generation_interval=[.5,1.],source_hashes={str(p.relative_to(source_root)):file_hash(p)
            for folder in ['src/molsteer','scripts'] for p in (source_root/folder).rglob('*.py')}))


if __name__=='__main__':main()
