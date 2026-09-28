"""Paired runtime validation of adaptive strength and revision handoffs."""
import argparse
import copy
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
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();cfg=json.loads(Path(a.config).read_text())
    if (root/'all_runs.json').exists():raise ValueError('Use fresh outputs')
    adapter=FlowrRootAdapter(cfg);cp=torch.load(cfg['resume_checkpoint'],weights_only=True,map_location='cpu')
    baseline=torch.load(Path(cfg['reward_reference_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(cfg['receptor']);pt=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=pt.GetRvdw(atom['atomic_number'])
    controls=json.loads(Path(cfg['control_trajectory']).read_text());vocab=load_config()
    good=json.loads(Path(cfg['reward_programs']['creativity']).read_text())
    source=Path(cfg['output']).parents[1]
    weak=json.loads((source/'programs/affinity_global_large_a3.json').read_text())
    knowledge=str(Path(__file__).resolve().parents[1]/'knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md')
    designs=[('fixed_300',False,good,300.,.04,2.),('adaptive_same_budget',True,good,30.,.04,2.),
        ('adaptive_wide',True,good,30.,.12,3.),('stress_fixed',False,weak,10000.,.2,5.),
        ('stress_monitored',True,weak,10000.,.2,5.)]
    write_json(root/'planned_runs.json',[dict(label=x[0],monitor=x[1],program_id=x[2]['program_id'],eta=x[3],step=x[4],path=x[5]) for x in designs])
    RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
    all_records=[];complete=[]
    for label,enabled,spec,strength,step_cap,path_cap in designs:
        directory=root/'continuations'/label
        path=root/'programs'/f'{label}.json';write_json(path,spec)
        variant=copy.deepcopy(cfg);variant.update(output=str(directory),reward_programs=dict(creativity=str(path)),
            budget=dict(strength=strength,max_step_angstrom=step_cap,max_path_angstrom=path_cap))
        variant.pop('monitor',None)
        if enabled:variant['monitor']=dict(reference=str(root/'reference.json'),knowledge_path=knowledge,policy=dict(initial_eta=strength))
        adapter.config=variant;adapter.restore(cp)
        reward=make_reward(spec,baseline,receptor,vocab,controls[spec['affinity_head']])
        write_json(directory/'execution.json',variant)
        summary=run_suffix(adapter,reward,directory/'creativity',GuidanceBudget(**variant['budget']),'creativity')
        record=dict(stage='t_0.50',label=label,strength=strength,clear_initial_self_condition=False,
            directory=str(directory),arm='creativity',config=variant,summary=summary)
        all_records.append(record);write_json(root/'all_runs.json',all_records)
        if summary.get('status','complete')=='complete':complete.append(record);write_json(root/'runs.json',complete)
        if label=='fixed_300':
            old=torch.load(Path(cfg['output'])/'creativity/resume_final.pt',weights_only=True,map_location='cpu')
            now=adapter.checkpoint();difference={k:unequal(old[k],now[k]) for k in ['curr','cond','rng','times']}
            write_json(root/'fixed_regression.json',dict(all_exact=not any(difference.values()),differences=difference))
            if any(difference.values()):raise ValueError('Static executor regression')
        if label=='adaptive_same_budget' and summary.get('status')=='complete':
            final=adapter.checkpoint()
            restart=torch.load(directory/'creativity/resume_t_0.75.pt',weights_only=True,map_location='cpu')
            resumed_cfg=copy.deepcopy(variant);resumed_cfg['resume_checkpoint']=str(directory/'creativity/resume_t_0.75.pt')
            adapter.config=resumed_cfg;adapter.restore(restart)
            resumed=run_suffix(adapter,reward,root/'resume_audit'/'creativity',GuidanceBudget(**variant['budget']),'creativity')
            actual=adapter.checkpoint();diff={k:unequal(final[k],actual[k]) for k in ['curr','cond','rng','times','guidance_state']}
            write_json(root/'monitor_resume_exact.json',dict(all_exact=not any(diff.values()),differences=diff,summary=resumed))
            if any(diff.values()):raise ValueError('Monitor restart changed the continuation')
        print('MONITOR_RUN '+label+' '+json.dumps(summary),flush=True)
    source_root=Path(__file__).resolve().parents[1]
    write_json(root/'source_hashes.json',{str(path.relative_to(source_root)):file_hash(path) for folder in ['src/molsteer','scripts'] for path in (source_root/folder).rglob('*.py')})


if __name__=='__main__':main()
