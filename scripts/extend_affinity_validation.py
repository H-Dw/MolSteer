"""Declared adaptive follow-up: stronger structure weights at the same budget."""
import argparse
import copy
import json
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import load_config,parse_pdb
from molsteer.common import write_json,file_hash
from molsteer.molthinker.affinity import build_affinity_program
from molsteer.molexecutor.program import make_reward
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molmonitor.live_gradient import check_live_gradient


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();records=json.loads((root/'runs.json').read_text())
    config=copy.deepcopy(next(r for r in records if r['label']=='affinity_global_large')['config'])
    config.pop('categorical_proposal',None)
    previous=json.loads((root/'final_quality.json').read_text())
    write_json(root/'adaptive_followup.json',dict(reason='Larger affinity improvements increased relaxation strain; test stronger structural weights at the same enlarged guidance budget',
        initial_results_sha256=file_hash(root/'final_quality.json'),initial_labels=[r['label'] for r in previous],
        planned_structure_weights=[3.,10.,30.],same_rng=True,new_diagnosis=False,random_seed_sweep=False))
    adapter=FlowrRootAdapter(config);cp=torch.load(config['resume_checkpoint'],weights_only=True,map_location='cpu')
    baseline=torch.load(Path(config['reward_reference_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(config['receptor']);table=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=table.GetRvdw(atom['atomic_number'])
    spec=json.loads((root/'reasoning/t_0.50/RewardProgram.json').read_text())
    intent=json.loads((root/'DesignIntent.json').read_text());controls=json.loads((root/'control_trajectory.json').read_text())
    RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
    for weight in [3.,10.,30.]:
        label=f'structure_weight_{int(weight)}'
        if any(r['label']==label for r in records):continue
        directory=root/'continuations'/label
        if directory.exists():raise ValueError('Incomplete previous output requires a new directory')
        program=build_affinity_program(spec,intent,structure_weight=weight)
        path=root/'programs'/f'{label}.json';write_json(path,program)
        reward=make_reward(program,baseline,receptor,load_config(),controls['pkd'])
        adapter.restore(cp);reward.set_time(float(adapter.times[0][0]))
        check=check_live_gradient(adapter,reward);write_json(root/'validation'/f'{label}_gradient.json',check)
        if not check['passed']:raise ValueError('Derivative check failed: '+label)
        adapter.restore(cp)
        variant=copy.deepcopy(config);variant.update(output=str(directory),reward_programs=dict(creativity=str(path)))
        adapter.config=variant;write_json(directory/'execution.json',variant)
        summary=run_suffix(adapter,reward,directory/'creativity',GuidanceBudget(**variant['budget']),'creativity')
        records.append(dict(stage='t_0.50',label=label,strength=variant['budget']['strength'],clear_initial_self_condition=False,
            directory=str(directory),arm='creativity',config=variant,summary=summary))
        write_json(root/'runs.json',records);print('STRUCTURE_RUN '+label+' '+json.dumps(summary),flush=True)
    source_root=Path(__file__).resolve().parents[1]
    write_json(root/'adaptive_source_hashes.json',{str(p.relative_to(source_root)):file_hash(p)
        for folder in ['src/molsteer','scripts'] for p in (source_root/folder).rglob('*.py')})


if __name__=='__main__':main()
