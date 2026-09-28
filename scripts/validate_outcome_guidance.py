"""Matched tests of outcome-aware continuous, discrete and feedback control."""
import argparse,copy,json
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import load_config,parse_pdb
from molsteer.common import write_json,file_hash
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import make_reward
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molthinker.outcomes import build_outcome_program,render_outcome_program
from molsteer.molmonitor.live_gradient import check_live_gradient
from run_exact_continuations import unequal


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--evaluation',required=True);a=p.parse_args()
    root=Path(a.root).resolve();evaluation=Path(a.evaluation).resolve();cfg=json.loads((root/'base_execution.json').read_text())
    if (root/'planned_runs.json').exists():raise ValueError('Do not overwrite a completed/partial experiment')
    packet=json.loads((evaluation/'StatePacket.with_outcomes.json').read_text());report=json.loads((evaluation/'provenance/t050_DiagnosticReport.json').read_text())
    old=json.loads((evaluation/'provenance/active_RewardProgram.json').read_text())
    knowledge=Path(__file__).resolve().parents[1]/'knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md'
    designs=[('control',old),('legacy_fixed300',old)]
    for label,discrete,feedback in [('outcome_continuous',False,False),('outcome_discrete',True,False),('outcome_adaptive',True,True)]:
        spec=build_outcome_program(old,packet,report,knowledge,root/'native_reference.json',discrete=discrete,feedback=feedback)
        designs.append((label,spec))
        for lang in ('en','zh'):path=root/'programs'/f'{label}.{lang}.md';path.parent.mkdir(exist_ok=True);path.write_text(render_outcome_program(spec,lang),encoding='utf-8')
    write_json(root/'planned_runs.json',[dict(label=label,program_id=spec['program_id'],eta=0 if label=='control' else 300.,
        max_step_angstrom=.04,max_path_angstrom=2.,discrete=spec.get('discrete_search'),feedback=spec.get('feedback_policy')) for label,spec in designs])
    adapter=FlowrRootAdapter(cfg);cp=torch.load(cfg['resume_checkpoint'],weights_only=True,map_location='cpu')
    baseline=torch.load(Path(cfg['saved_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(cfg['receptor']);pt=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=pt.GetRvdw(atom['atomic_number'])
    controls=json.loads(Path(cfg['control_trajectory']).read_text());vocab=load_config();records=[];checked=False
    RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
    for label,spec in designs:
        path=root/'programs'/f'{label}.json';write_json(path,spec)
        variant=copy.deepcopy(cfg);variant.pop('monitor',None)
        variant.update(output=str(root/'continuations'/label),reward_programs=dict(creativity=str(path)),budget=dict(strength=300.,max_step_angstrom=.04,max_path_angstrom=2.))
        adapter.config=variant;adapter.restore(cp)
        reward=make_reward(spec,baseline,receptor,vocab,controls[spec['affinity_head']])
        if spec.get('evaluator')=='outcome_aware' and not checked:
            reward.set_time(float(adapter.times[0][0]));check=check_live_gradient(adapter,reward)
            write_json(root/'validation/continuous_gradient.json',check)
            if not check['passed']:raise ValueError('Outcome live gradient preflight failed')
            checked=True;adapter.restore(cp)
        arm='unguided' if label=='control' else 'creativity'
        write_json(Path(variant['output'])/'execution.json',variant)
        result=run_suffix(adapter,reward,Path(variant['output'])/arm,GuidanceBudget(**variant['budget']),arm)
        record=dict(label=label,stage='t_0.50',strength=0. if label=='control' else 300.,clear_initial_self_condition=False,
            directory=variant['output'],arm=arm,config=variant,summary=result)
        records.append(record);write_json(root/'runs.json',records)
        if label in ('control','legacy_fixed300'):
            reference=Path(cfg['saved_stage']).parent/'final/runtime.pt' if label=='control' else Path(cfg['model_root'])/'output/molsteer_monitor_20260923/continuations/fixed_300/creativity/resume_final.pt'
            expected=torch.load(reference,weights_only=True,map_location='cpu');actual=adapter.checkpoint()
            differences={k:unequal(expected[k],actual[k]) for k in ('curr','cond','times','rng')}
            write_json(root/'validation'/f'{label}_exact.json',dict(all_exact=not any(differences.values()),differences=differences))
            if any(differences.values()):raise ValueError('Existing execution changed: '+label)
        print('RUN '+label+' '+json.dumps(result),flush=True)
    write_json(root/'provenance.json',dict(origin_runtime=cfg['resume_checkpoint'],origin_runtime_sha256=file_hash(cfg['resume_checkpoint']),
        baseline_reference_sha256=file_hash(root/'native_reference.json'),model_checkpoint=cfg['checkpoint'],model_checkpoint_sha256=cp['model_checkpoint_sha256'],
        seed_policy='Restore identical original full runtime for every arm; no seed sweep',
        source_hashes={str(p.relative_to(Path(__file__).resolve().parents[1])):file_hash(p) for p in (Path(__file__).resolve().parents[1]/'src/molsteer').rglob('*.py')}))


if __name__=='__main__':main()
