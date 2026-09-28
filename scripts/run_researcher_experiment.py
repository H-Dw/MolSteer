"""Full matched suffix tests, optional-mode replay and exact research restart."""
import argparse,copy,json
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import load_config,parse_pdb
from molsteer.common import write_json,digest,file_hash
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import make_reward
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.interfaces import GuidanceBudget
from run_exact_continuations import unequal


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--previous-rebalanced',required=True);ap.add_argument('--continue-completed',action='store_true');a=ap.parse_args()
    root=Path(a.root).resolve();cfg=json.loads((root/'base_execution.json').read_text())
    if (root/'runs.json').exists() and not a.continue_completed:raise ValueError('Fresh run directory required, or explicitly continue verified completed arms')
    adapter=FlowrRootAdapter(cfg);cp=torch.load(cfg['resume_checkpoint'],weights_only=True,map_location='cpu')
    baseline=torch.load(Path(cfg['saved_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    rec,_=parse_pdb(cfg['receptor']);periodic=Chem.GetPeriodicTable()
    for atom in rec:atom['vdw_radius']=periodic.GetRvdw(atom['atomic_number'])
    controls=json.loads(Path(cfg['control_trajectory']).read_text());vocab=load_config();records=json.loads((root/'runs.json').read_text()) if a.continue_completed else [];finals={}
    RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
    plans=json.loads((root/'planned_runs.json').read_text())
    def execute(label,spec,strength,checkpoint):
        config=copy.deepcopy(cfg);config['output']=str(root/'continuations'/label);config['budget']=dict(strength=strength,max_step_angstrom=.04,max_path_angstrom=2.)
        config['reward_programs']=dict(creativity=str(root/'programs'/f'{label}.json'));config.pop('monitor',None)
        adapter.config=config;adapter.restore(checkpoint)
        reward=make_reward(spec,baseline,rec,vocab,controls[spec['affinity_head']])
        write_json(Path(config['output'])/'execution.json',config)
        result=run_suffix(adapter,reward,Path(config['output'])/'creativity',GuidanceBudget(**config['budget']),'creativity')
        return result,adapter.checkpoint()
    for plan in plans:
        label=plan['label'];spec=json.loads((root/'programs'/f'{label}.json').read_text())
        completed=next((r for r in records if r['label']==label),None)
        if completed:
            if completed['summary']['initial_program_id']!=spec['program_id']:raise ValueError('Cannot continue after changing a completed reward')
            finals[label]=torch.load(root/'continuations'/label/'creativity/resume_final.pt',weights_only=True,map_location='cpu')
            continue
        result,final=execute(label,spec,plan['eta_coordinates'],cp);finals[label]=final
        records.append(dict(label=label,summary=result));write_json(root/'runs.json',records)
        print('COMPLETE '+label,flush=True)
    old=torch.load(Path(a.previous_rebalanced),weights_only=True,map_location='cpu')
    diffs={k:unequal(old[k],finals['research_off'][k]) for k in ('curr','cond','times','rng')}
    write_json(root/'validation/off_exact.json',dict(all_exact=not any(diffs.values()),differences=diffs))
    if any(diffs.values()):raise ValueError('Disabled research changed the established comparator')
    # Shadow leaves execution untouched; no network is called by the executor.
    active=json.loads((root/'programs/research_dynamic.json').read_text());shadow=copy.deepcopy(active)
    shadow.pop('program_id');shadow['research']['mode']='shadow';shadow['program_id']='rp_'+digest(shadow)[:24]
    _,shadow_final=execute('shadow_validation',shadow,300.,cp)
    shadow_diff={k:unequal(finals['research_off'][k],shadow_final[k]) for k in ('curr','cond','times','rng')}
    write_json(root/'validation/shadow_exact.json',dict(all_exact=not any(shadow_diff.values()),differences=shadow_diff))
    if any(shadow_diff.values()):raise ValueError('Shadow research changed sampling')
    # Resume preserves SC/RNG, weights, eta, evidence history, KL and MMFF references.
    restart=torch.load(root/'continuations/research_dynamic/creativity/resume_t_0.75.pt',weights_only=True,map_location='cpu')
    _,resumed=execute('restart_validation',active,300.,restart)
    differences={k:unequal(finals['research_dynamic'][k],resumed[k]) for k in ('curr','cond','times','rng','guidance_state')}
    write_json(root/'validation/restart_exact.json',dict(all_exact=not any(differences.values()),differences=differences))
    if any(differences.values()):raise ValueError('Research dynamic restart mismatch')
    untouched={label:all(torch.equal(f['curr'][k][:2],finals['research_off']['curr'][k][:2]) for k in ('coords','atomics','bonds','charges')) for label,f in finals.items()}
    write_json(root/'validation/batch_isolation.json',dict(all_exact=all(untouched.values()),arms=untouched))
    if not all(untouched.values()):raise ValueError('Non-target batch members changed')
    write_json(root/'execution_provenance.json',dict(origin_runtime=cfg['resume_checkpoint'],origin_runtime_sha256=file_hash(cfg['resume_checkpoint']),
        model_checkpoint_sha256=cp['model_checkpoint_sha256'],sampler_seed_sweep=False,all_validation_passed=True,
        source_hashes={str(p.relative_to(Path(__file__).resolve().parents[1])):file_hash(p) for p in (Path(__file__).resolve().parents[1]/'src').rglob('*.py')}))
    print('ALL LIVE TESTS PASSED',flush=True)


if __name__=='__main__':main()
