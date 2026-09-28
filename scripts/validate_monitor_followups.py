"""Agent-authored reward revision and changed-graph coverage validation."""
import argparse
import copy
import json
from pathlib import Path
import torch
from molsteer.common import digest,write_json,file_hash
from molsteer.molexecutor.runner import run
from molsteer.molthinker.feedback import apply_revision_response,prepare_revision_context
from run_exact_continuations import unequal


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();records=json.loads((root/'all_runs.json').read_text())
    source=next(r for r in records if r['label']=='stress_monitored')
    cp_path=Path(source['directory'])/'creativity/resume_revision.pt'
    checkpoint=torch.load(cp_path,weights_only=True,map_location='cpu')
    pending=checkpoint['guidance_state']['pending_request']
    old=json.loads(Path(source['config']['reward_programs']['creativity']).read_text())
    revised=copy.deepcopy(old)
    revised.update(parent_program_id=old['program_id'],affinity_weight=1.,structure_weight=30.,
        design_intent='Respond to independent strain worsening while the optimized affinity reward improves; increase global structural pressure without changing guards or restoring the original graph.')
    revised.pop('program_id');revised['program_id']='rp_'+digest(revised)[:24]
    response=dict(request_id=pending['request_id'],parent_program_id=old['program_id'],program=revised,
        author='reasoning agent in the current task; explicit response, not a hidden language-model call',
        rationale=dict(observed=pending['observations']['independent_sentinel'],
            interpretation='Persistent affinity/independent-strain disagreement is evidence against relying on strength adaptation alone. It does not prove a specific physical mechanism.',
            change='Reduce affinity weight 3 to 1 and increase structure weight 1 to 30. Preserve geometry units, hard constraints, graph-change permission and the original t=0.50 evidence binding.',
            knowledge_map=[dict(function_id='P01',use='Independent same-graph force-field strain validation; not an implemented force-field autograd term.'),
                dict(function_id='G01',use='Graph-conditioned geometry residuals with existing global soft curvature; stronger structural pressure.'),
                dict(function_id='S05',use='Check the affinity/structure tradeoff explicitly; no population Pareto selection is claimed.')],
            limitation='Remaining cumulative displacement is small. This response may be too late to repair the trajectory; the geometry proxy omits force-field torsion and nonbonded contributions.'),
        validation_plan=['Preserve curr, cond, RNG, times and cumulative path budget exactly at handoff.',
            'Run live gradient finite-difference preflight for the revised reward.',
            'Continue only within the inherited budget and record a further revision request if independent checks remain adverse.',
            'Evaluate any completed native final by MolReader, all four predicted affinity heads, graph identity, strain, clashes and PoseBusters.'])
    response_path=root/'revision_response.json';write_json(response_path,response)
    transferred,_=apply_revision_response(checkpoint,response)
    diff={k:unequal(checkpoint[k],transferred[k]) for k in ['curr','cond','rng','times']}
    diff['path_used']=unequal(checkpoint['guidance_state']['path_used'],transferred['guidance_state']['path_used'])
    write_json(root/'revision_transfer_exact.json',dict(all_exact=not any(diff.values()),differences=diff))
    if any(diff.values()):raise ValueError('Revision handoff changed physical runtime state')
    context=prepare_revision_context(pending,old,source['config']['monitor']['knowledge_path'])
    write_json(root/'revision_context_compact.json',context)
    plans=[]
    cfg=copy.deepcopy(source['config']);cfg.update(output=str(root/'continuations/stress_revised'),
        resume_checkpoint=str(cp_path),reward_revision_response=str(response_path),arms=['creativity'])
    plans.append(('stress_revised',cfg))
    wide=next(r for r in records if r['label']=='adaptive_wide')
    cfg=copy.deepcopy(wide['config']);cfg.update(output=str(root/'continuations/adaptive_wide_review'),arms=['creativity'])
    plans.append(('adaptive_wide_review',cfg))
    for label,cfg in plans:
        result=run(cfg);summary=result['executions'][0]
        record=dict(stage='t_0.50',label=label,strength=cfg['budget']['strength'],clear_initial_self_condition=False,
            directory=cfg['output'],arm='creativity',config=cfg,summary=summary)
        records.append(record);write_json(root/'all_runs.json',records)
        write_json(root/'runs.json',[r for r in records if r['summary'].get('status','complete')=='complete'])
        print('FOLLOWUP_RUN '+label+' '+json.dumps(summary),flush=True)
    source_root=Path(__file__).resolve().parents[1]
    write_json(root/'followup_source_hashes.json',{str(path.relative_to(source_root)):file_hash(path) for folder in ['src/molsteer','scripts'] for path in (source_root/folder).rglob('*.py')})


if __name__=='__main__':main()
