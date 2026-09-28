"""Explicit agent decision: retain a failed branch as evidence without more control."""
import argparse
import copy
import json
from pathlib import Path
import torch
from molsteer.common import write_json,file_hash
from molsteer.molexecutor.runner import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--label',required=True);p.add_argument('--output-label',required=True);a=p.parse_args()
    root=Path(a.root).resolve();records=json.loads((root/'all_runs.json').read_text())
    parent=next(r for r in records if r['label']==a.label)
    if any(r['label']==a.output_label for r in records):raise ValueError('Do not overwrite a recorded experiment')
    cp_path=Path(parent['directory'])/'creativity/resume_revision.pt'
    cp=torch.load(cp_path,weights_only=True,map_location='cpu');pending=cp['guidance_state']['pending_request']
    program=json.loads(Path(parent['config']['reward_programs']['creativity']).read_text())
    response=dict(request_id=pending['request_id'],parent_program_id=program['program_id'],program=program,resolution='stop_guidance',
        rationale='Independent structural risk persists and the cumulative injection budget is exhausted. A further coefficient change has no declared control capacity. Retain this branch as a failed/control-limit observation and stop additional guidance without resetting the checkpoint or budget.',
        validation_plan=['Continue native sampling only to observe the final consequence; this is not a successful rescue.',
            'Require independent final Reader assessment before accepting any generated structure.',
            'An earlier restart with a revised objective would be a separate experiment, not an extension with a silently reset budget.'])
    response_path=root/(a.output_label+'_response.json');write_json(response_path,response)
    cfg=copy.deepcopy(parent['config']);cfg.update(output=str(root/'continuations'/a.output_label),
        resume_checkpoint=str(cp_path),reward_revision_response=str(response_path),arms=['creativity'])
    summary=run(cfg)['executions'][0]
    record=dict(stage='t_0.50',label=a.output_label,strength=cfg['budget']['strength'],clear_initial_self_condition=False,
        directory=cfg['output'],arm='creativity',config=cfg,summary=summary,
        interpretation='Native tail after explicit stop; final quality remains to be assessed.')
    records.append(record);write_json(root/'all_runs.json',records)
    write_json(root/'runs.json',[r for r in records if r['summary'].get('status','complete')=='complete'])
    source_root=Path(__file__).resolve().parents[1]
    write_json(root/'delivered_source_hashes.json',{str(path.relative_to(source_root)):file_hash(path) for folder in ['src/molsteer','scripts'] for path in (source_root/folder).rglob('*.py')})
    print('STOP_RESULT '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
