"""Agent entry point: a declared adapter and JSON configuration produce a run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from rdkit import Chem, RDLogger
from molreader.io import load_config, parse_pdb
from .flowr import FlowrRootAdapter
from .program import MolecularReward,make_reward
from .interfaces import GuidanceBudget
from .engine import run_suffix


ADAPTERS={'flowr_root':FlowrRootAdapter}


def register_adapter(name, factory):
    if not name or name in ADAPTERS:raise ValueError('Duplicate/empty adapter name')
    ADAPTERS[name]=factory


def run(config):
    output=Path(config['output']).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory; experimental results cannot be silently overwritten')
    output.mkdir(parents=True,exist_ok=True)
    (output/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    adapter=ADAPTERS[config['adapter']](config)
    if config.get('resume_checkpoint'):
        checkpoint=torch.load(config['resume_checkpoint'],map_location='cpu',weights_only=True)
        if config.get('reward_revision_response'):
            from molsteer.molthinker.feedback import apply_revision_response
            response=json.loads(Path(config['reward_revision_response']).read_text())
            checkpoint,new_program=apply_revision_response(checkpoint,response)
            path=output/'revised_program.json';path.write_text(json.dumps(new_program,indent=2),encoding='utf-8')
            config['reward_programs']={'creativity':str(path)}
            config['arms']=['creativity'];config['gradient_preflight']=response.get('resolution')!='stop_guidance'
        adapter.restore(checkpoint)
        replay=dict(restored_checkpoint=config['resume_checkpoint'],step_index=adapter.step_index)
    else:
        replay=adapter.replay_to(config.get('start_step',50))
        checkpoint=adapter.checkpoint()
    torch.save(checkpoint,output/'resume_start.pt')
    (output/'resume_verification.json').write_text(json.dumps(replay,indent=2),encoding='utf-8')
    print('RESUME_VERIFICATION '+json.dumps(replay),flush=True)
    if config.get('prepare_only'):return replay
    receptor,_=parse_pdb(config['receptor'])
    table=Chem.GetPeriodicTable()
    for a in receptor:a['vdw_radius']=table.GetRvdw(a['atomic_number'])
    vocabulary=load_config()
    # Keep the proposal's X0 and p0 fixed to the original supplied prediction,
    # including when a complete runtime checkpoint is resumed at a later time.
    saved_baseline=torch.load(Path(config.get('reward_reference_stage',config['saved_stage']))/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in saved_baseline.items() if torch.is_tensor(v)}
    controls=json.loads(Path(config['control_trajectory']).read_text()) if config.get('control_trajectory') else {}
    def evaluator(program):
        return make_reward(program,baseline,receptor,vocabulary,controls.get(program.get('affinity_head')))
    if not config.get('gradient_preflight',True):
        for path in config['reward_programs'].values():
            if json.loads(Path(path).read_text(encoding='utf-8')).get('evaluator')=='agent_expert':
                raise ValueError('Expert live programs require component-gradient preflight')
    if config.get('gradient_preflight',True):
        from molsteer.molmonitor.live_gradient import check_live_gradient
        preflight={}
        for mode in config['reward_programs']:
            program=json.loads(Path(config['reward_programs'][mode]).read_text(encoding='utf-8'))
            reward=evaluator(program)
            if (program.get('evaluator')=='augmented_lagrangian' and adapter.guidance_state
                    and adapter.guidance_state.get('augmented_lagrangian')):
                reward.restore_controller(adapter.guidance_state['augmented_lagrangian'])
            if hasattr(reward,'set_time'):reward.set_time(float(adapter.times[0][0]))
            preflight[mode]=check_live_gradient(adapter,reward)
        (output/'gradient_preflight.json').write_text(json.dumps(preflight,indent=2),encoding='utf-8')
        if not all(r['passed'] for r in preflight.values()):raise ValueError('Live gradient preflight failed; see gradient_preflight.json')
        print('GRADIENT_PREFLIGHT_PASSED',flush=True)
    summaries=[]
    RDLogger.DisableLog('rdApp.warning')
    for arm in config.get('arms',['unguided','selection','creativity']):
        adapter.restore(checkpoint)
        program_key=arm if arm!='unguided' else next(iter(config['reward_programs']))
        program=json.loads(Path(config['reward_programs'][program_key]).read_text(encoding='utf-8'))
        reward=evaluator(program)
        armout=output/arm
        armout.mkdir(parents=True,exist_ok=True)
        # Let MolReader recover receptor provenance using the original input layout.
        shutil.copytree(Path(config['saved_stage']).parents[2]/'inputs',armout/'inputs',dirs_exist_ok=True)
        summary=run_suffix(adapter,reward,armout,GuidanceBudget(**config.get('budget',{})),arm)
        summaries.append(summary)
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(config['stage_runner'])]}
    report=dict(replay=replay,executions=summaries,source_hashes=source_hashes,
        discrete_graph_preference='Frozen categorical marginal ranking regularizer; zero coordinate derivative; used in proposal acceptance',
        uncalibrated_parameters=True)
    (output/'experiment_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def emit(config_path, destination):
    """Emit a portable PyTorch entry script and an immutable JSON input copy."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    (destination/'execution.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    script='''#!/usr/bin/env python3
from pathlib import Path
import json
from molsteer.molexecutor.runner import run
if __name__ == "__main__":
    run(json.loads(Path(__file__).with_name("execution.json").read_text(encoding="utf-8")))
'''
    (destination/'run_guidance.py').write_text(script,encoding='utf-8')
    return destination/'run_guidance.py'


def main():
    p=argparse.ArgumentParser(description='MolExecutor live sampler execution')
    p.add_argument('--config',required=True)
    p.add_argument('--emit')
    p.add_argument('--output',help='Fresh output directory override')
    p.add_argument('--resume',help='Complete runtime checkpoint override')
    p.add_argument('--revision-response',help='Validated response to a pending monitor revision request')
    p.add_argument('--arm',choices=['unguided','selection','creativity','agent'],action='append')
    a=p.parse_args()
    config=json.loads(Path(a.config).read_text(encoding='utf-8'))
    if a.output:config['output']=a.output
    if a.resume:config['resume_checkpoint']=a.resume
    if a.revision_response:config['reward_revision_response']=a.revision_response
    if a.arm:config['arms']=a.arm
    if a.emit:
        directory=Path(a.emit);directory.mkdir(parents=True,exist_ok=True)
        path=directory/'execution.json';path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        print(emit(path,directory))
    else:run(config)


if __name__=='__main__':main()
