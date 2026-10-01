"""Real GLM expert design and Executor tests on a verified saved 5i0b state.

This phase does not bind a live generator, certify its Jacobian or run continuations.
An optional design review retains scientifically necessary unresolved goals.
"""
import argparse
import json
from pathlib import Path
import shutil
from test_5i0b_api_weights import REPO, WEIGHTS, observed_api_runtime, audit_sources, write, sha
from test_glm53_expert_design import prepare
from molsteer.agents.config import load_config
from molsteer.agents.trace import load_checkpoint
from molsteer.common import digest
from molsteer.molexecutor.agent_bridge import compile_validated_agent_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-test-root',type=Path,default=REPO/'test/5i0b_t050_api_weights_20260930')
    parser.add_argument('--source-attempt',default='run_06')
    parser.add_argument('--test-root',type=Path,default=REPO/'test/glm_design_audit_20260930')
    parser.add_argument('--attempt',default='glm_design_01')
    parser.add_argument('--agent-config',type=Path,default=REPO/'configs/agents.json')
    parser.add_argument('--allow-design-only',action='store_true',
        help='Retain a real-API scientific design review with unresolved execution inputs; never mark it executable.')
    args=parser.parse_args()
    for name in (args.source_attempt,args.attempt):
        if Path(name).name!=name or name in ('','.','..'):
            raise ValueError('Use one nonempty experiment directory name')
    config=load_config(args.agent_config)
    if (config.mode!='api' or {p.model for p in config.models.values()}!={'z-ai/glm-5.3'}
        or not config.thinker.require_design_audit):
        raise ValueError('Actual GLM-5.3 API and scientific design audits are required')
    root=prepare(args.source_test_root,args.test_root)
    source=(args.source_test_root/args.source_attempt).resolve()
    source_checkpoint=source/'agent_runs'/('api_5i0b_'+args.source_attempt+'.checkpoint.json')
    previous=load_checkpoint(source_checkpoint)['artifacts']
    packet=previous['packet']
    if (packet['identity']['stage_t']!=.5 or '5i0b' not in packet['identity']['target_id'].split('_')
        or packet['identity']['ligand_id']!='ligand_000'):
        raise ValueError('Expected copied 5i0b molecule 000 at t=.50')
    if previous['reward_spec']['packet_id']!=packet['packet_id']:
        raise ValueError('Saved API state is not bound to its original reward')
    output=root/args.attempt;output.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),output/'test_harness.py')
    write(output/'StatePacket.json',packet)
    write(output/'saved_state_provenance.json',dict(source=str(source_checkpoint),
        source_sha256=sha(source_checkpoint),packet_sha256=digest(packet),
        phase='saved_state_api_design',live_generator_derivative='not_run',
        weights={str(w):{'status':'not_run'} for w in WEIGHTS}))
    dynamics=dict(previous['model_dynamics'],live_derivative='unavailable',
        source='Previously verified FLOWR mapping; current saved-state API phase has no live generator')
    config.agents['molmonitor'].enabled=False;config.monitoring.graph_review_enabled=False
    config.runtime.trace_dir=(output/'agent_runs').relative_to(REPO)
    write(output/'agents.api.json',config.model_dump(mode='json'))
    try:
        result=observed_api_runtime(config,dynamics,output/'api_requests.jsonl').run(
            packet,run_id='api_5i0b_'+args.attempt,execute=False,feedback={
                'kind':'UserTaskContext','monitor_enabled':False,'graph_review_enabled':False,
                'task':'Re-design the copied 5i0b molecule at t=.50 using all measured biochemical factors and reviewed mathematical functions. Validate on coordinate copies for later matched weights 1,10,100,300,500.',
                'scope':'Real API design and optional Executor coordinate-copy phase only; no live generator is bound here. Live pullback and continuation are not_run. An explicitly bounded chemical-hypothesis pilot can be numerically tested; do not certify live guidance or terminal binding utility.',
                'requirements':'Consider bond/angle/torsion geometry, whole stability, target contacts/area and functional-group suitability. Missing inputs remain explicit. Screening cutoffs are not calibrated physical tolerances.'})
        write(output/'api_status.json',dict(status=result['status'],mode=config.mode,
            phase='saved_state_api_design',errors=result.get('errors',[]),
            checkpoint_path=result['checkpoint_path'],monitor_enabled=False,graph_review_enabled=False,
            live_generator_derivative='not_run',trace_nodes=sorted({e['node'] for e in result['trace']})))
        if result['status']=='design_deferred' and args.allow_design_only:
            write(output/'ScientificDesign.deferred.json',dict(
                biology_plan=result.get('biology_plan'),mathematical_design=result.get('mathematical_design'),
                workspaces=result.get('expert_workspace_history',[]),reason=result.get('reward_design_deferral'),
                executable=False,live_generator_derivative='not_run'))
            print(json.dumps({'status':'design_deferred','phase':'saved_state_api_design','output':str(output)}),flush=True)
            return
        if result['status']!='validated':
            raise RuntimeError('Real GLM expert design did not validate; no fallback')
        guard=json.loads((REPO/'experiments/guidance/creativity.json').read_text())
        program,*_=compile_validated_agent_checkpoint(result['checkpoint_path'],guard)
        write(output/'RewardProgram.api.json',program)
        print(json.dumps({'status':'validated','phase':'saved_state_api_design','output':str(output)}),flush=True)
    finally:
        if not audit_sources(root,'after')['passed']:
            raise ValueError('Source integrity changed during the API design phase')


if __name__=='__main__':main()
