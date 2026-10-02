"""Real API reward design and matched t=.50 FLOWR continuations, MolMonitor off.

Only reads the copied test sources. No offline reward fallback or data writes.
Run in the CUDA environment recorded by the selected trajectory.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
WEIGHTS = (1, 10, 100, 300, 500)


def normalized_weights(values):
    """Keep the requested matched experiment arms explicit and reproducible."""
    weights = tuple(int(v) if float(v).is_integer() else float(v) for v in values)
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Weights must be finite positive numbers')
    if len(set(weights)) != len(weights):
        raise ValueError('Each weight must identify one unique experiment arm')
    return weights


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def audit_sources(root, phase):
    """Check the source data and copies without opening any source for writing."""
    manifest = json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    expected = manifest['data_files_before']
    data = REPO/'data'
    observed = {}
    for path in sorted(data.rglob('*')):
        if path.is_file():
            stat = path.stat()
            observed[path.relative_to(data).as_posix()] = dict(
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(path))
    changed = sorted(key for key in set(expected)|set(observed)
                     if expected.get(key) != observed.get(key))
    sources = []
    for record in manifest['source_files']:
        source = Path(record['source'])
        subdir = 'source_trajectory' if Path(record['copy']).suffix == '.pt' else 'inputs'
        copy = root/subdir/Path(record['copy']).name
        sources.append(dict(source=str(source), copy=str(copy),
            source_unchanged=source.is_file() and sha(source)==record['sha256'],
            copy_unchanged=copy.is_file() and sha(copy)==record['sha256']))
    result = dict(phase=phase, data_unchanged=not changed,
                  changed_data_files=changed, data_file_count=len(observed), sources=sources)
    result['passed'] = result['data_unchanged'] and all(
        row['source_unchanged'] and row['copy_unchanged'] for row in sources)
    write(root/f'source_integrity_{phase}.json', result)
    return result


def observed_api_runtime(config, dynamics, receipt_path):
    """Record real request counts and token usage, without prompts or responses."""
    from langchain_core.callbacks import BaseCallbackHandler
    from molsteer.agents.runtime import AgentRuntime

    class Receipts(BaseCallbackHandler):
        def __init__(self, role):
            self.role = role
            self.started = {}

        def record(self, event):
            with receipt_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(agent=self.role, **event))+'\n')

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            self.started[str(run_id)] = time.perf_counter()
            self.record(dict(event='request_started', request_id=str(run_id)))

        def on_llm_end(self, response, *, run_id, **kwargs):
            usage = []
            finish_reasons = []
            served_models = []
            for group in response.generations:
                for generation in group:
                    metadata = getattr(generation.message, 'usage_metadata', None) or {}
                    usage.append({key:metadata[key] for key in
                                  ('input_tokens','output_tokens','total_tokens') if key in metadata})
                    reasoning_tokens=metadata.get('output_token_details',{}).get('reasoning')
                    if reasoning_tokens is not None:usage[-1]['reasoning_tokens']=reasoning_tokens
                    cached_tokens=metadata.get('input_token_details',{}).get('cache_read')
                    if cached_tokens is not None:usage[-1]['cached_tokens']=cached_tokens
                    receipt=(getattr(generation.message,'additional_kwargs',None) or {}).get('_openrouter_receipt',{})
                    if receipt:usage[-1]['provider_receipt']=receipt
                    finish_reasons.append((getattr(generation.message,'response_metadata',None) or {}).get('finish_reason'))
                    served_models.append((getattr(generation.message,'response_metadata',None) or {}).get('model_name'))
            self.record(dict(event='request_completed', request_id=str(run_id), usage=usage,
                finish_reasons=finish_reasons,
                served_models=served_models,
                wall_seconds=time.perf_counter()-self.started.pop(str(run_id),time.perf_counter())))

        def on_llm_error(self, error, *, run_id, **kwargs):
            frames=[];traceback=error.__traceback__
            while traceback:
                frames.append(traceback.tb_frame.f_code.co_name)
                traceback=traceback.tb_next
            decoder = ({'position':error.pos,'body_character_count':len(error.doc),
                        'line':error.lineno,'column':error.colno} if isinstance(error,json.JSONDecodeError) else {})
            self.record(dict(event='request_failed', request_id=str(run_id),
                error_type=type(error).__name__, status_code=getattr(error,'status_code',None),
                stream_receipt=getattr(error,'stream_receipt',{}),
                decoder=decoder, frame_names=frames[-6:],
                wall_seconds=time.perf_counter()-self.started.pop(str(run_id),time.perf_counter())))

    class ObservedRuntime(AgentRuntime):
        def run(self, *args, **kwargs):
            from molsteer.agents import loop
            original = loop.append_trace
            def record_tool(state, **event):
                result = original(state, **event)
                # append_trace already filters the observation through the
                # normal audit redactor; provider messages never enter this file.
                with receipt_path.with_name('api_tools.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(state['trace'][-1],allow_nan=False)+'\n')
                return result
            loop.append_trace = record_tool
            try:
                return super().run(*args, **kwargs)
            finally:
                loop.append_trace = original

        def _model(self, name):
            is_new = name not in self.models
            model = super()._model(name)
            if is_new:
                model.callbacks = [*(model.callbacks or []), Receipts(name)]
            return model

    return ObservedRuntime(config, model_dynamics=dynamics)


def preflight(root, agent_file, weights=WEIGHTS):
    manifest = json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    saved = json.loads((root/'generation_provenance.json').read_text(encoding='utf-8'))
    config = json.loads(agent_file.read_text(encoding='utf-8'))
    blockers = []
    if not audit_sources(root, 'before')['passed']:
        blockers.append('source_data_or_copy_changed')
    modules = {n:importlib.util.find_spec(n) is not None for n in
               ('torch', 'rdkit', 'langgraph', 'langchain_core', 'langchain_openrouter')}
    if not all(modules.values()):blockers.append('missing_runtime_dependencies')
    credentials = {}
    for name in ('molreader', 'molthinker', 'molexecutor'):
        profile = config['models'][config['agents'][name]['model']]
        provider = config['providers'][profile['provider']]
        credentials[name] = bool(os.getenv(provider['api_key_env']))
    environment_credentials = dict(credentials)
    from molsteer.agents.config import load_config as load_agents, SecretStore
    resolved_config = load_agents(agent_file)
    store = SecretStore(resolved_config)
    for name in credentials:
        profile = resolved_config.models[resolved_config.agents[name].model]
        try:
            credentials[name] = bool(store.get(profile.provider))
        except ValueError:
            credentials[name] = False
    if not all(credentials.values()):
        blockers.append('missing_api_credentials')
    torch_state = {}
    if modules['torch']:
        try:
            import torch
            torch_state = dict(version=str(torch.__version__), gpu_count=torch.cuda.device_count(),
                               gpu_available=torch.cuda.is_available())
            if not torch_state['gpu_available']:blockers.append('no_visible_gpu_for_saved_cuda_rng')
            if str(torch.__version__) != saved['environment']['torch']:
                blockers.append('torch_version_differs_from_recorded_trajectory')
            if torch.cuda.device_count() != len(saved['environment']['gpu']):
                blockers.append('visible_gpu_layout_differs_from_recorded_trajectory')
        except Exception as exc:
            torch_state = dict(import_error_type=type(exc).__name__)
            blockers.append('torch_runtime_unavailable')
    copies = {}
    for record in manifest['source_files']:
        subdir = 'source_trajectory' if Path(record['copy']).suffix == '.pt' else 'inputs'
        path = root/subdir/Path(record['copy']).name
        copies[path.relative_to(root).as_posix()] = path.is_file() and sha(path)==record['sha256']
    if not all(copies.values()):blockers.append('copied_input_hash_mismatch')
    if config['mode'] != 'api':blockers.append('api_mode_required')
    result = dict(status='blocked' if blockers else 'ready', blockers=blockers,
                  modules=modules, credentials_present=credentials,
                  environment_credentials_present=environment_credentials, runtime=torch_state,
                  recorded_torch=saved['environment']['torch'], source_hashes_match=copies,
                  monitor_enabled=False, graph_review_enabled=False,
                  weights={str(w):{'status':'not_run'} for w in weights})
    write(root/'readiness.json', result)
    print(json.dumps(result), flush=True)
    return result


def live_adapter(root, output):
    import torch
    from molsteer.preprocessing.checkpoints import load_bundle, configure_numerics, source_hashes, validate_sources
    from molsteer.preprocessing.flowr_dataset import import_runner
    from molsteer.preprocessing.trajectory import Trajectory
    from molsteer.molexecutor.flowr import FlowrRootAdapter
    bundle = load_bundle(root/'source_trajectory/molecule_000.pt')
    if bundle['molecule_index']!=0 or bundle['batch_index']!=0 or bundle['verification']['status']!='passed':
        raise ValueError('Expected the verified copied molecule 000')
    stage_runner = REPO/'integrations/flowr_root/stage_runner.py'
    model_root = REPO/'flowr_root'
    validate_sources(bundle['provenance']['sources'], source_hashes(model_root,stage_runner))
    checkpoint = model_root/'checkpoints/flowr_root_v2.2.ckpt'
    if sha(checkpoint)!=bundle['provenance']['model_sha256']:
        raise ValueError('Model checkpoint does not match the copied trajectory')
    device = torch.device('cuda:0');configure_numerics(device)
    runner = import_runner(model_root,stage_runner)
    args = SimpleNamespace(**bundle['generator_args'])
    args.ckpt_path = str(checkpoint);args.save_dir = str(output)
    model,hparams,*_ = runner.load_model(args)
    model = model.to(device).eval().requires_grad_(False)
    trajectory = Trajectory.restore(model,bundle['shared'],bundle['checkpoints']['0.50'],device)
    a = FlowrRootAdapter.__new__(FlowrRootAdapter)
    a.runner=runner;a.model=model;a.hparams=hparams;a.args=args;a.device=device
    a.source_hash=sha(stage_runner);a.precision='highest';a.batch=bundle['shared']['batch_size']
    a.index=bundle['batch_index'];a.guidance_state={}
    for name in ('prior','pocket','curr','cond','times','equis','invs','grid','step_index'):
        setattr(a,name,getattr(trajectory,name))
    a.pocket['complex']=[SimpleNamespace(com=c) for c in trajectory.com]
    a.resume_fidelity='Exact copied full batch, original self-conditioning, categories and saved RNG'
    inputs=root/'inputs'
    a.config=dict(model_root=str(model_root),checkpoint=str(checkpoint),stage_runner=str(stage_runner),
        receptor=str(inputs/Path(bundle['target']['receptor']).name),
        reference_ligand=str(inputs/Path(bundle['target']['reference_ligand']).name),
        target_id=bundle['target']['target_id'],ligand_index=0,gpu=0,output=str(output),
        saved_stage=str(output/'capture'/bundle['target']['target_id']/'ligand_000/t_0.50'),
        monitor={'enabled':False,'graph_review':{'enabled':False}},
        guidance_interval=[.5,1.],record_tensor_trace=True)
    return a,bundle


def execute(root, agent_file, attempt, weights=WEIGHTS):
    import torch
    from molsteer.common import digest
    from molsteer.preprocessing.checkpoints import assert_equal, cpu_copy
    from molsteer.molexecutor.flowr import snapshot_rng
    from molsteer.molexecutor.program import make_reward
    from molsteer.molexecutor.runner import run
    from molsteer.molexecutor.engine import run_suffix
    from molsteer.molexecutor.interfaces import GuidanceBudget
    from molsteer.molmonitor.live_gradient import check_live_gradient
    from molreader.io import load_stage,load_config,parse_pdb
    from molreader.packet import build_packet
    from molreader.localized_report import make_localized_report
    from molsteer.molreader import enrich_packet
    from molsteer.molreader.reporting import render_diagnostic
    from molsteer.agents.config import load_config as agent_config
    from molsteer.molexecutor.agent_bridge import compile_validated_agent_checkpoint
    from rdkit import Chem
    output=root/attempt
    output.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),output/'test_harness.py')
    adapter,bundle=live_adapter(root,output)
    initial=adapter.checkpoint()
    adapter.save_stage(output/'capture','t_0.50',.5)
    stage=Path(adapter.config['saved_stage'])
    torch.save(initial,stage/'runtime.pt')
    contexts=[load_stage(stage,view,receptor=adapter.config['receptor']) for view in ('state','prediction')]
    packet=enrich_packet(build_packet(contexts),contexts)
    report=make_localized_report(packet)
    write(output/'StatePacket.json',packet);write(output/'DiagnosticReport.json',report)
    (output/'DiagnosticReport.zh.md').write_text(render_diagnostic(report,'zh'),encoding='utf-8')
    # Confirm a real endpoint-to-current-state derivative before supplying the
    # dynamics contract to the API; reward-specific derivatives are tested below.
    adapter.restore(initial)
    x=adapter.curr['coords'].detach().requires_grad_(True)
    prediction,_=adapter.predict(coordinates=x)
    probe=(adapter.endpoint(prediction)['coords']**2).sum()
    gradient,=torch.autograd.grad(probe,x)
    derivative_ok=bool(torch.isfinite(gradient).all()) and float(gradient[adapter.index].norm())>0
    write(output/'AdapterDerivativeProbe.json',dict(passed=derivative_ok,
        scope='Live endpoint squared-coordinate sum; every final reward requires its own preflight'))
    if not derivative_ok:raise ValueError('Live endpoint derivative is unavailable')
    dynamics=adapter.describe_dynamics()
    dynamics.update(live_derivative='available',editable_atom_ids=list(range(len(contexts[0].atom_ids))))
    config=agent_config(agent_file)
    config.agents['molmonitor'].enabled=False;config.monitoring.graph_review_enabled=False
    config.runtime.trace_dir=(output/'agent_runs').relative_to(REPO)
    write(output/'agents.api.json',config.model_dump(mode='json'))
    result=observed_api_runtime(config,dynamics,output/'api_requests.jsonl').run(packet,report,
        run_id='api_5i0b_'+attempt,execute=False,feedback={
            'kind':'UserTaskContext',
            'task':'Test one 5i0b molecule from the copied t=0.50 checkpoint: design and validate a real coordinate reward, then run live gradient guidance at weights '+', '.join(map(str,weights))+'.',
            'monitor_enabled':False,'graph_review_enabled':False,
            'scope':'A controlled mechanism experiment under the current prediction hypothesis; native categorical sampling continues. Independently evaluate final molecules, including changed graph applicability. Do not claim terminal repair from an intermediate reward decrease.',
            'comparison':'Same checkpoint, self-conditioning, RNG, execution strength and displacement budgets for every arm.'})
    write(output/'api_status.json',dict(status=result['status'],errors=result.get('errors',[]),
        checkpoint_path=result['checkpoint_path'],mode=config.mode,
        monitor_enabled=False,trace_nodes=sorted({e['node'] for e in result['trace']})))
    if result['status']!='validated':
        raise RuntimeError('Real API workflow did not validate an executable reward; no fallback is allowed')
    guard=json.loads((REPO/'experiments/guidance/creativity.json').read_text(encoding='utf-8'))
    program,strength,editable,_=compile_validated_agent_checkpoint(result['checkpoint_path'],guard)
    write(output/'RewardProgram.api.json',program)
    adapter.config['editable_atom_ids']=editable
    # Agent tests and evidence probes may consume random numbers. Every arm must
    # restart from the original t=.50 state, SC and RNG, including the native arm.
    adapter.restore(initial)
    initial=adapter.checkpoint()
    baseline=torch.load(stage/'world_prediction.pt',map_location=adapter.device,weights_only=True)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(adapter.config['receptor'])
    for atom in receptor:atom['vdw_radius']=Chem.GetPeriodicTable().GetRvdw(atom['atomic_number'])
    vocabulary=load_config()
    budget=GuidanceBudget(strength=strength,max_step_angstrom=.02,max_path_angstrom=.5)
    adapter.restore(initial)
    native=run_suffix(adapter,make_reward(program,baseline,receptor,vocabulary),output/'unguided',budget,'unguided')
    # A single molecule is edited; keep the original batch for sampler randomness.
    for key in ('curr','cond','times'):
        assert_equal(getattr(adapter,key),bundle['checkpoints']['1.00'][key],'native.'+key)
    assert_equal(snapshot_rng(),bundle['final']['rng'],'native.rng')
    with torch.no_grad():
        final,_=adapter.predict(times=adapter.model._update_times(adapter.times,-1e-4))
    assert_equal(cpu_copy(final),bundle['final']['prediction'],'native.final_prediction')
    write(output/'native_replay_verification.json',{'passed':True,'scope':'All batch state, SC, RNG and final head'})
    matrix=[]
    for weight in weights:
        weighted=deepcopy(program);weighted['reward_weight']=weight
        weighted['program_id']='rp_'+digest({k:v for k,v in weighted.items() if k!='program_id'})[:24]
        armout=output/f'weight_{weight}';armout.mkdir()
        write(armout/'RewardProgram.json',weighted)
        adapter.restore(initial)
        reward=make_reward(weighted,baseline,receptor,vocabulary)
        check=check_live_gradient(adapter,reward)
        write(armout/'gradient_preflight.json',check)
        if not check['passed']:raise ValueError('Reward-specific live gradient check failed at weight '+str(weight))
        adapter.restore(initial)
        execution=run_suffix(adapter,reward,armout/'run',budget,'agent')
        rows=[json.loads(line) for line in (armout/'run/guidance_trace.jsonl').read_text().splitlines()]
        item=dict(weight=weight,**execution,
            effective_steps=sum(r.get('accepted_guidance_l2_angstrom',0)>1e-12 for r in rows),
            initial_gradient_norm=next((r['gradient_norm'] for r in rows if 'gradient_norm' in r),None),
            maximum_injected_angstrom=max((r.get('injected_max_angstrom',0) for r in rows),default=0))
        matrix.append(item)
        write(output/'weight_summary.json',dict(status='in_progress',native=native,weights=matrix))
    write(output/'weight_summary.json',dict(status='completed',native=native,weights=matrix,
        monitor_enabled=False,graph_review_enabled=False,weight_definition='Rw=wR; expert final direction is also multiplied by w'))
    print(json.dumps({'status':'completed','output':str(output),'weights':list(weights)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root',type=Path,default=REPO/'test/5i0b_t050_api_weights_20260930')
    parser.add_argument('--agent-config',type=Path,default=REPO/'configs/agents.json')
    parser.add_argument('--attempt',default='run_01')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--weights',type=float,nargs='+',default=WEIGHTS,
                        help='Positive distinct global reward weights; all use the same reward and native randomness.')
    args=parser.parse_args()
    root=args.test_root.resolve()
    if not root.is_relative_to(REPO/'test') or Path(args.attempt).name!=args.attempt:
        parser.error('All output must remain under the repository test directory')
    try:
        weights=normalized_weights(args.weights)
    except ValueError as exc:
        parser.error(str(exc))
    readiness=preflight(root,args.agent_config.resolve(),weights)
    if readiness['status']!='ready':return 2
    if args.preflight_only:return 0
    exit_code = 0
    try:
        execute(root,args.agent_config.resolve(),args.attempt,weights)
    except Exception as exc:
        write(root/(args.attempt+'.failure.json'),dict(status='failed',error_type=type(exc).__name__,
            message=str(exc)[:400],weights={str(w):{'status':'see_attempt_artifacts'} for w in weights}))
        raise
    finally:
        if not audit_sources(root, 'after')['passed']:
            exit_code = 3
            print(json.dumps({'status':'source_integrity_failed',
                              'audit':str(root/'source_integrity_after.json')}), flush=True)
    return exit_code


if __name__=='__main__':raise SystemExit(main())
