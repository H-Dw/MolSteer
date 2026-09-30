#!/usr/bin/env python3
"""Resume FLOWR from a validated Agent reward and a matched live checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch


def _load_generator_first(flowr_root: Path) -> Path:
    """Load FLOWR's native extensions before the Agent dependency graph."""
    model_root=flowr_root.resolve(strict=True)
    if not (model_root/'flowr/__init__.py').is_file():
        raise ValueError('flowr_root does not contain the generator package')
    if str(model_root) not in sys.path:
        sys.path.insert(0,str(model_root))
    module=importlib.import_module('flowr.gen.generate_from_pdb')
    if not Path(module.__file__).resolve().is_relative_to(model_root):
        raise ValueError('Loaded FLOWR package does not match --flowr-root')
    return model_root


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _within(path: Path, root: Path) -> bool:
    return path==root or root in path.parents


def prepare(agent_checkpoint: Path, flowr_config: Path, flowr_root: Path, output_root: Path):
    model_root=_load_generator_first(flowr_root)
    from molsteer.molexecutor.agent_bridge import compile_validated_agent_checkpoint
    output_root=output_root.resolve()
    output_dir=model_root/'output'
    if not output_dir.is_dir() or not _within(output_root.resolve(),output_dir):
        raise ValueError('A fresh output root under flowr_root/output is required')
    if output_root.exists():
        raise ValueError('Refusing to overwrite an existing output root')
    base=json.loads(flowr_config.read_text(encoding='utf-8'))
    if base.get('adapter')!='flowr_root' or Path(base['model_root']).resolve(strict=True)!=model_root:
        raise ValueError('FLOWR configuration does not match the selected generator')
    for name in ('saved_stage','reward_reference_stage','resume_checkpoint','stage_runner',
                 'checkpoint','receptor','reference_ligand'):
        path=Path(base[name]).resolve(strict=True)
        if not _within(path,model_root):
            raise ValueError('FLOWR configuration contains an external path: '+name)
        base[name]=str(path)
    stage=Path(base['saved_stage'])
    if Path(base['reward_reference_stage'])!=stage or Path(base['resume_checkpoint'])!=stage/'runtime.pt':
        raise ValueError('Stage, reward reference and runtime checkpoint differ')
    template_path=Path(next(iter(base['reward_programs'].values()))).resolve(strict=True)
    if not _within(template_path,model_root):
        raise ValueError('Guard template is outside the generator checkout')
    template=json.loads(template_path.read_text(encoding='utf-8'))
    program,strength,editable,audit=compile_validated_agent_checkpoint(agent_checkpoint,template)
    packet=audit['artifacts']['packet']
    identity=packet['identity']
    if (identity['target_id']!=base['target_id'] or identity['ligand_id']!=f"ligand_{base['ligand_index']:03d}"
            or identity['stage']!=stage.name or stage.parent.name!=identity['ligand_id']
            or stage.parent.parent.name!=identity['target_id']):
        raise ValueError('Agent packet and FLOWR stage identity differ')
    sources=packet['provenance']['sources']
    expected={name:stage/name for name in ('state.pt','world_prediction.pt',
        'structure_affinity_prediction.pt','predictions.json','runtime.pt','ligand.sdf')}
    expected['stage_runner.py']=Path(base['stage_runner'])
    expected[Path(base['receptor']).name]=Path(base['receptor'])
    for name,path in expected.items():
        record=sources.get(name)
        if (not isinstance(record,dict) or Path(record['path']).resolve(strict=True)!=path
                or _sha(path)!=record.get('sha256')):
            raise ValueError('Agent evidence does not match FLOWR source: '+name)
    runtime=torch.load(base['resume_checkpoint'],map_location='cpu',weights_only=True)
    if (runtime.get('format')!='flowr_root_live_runtime'
            or runtime.get('step_index')!=base['start_step']
            or runtime.get('source_hash')!=_sha(Path(base['stage_runner']))
            or runtime.get('model_checkpoint_sha256')!=_sha(Path(base['checkpoint']))
            or runtime.get('config',{}).get('target_id')!=base['target_id']):
        raise ValueError('FLOWR runtime provenance or resume step differs')
    if base.get('float32_matmul_precision')!=runtime.get('precision') or base.get('gpu')!=runtime.get('cuda_device_index'):
        raise ValueError('FLOWR precision or GPU differs from the saved runtime')
    budget=dict(base['budget'])
    budget['strength']=min(float(budget['strength']),strength)
    if not 0<budget['strength']<=1:
        raise ValueError('Invalid mapped Agent strength')
    config=dict(base)
    config.update(model_root=str(model_root),output=str(output_root/'run'),
                  reward_programs={'agent':str(output_root/'RewardProgram.agent.json')},
                  arms=['unguided','agent'],editable_atom_ids=editable,
                  budget=budget,gradient_preflight=True,record_tensor_trace=True,
                  agent_run_id=audit['run_id'],agent_reward_id=program['source_reward_id'],
                  agent_checkpoint_sha256=_sha(agent_checkpoint))
    output_root.mkdir(parents=True)
    (output_root/'RewardProgram.agent.json').write_text(json.dumps(program,ensure_ascii=False,indent=2),encoding='utf-8')
    (output_root/'execution.agent.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (output_root/'bridge_manifest.json').write_text(json.dumps({
        'agent_run_id':audit['run_id'],'agent_reward_id':program['source_reward_id'],
        'program_id':program['program_id'],'packet_id':program['packet_id'],
        'editable_atom_ids':editable,'mapped_strength':budget['strength'],
        'source_runtime_sha256':_sha(Path(base['resume_checkpoint'])),
        'source_stage_runner_sha256':runtime['source_hash'],
        'continuation_status':'prepared','arms':['unguided','agent'],
        'view_contract':'state terms use live X_t in receptor-world coordinates; prediction terms use the differentiable X_hat_1 head',
    },ensure_ascii=False,indent=2),encoding='utf-8')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-checkpoint',type=Path,required=True)
    parser.add_argument('--flowr-config',type=Path,required=True)
    parser.add_argument('--flowr-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--graph-review-agent-config',type=Path,
                        help='Enable live graph-change Reader/Thinker/Executor review using this Agent configuration')
    parser.add_argument('--max-graph-reviews',type=int,default=4)
    args=parser.parse_args()
    _load_generator_first(args.flowr_root)
    from molsteer.molexecutor.runner import run
    output_root=args.output_root.resolve()
    config=prepare(args.agent_checkpoint.resolve(strict=True),args.flowr_config.resolve(strict=True),
                   args.flowr_root,output_root)
    if args.graph_review_agent_config:
        if args.max_graph_reviews<1:raise ValueError('max-graph-reviews must be positive')
        config['graph_review']=dict(agent_config=str(args.graph_review_agent_config.resolve(strict=True)),
                                    max_reviews=args.max_graph_reviews)
        (output_root/'execution.agent.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AGENT_FLOWR_PREPARED '+str(output_root),flush=True)
    if args.prepare_only:return
    summary=run(config)
    (output_root/'bridge_result.json').write_text(json.dumps({
        'status':'completed','agent_run_id':config['agent_run_id'],
        'agent_reward_id':config['agent_reward_id'],'executions':summary['executions'],
        'source_hashes':summary['source_hashes'],
    },ensure_ascii=False,indent=2),encoding='utf-8')
    manifest_path=output_root/'bridge_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['continuation_status']='completed'
    manifest['execution_summary_path']=str(output_root/'run/experiment_summary.json')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AGENT_FLOWR_COMPLETED '+str(output_root),flush=True)


if __name__=='__main__':main()
