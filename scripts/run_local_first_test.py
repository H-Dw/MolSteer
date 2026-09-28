"""Build a matched local reference and run local-first FLOWR suffix tests."""
import argparse
import copy
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

from molreader.io import load_config, parse_pdb
from molsteer.common import digest, file_hash, write_json
from molsteer.molexecutor.chemistry import decode_endpoint, signature
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molexecutor.local_first_engine import run_local_first_suffix
from molsteer.molexecutor.local_first_reward import LocalMMFFRelaxation, local_geometry
from molsteer.molexecutor.program import make_reward


def build_local_reference(endpoints_path, native_reference_path, runtime_path, output, vocabulary, config):
    endpoints=torch.load(endpoints_path,weights_only=True,map_location='cpu');oracle=LocalMMFFRelaxation();frames=[]
    for key,pred in sorted(endpoints.items(),key=lambda item:float(item[0])):
        try:
            mol=decode_endpoint(pred,vocabulary);x=pred['coords'];value,detail=local_geometry(x,mol,config['core'],
                config['bond_tolerance_fraction'],config['angle_tolerance_degrees'],config['tau_region'])
            local=oracle.evaluate(mol,x.numpy(),config['core'])
            frames.append(dict(time=float(key),status='ok',graph_signature=signature(mol),smiles=Chem.MolToSmiles(mol),
                halo=local['halo'],local_strain_kcal_mol=local['strain'],local_geometry_loss=float(value),
                local_geometry_max_residual=detail['max_residual'],relation_count=detail['relation_count']))
        except (ValueError,RuntimeError) as exc:
            frames.append(dict(time=float(key),status='unavailable',reason=str(exc)))
    result=dict(kind='MatchedNativeLocalReference',core_atom_slots=config['core'],frames=frames,
        endpoint_source=dict(path=str(endpoints_path),sha256=file_hash(endpoints_path)),
        native_reference=dict(path=str(native_reference_path),sha256=file_hash(native_reference_path)),
        origin_runtime=dict(path=str(runtime_path),sha256=file_hash(runtime_path)),
        method='Exact saved native endpoints; MMFF local relaxation with heavy atoms outside the one-bond halo fixed')
    write_json(output,result);return result


def finite_difference_preflight(endpoint,vocabulary,config):
    mol=decode_endpoint(endpoint,vocabulary);oracle=LocalMMFFRelaxation();x=endpoint['coords'].detach().clone().requires_grad_(True)
    value,result=oracle.tensor(x,mol,config['core']);gradient,=torch.autograd.grad(value,x)
    direction=torch.zeros_like(x);direction[config['core'][0],0]=1.;direction[config['core'][-1],1]=-.7
    direction/=direction.norm();eps=1e-4
    plus=oracle.evaluate(mol,(x.detach()+eps*direction).numpy(),config['core'])['strain']
    minus=oracle.evaluate(mol,(x.detach()-eps*direction).numpy(),config['core'])['strain']
    fd=(plus-minus)/(2*eps);ad=float((gradient*direction).sum());scale=max(1.,abs(fd),abs(ad))
    return dict(passed=abs(fd-ad)/scale<.05,finite_difference=fd,autograd_directional=ad,
        relative_scaled_error=abs(fd-ad)/scale,halo=result['halo'],epsilon_angstrom=eps)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();source=Path(a.source_root).resolve();root.mkdir(parents=True,exist_ok=False)
    integrated=source/'output/molsteer_researcher_integrated_20260924'
    outcome=source/'output/molsteer_outcome_guidance_20260923/final_validation'
    cfg=json.loads((integrated/'base_execution.json').read_text());runtime=Path(cfg['resume_checkpoint'])
    endpoints=outcome/'native_endpoints.pt';native_reference=outcome/'native_reference.json';vocabulary=load_config()
    local_config=dict(core=[1,10,14],tau_region=.05,bond_tolerance_fraction=.10,angle_tolerance_degrees=30.,
        strain_scale_kcal_mol=5.,minimum_extra_improvement_kcal_mol=2.,persistence_frames=3,
        geometry_zero_tolerance=1e-7,minimum_same_time_gain=1e-6,max_endpoint_outside_halo_delta_angstrom=.10,
        graph_change_max_geometry_loss=1e-5,graph_change_strain_allowance_kcal_mol=0.,geometry_regression_allowance=.002,
        strain_regression_allowance_kcal_mol=.5,review_every=5,
        oracle_guard=dict(vina_regression_allowance=.25,strain_regression_allowance=2.))
    local_ref_path=root/'local_native_reference.json'
    local_ref=build_local_reference(endpoints,native_reference,runtime,local_ref_path,vocabulary,local_config)
    preflight=finite_difference_preflight(torch.load(endpoints,weights_only=True,map_location='cpu')['0.50'],vocabulary,local_config)
    write_json(root/'gradient_preflight.json',preflight)
    if not preflight['passed']:raise ValueError('Local MMFF gradient preflight failed')
    base_program=json.loads((integrated/'programs/research_off.json').read_text());programs=root/'programs';programs.mkdir()
    template=copy.deepcopy(base_program);template.pop('program_id',None);template['evaluator']='local_first'
    template['region_atom_ids']=local_config['core'];template['local_first']={k:v for k,v in local_config.items() if k!='core'}
    template['local_native_reference']=dict(path=str(local_ref_path),sha256=file_hash(local_ref_path))
    template['reward']='branch gate -> conditional local geometry -> local relaxation strain -> projected matched-control pKd'
    template['active_objectives']=['conditional_local_geometry','local_relaxation_strain','terminal_pkd']
    template['weights']=[1.,1.,1.]
    template['aggregation_policy']='lexicographic_staged_noncompensable'
    records=[];RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    baseline=torch.load(Path(cfg['saved_stage'])/'world_prediction.pt',weights_only=True,map_location=f"cuda:{cfg['gpu']}")
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(cfg['receptor']);periodic=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=periodic.GetRvdw(atom['atomic_number'])
    controls=json.loads(Path(cfg['control_trajectory']).read_text());checkpoint=torch.load(runtime,weights_only=True,map_location='cpu')
    for eta in [100.,300.]:
        label=f'local_first_eta{int(eta)}';spec=copy.deepcopy(template);spec['test_strength_eta']=eta
        spec['program_id']='rp_'+digest(spec)[:24];program_path=programs/f'{label}.json';write_json(program_path,spec)
        run_cfg=copy.deepcopy(cfg);run_cfg.update(output=str(root/'continuations'/label),
            reward_programs=dict(creativity=str(program_path)),budget=dict(strength=eta,max_step_angstrom=.04,max_path_angstrom=2.),
            guidance_interval=[.5,1.]);run_cfg.pop('monitor',None)
        adapter=FlowrRootAdapter(run_cfg);adapter.restore(checkpoint)
        reward=make_reward(spec,baseline,receptor,vocabulary,controls[spec['affinity_head']])
        write_json(root/'continuations'/label/'execution.json',run_cfg)
        summary=run_local_first_suffix(adapter,reward,root/'continuations'/label/'local_first',GuidanceBudget(**run_cfg['budget']))
        records.append(dict(label=label,eta=eta,summary=summary));write_json(root/'runs.json',records)
        print('COMPLETE '+label,flush=True)
        del adapter,reward;torch.cuda.empty_cache()
    original=source/'output/crossdocked_100target_stage_test/5i0b_A__5vef_M77/ligand_002/t_0.50'
    exact=source/'output/crossdocked_100target_stage_test_exact_20260923/5i0b_A__5vef_M77/ligand_002/t_0.50'
    write_json(root/'source_binding.json',dict(requested_stage=str(original),requested_state_sha256=file_hash(original/'state.pt'),
        requested_sdf_sha256=file_hash(original/'ligand.sdf'),requested_state_has_runtime_context=False,
        continuation_runtime=str(runtime),continuation_runtime_sha256=file_hash(runtime),exact_stage=str(exact),
        reason='The requested state.pt has molecular tensors only; the content-bound exact runtime supplies saved RNG and self-conditioning.'))
    write_json(root/'execution_provenance.json',dict(model_checkpoint_sha256=checkpoint['model_checkpoint_sha256'],
        native_reference_sha256=file_hash(native_reference),native_endpoints_sha256=file_hash(endpoints),
        local_reference_sha256=file_hash(local_ref_path),sampler_seed_sweep=False,gradient_preflight=preflight,
        parameter_status=dict(tau_region='pilot smooth-maximum temperature',strain_scale='inherited 5 kcal/mol pilot scale',
            minimum_extra_improvement='2 kcal/mol threshold inherited from the independently used strain regression scale')))
    print('ALL LOCAL-FIRST RUNS COMPLETE',flush=True)


if __name__=='__main__':main()
