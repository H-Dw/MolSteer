"""Capture matched native observables without changing the sampling path."""
import argparse,json
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import load_config,parse_pdb
from molsteer.common import write_json,file_hash
from molsteer.molexecutor.flowr import FlowrRootAdapter,tree_map,snapshot_rng,restore_rng
from molsteer.molexecutor.outcome_reward import OutcomeObservables
from molsteer.molexecutor.interface_energy import prepare_protein
from molsteer.molexecutor.oracles import VinaOracle
from run_exact_continuations import unequal


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'native_reference.json').exists():raise ValueError('Use fresh reference output')
    cfg=json.loads(Path(a.config).read_text());cfg.pop('monitor',None);cfg.pop('categorical_proposal',None)
    evaluation=Path(a.evaluation);context=json.loads((evaluation/'ReaderComparisonContext.json').read_text())
    native=next(r for r in context['observations'] if r['label']=='native_final')
    mol=Chem.MolFromMolFile(native['source_sdf'],removeHs=True)
    prepared=evaluation/'preparation/receptor_prepared.pdb'
    protein=prepare_protein(prepared,mol.GetConformer().GetPositions())
    observables=OutcomeObservables(protein)
    scorer_config=dict(receptor_pdbqt=str(evaluation/'preparation/receptor.pdbqt'),
        receptor_sha256=file_hash(evaluation/'preparation/receptor.pdbqt'),center=[13.7376,34.2223,14.51965],box_size=[24.,24.,24.])
    vocab=load_config();oracle=VinaOracle(scorer_config,observables,vocab)
    adapter=FlowrRootAdapter(cfg);checkpoint=torch.load(cfg['resume_checkpoint'],weights_only=True,map_location='cpu');adapter.restore(checkpoint)
    receptor,_=parse_pdb(cfg['receptor']);rxyz=torch.tensor([r['coords'] for r in receptor],device=adapter.device)
    frames=[];endpoints={};RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    while True:
        step=adapter.step_index;t=step/adapter.args.integration_steps
        times=adapter.model._update_times(adapter.times,-1e-4) if t==1 else adapter.times
        with torch.no_grad():pred,cond=adapter.predict(times=times);endpoint=adapter.endpoint(pred)
        frame=dict(time=t,model_time=float(times[0][0]),affinity={k:float(v) for k,v in endpoint['affinity'].items()})
        endpoints[f'{t:.2f}']=tree_map(lambda v:v.detach().cpu(),endpoint)
        assessment_rng=snapshot_rng()
        try:
            with torch.no_grad():values,decoded,_=observables.evaluate(endpoint,vocab)
            frame.update(status='ok',observables={k:float(v) for k,v in values.items()},smiles=Chem.MolToSmiles(decoded),
                pocket=float(torch.relu(torch.cdist(endpoint['coords'],rxyz).min(-1).values-4.5).square().mean()))
            if step%5==0:frame['oracle']=oracle.score(endpoint)
        except ValueError as exc:frame.update(status='unavailable',reason=str(exc))
        frame['assessment_rng_changes']=unequal(assessment_rng,snapshot_rng())
        restore_rng(assessment_rng)
        frames.append(frame)
        if step%10==0:print(json.dumps(frame),flush=True)
        if t==1:break
        adapter.native_step(pred,cond,adapter.grid[step+1]-adapter.grid[step])
    expected=torch.load(Path(cfg['saved_stage']).parent/'final/runtime.pt',weights_only=True,map_location='cpu')
    actual=adapter.checkpoint();diff={k:unequal(expected[k],actual[k]) for k in ('curr','cond','times','rng')}
    write_json(out/'reference_fidelity.json',dict(all_exact=not any(diff.values()),differences=diff))
    if any(diff.values()):raise ValueError('Reference measurement altered native generation')
    bundle=dict(kind='OutcomeNativeReference',subject=context['subject'],frames=frames,protein=protein,
        mmff_references=observables.mmff.references,oracle_config=scorer_config,
        native_final_sdf=dict(path=native['source_sdf'],sha256=native['source_sha256']),
        bindings=dict(origin_runtime=cfg['resume_checkpoint'],origin_runtime_sha256=file_hash(cfg['resume_checkpoint']),
            model_checkpoint=cfg['checkpoint'],model_checkpoint_sha256=checkpoint['model_checkpoint_sha256'],receptor=cfg['receptor'],
            prepared_receptor=str(prepared),prepared_receptor_sha256=file_hash(prepared)),
        fidelity=dict(all_exact=True,differences=diff),
        method='Full native trajectory; fixed graph-local MMFF minima, H-envelope strain, directional proxy and smooth SASA; no stochastic seed sweep')
    write_json(out/'native_reference.json',bundle);torch.save(endpoints,out/'native_endpoints.pt')
    write_json(out/'base_execution.json',cfg)
    print('COMPLETE '+str(sum(r['status']=='ok' for r in frames))+'/'+str(len(frames)),flush=True)


if __name__=='__main__':main()
