"""Read-only live gradient attribution on saved guided checkpoints."""
import argparse
import json
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import parse_pdb,load_config
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import make_reward,angle,balance
from molsteer.common import write_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());directory=Path(cfg['output'])/'creativity';adapter=FlowrRootAdapter(cfg)
    baseline=torch.load(Path(cfg['reward_reference_stage'])/'world_prediction.pt',map_location=adapter.device,weights_only=True)
    baseline={k:v[0] for k,v in baseline.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(cfg['receptor']);table=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=table.GetRvdw(atom['atomic_number'])
    spec=json.loads(Path(cfg['reward_programs']['creativity']).read_text());controls=json.loads(Path(cfg['control_trajectory']).read_text())
    reward=make_reward(spec,baseline,receptor,load_config(),controls[spec['affinity_head']]);records=[]
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    for t in [.65,.75,.85,.95]:
        path=directory/f'resume_t_{t:.2f}.pt';cp=torch.load(path,map_location='cpu',weights_only=True);adapter.restore(cp)
        reward.set_time(float(adapter.times[0][0]));latent=adapter.curr['coords'].detach().requires_grad_(True)
        pred,_=adapter.predict(coordinates=latent);endpoint=adapter.endpoint(pred)
        try:value,details=reward.evaluate(endpoint)
        except ValueError as exc:records.append(dict(time=t,status='unavailable',reason=str(exc)));continue
        utility,_=reward.affinity_utility(endpoint);chem=reward.chemistry(endpoint);x=endpoint['coords'];z=[]
        for kind,ids,ref,tol in chem['geometries']:
            measured=angle(x,ids) if kind=='angle' else (x[ids[0]]-x[ids[1]]).norm()
            z.append((measured-ref)/tol)
        z=torch.stack(z);geometry=torch.sqrt(1+torch.relu(z.abs()-1).square().mean())-1+spec['geometry_soft_weight']*z.square().mean()
        protein,intra=reward.overlaps(x,chem);clash=torch.maximum(protein.max(),intra.max()).clamp(min=0).square()/spec['clash_scale_angstrom']**2
        pocket=(torch.relu(reward.pocket_distance(x)-spec['pocket_contact_distance'])/spec['pocket_distance_scale']).square().mean()
        reconstructed=spec['affinity_weight']*utility-spec['structure_weight']*balance([geometry,clash],spec['weights'],spec['tau'],spec['rho'])-spec['pocket_weight']*pocket
        if not torch.allclose(value,reconstructed,atol=1e-6,rtol=1e-6):raise ValueError('Attribution expression differs from active reward')
        weights=x.new_tensor(spec['weights']);mix=(torch.softmax(torch.stack([geometry,clash])*weights/spec['tau'],0)+spec['rho'])*weights
        local_terms=dict(affinity=spec['affinity_weight']*utility,geometry=-spec['structure_weight']*mix[0].detach()*geometry,
            clash=-spec['structure_weight']*mix[1].detach()*clash,pocket=-spec['pocket_weight']*pocket)
        gradients={k:torch.autograd.grad(v,latent,retain_graph=True)[0][adapter.index].detach() for k,v in local_terms.items()}
        total=torch.autograd.grad(value,latent)[0][adapter.index].detach();error=float((sum(gradients.values())-total).abs().max())
        if error>1e-4:raise ValueError('Gradient decomposition does not conserve total')
        cosine=lambda a,b:float((a*b).sum()/(a.norm()*b.norm()).clamp(min=1e-20))
        rows={k:dict(norm=float(g.norm()),cosine_to_total=cosine(g,total),
            projection_share=float((g*total).sum()/total.square().sum().clamp(min=1e-20)),
            per_atom_norm=g.norm(dim=-1).cpu().tolist()) for k,g in gradients.items()}
        records.append(dict(time=t,status='ok',checkpoint=str(path),checkpoint_sha256=file_hash(path),details=details,
            components=rows,total_norm=float(total.norm()),affinity_geometry_cosine=cosine(gradients['affinity'],gradients['geometry']),
            sum_max_abs_error=error,interpretation='Local differential attribution, not a causal fraction of final benefit; coefficients of nonlinear aggregation are evaluated at this state.'))
    write_json(a.output,dict(program_id=spec['program_id'],program_sha256=file_hash(cfg['reward_programs']['creativity']),checkpoints=records,
        scope='Four read-only saved states; no inference continuation, parameter updates or seed sweep'))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
