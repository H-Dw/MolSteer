"""Real FLOWR pullback finite differences and mid-suffix restart audit."""
import argparse
import json
from pathlib import Path
import torch
from rdkit import Chem
from molreader.io import load_config,parse_pdb
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import MolecularReward
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.interfaces import GuidanceBudget


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gradient-only',action='store_true')
    args=p.parse_args();config=json.loads(Path(args.config).read_text())
    output=Path(config['output'])
    adapter=FlowrRootAdapter(config)
    checkpoint=torch.load(output/'resume_start.pt',weights_only=True,map_location='cpu')
    adapter.restore(checkpoint)
    b=torch.load(Path(config['saved_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in b.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(config['receptor']);table=Chem.GetPeriodicTable()
    for a in receptor:a['vdw_radius']=table.GetRvdw(a['atomic_number'])
    from molsteer.molmonitor.live_gradient import check_live_gradient
    audits=[]
    for arm in ['selection','creativity']:
        spec=json.loads(Path(config['reward_programs'][arm]).read_text())
        reward=MolecularReward(spec,baseline,receptor,load_config())
        result=check_live_gradient(adapter,reward)
        result['arm']=arm
        audits.append(result)
    (output/'live_gradient_audit.json').write_text(json.dumps(audits,indent=2))
    print(json.dumps(audits),flush=True)
    if args.gradient_only:return
    # Restart uses the saved guidance path budget, self-conditioning and RNG.
    adapter.restore(torch.load(output/'creativity/resume_t_0.75.pt',weights_only=True,map_location='cpu'))
    spec=json.loads(Path(config['reward_programs']['creativity']).read_text())
    reward=MolecularReward(spec,baseline,receptor,load_config())
    restart=output/'restart_audit'
    result=run_suffix(adapter,reward,restart,GuidanceBudget(**config['budget']),'creativity')
    suffix=Path(config['target_id'])/f"ligand_{config['ligand_index']:03d}"/'final/structure_affinity_prediction.pt'
    a=torch.load(output/'creativity'/suffix,weights_only=True,map_location='cpu')
    b=torch.load(restart/suffix,weights_only=True,map_location='cpu')
    comparison={k:float((a[k]-v).abs().max()) for k,v in b.items() if torch.is_tensor(v)}
    result.update(final_max_abs_difference=comparison,
        categories_match=all(torch.equal(a[k].argmax(-1),b[k].argmax(-1)) for k in ['atomics','bonds','charges']))
    (output/'restart_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
