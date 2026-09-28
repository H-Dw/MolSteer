"""Paired continuation sweep: vary external strength, keep reward/budgets fixed."""
import argparse
import json
import random
import hashlib
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
from molreader.io import load_config,parse_pdb
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.program import MolecularReward
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molexecutor.engine import run_suffix
from molsteer.molmonitor.live_gradient import check_live_gradient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--weights',type=float,nargs='+',default=[0.,.3,1.,3.,10.,30.,100.,300.])
    parser.add_argument('--guide-until',type=float,default=1.)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    if output.exists():raise ValueError('Sweep output must be new')
    output.mkdir(parents=True)
    if not .5<args.guide_until<=1.:raise ValueError('Invalid guidance interval')
    if any(w<0 or not np.isfinite(w) for w in args.weights):raise ValueError('Invalid strength')
    config.update(output=str(output),record_tensor_trace=True,guidance_interval=[.5,args.guide_until])
    weights=args.weights
    seeds=[None,2026092301,2026092302]
    manifest=dict(config=config,weights=weights,suffix_seeds=seeds,
        controlled_variables='External gradient multiplier only; reward, SC start, original state, precision and step/path budgets fixed',
        seed_meaning='None restores the original shared runtime RNG; integer seeds replace only suffix randomness after restoring the same state/SC',
        checkpoint_sha256=hashlib.sha256(Path(config['resume_checkpoint']).read_bytes()).hexdigest(),
        reward_sha256=hashlib.sha256(Path(config['reward_programs']['creativity']).read_bytes()).hexdigest())
    (output/'sweep_manifest.json').write_text(json.dumps(manifest,indent=2))
    adapter=FlowrRootAdapter(config)
    checkpoint=torch.load(config['resume_checkpoint'],weights_only=True,map_location='cpu')
    adapter.restore(checkpoint)
    b=torch.load(Path(config['saved_stage'])/'world_prediction.pt',weights_only=True,map_location=adapter.device)
    baseline={k:v[0] for k,v in b.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(config['receptor']);periodic=Chem.GetPeriodicTable()
    for a in receptor:a['vdw_radius']=periodic.GetRvdw(a['atomic_number'])
    spec=json.loads(Path(config['reward_programs']['creativity']).read_text())
    (output/'RewardProgram.json').write_text(json.dumps(spec,indent=2))
    preflight=check_live_gradient(adapter,MolecularReward(spec,baseline,receptor,load_config()))
    (output/'gradient_preflight.json').write_text(json.dumps(preflight,indent=2))
    if not preflight['passed']:raise ValueError('Gradient preflight failed')
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    records=[]
    for seed_index,seed in enumerate(seeds):
        for weight in weights:
            adapter.restore(checkpoint)
            if seed is not None:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            run_dir=output/f'seed_{seed_index}'/('eta_'+format(weight,'g').replace('.','p'))
            run_dir.mkdir(parents=True)
            arm='creativity' if weight else 'unguided'
            budget=GuidanceBudget(strength=weight or 1.,max_step_angstrom=config['budget']['max_step_angstrom'],
                max_path_angstrom=config['budget']['max_path_angstrom'])
            reward=MolecularReward(spec,baseline,receptor,load_config())
            summary=run_suffix(adapter,reward,run_dir/arm,budget,arm)
            summary.update(seed_index=seed_index,suffix_seed=seed,strength=weight,run_dir=str(run_dir),
                budget=vars(budget),arm=arm)
            records.append(summary)
            (output/'sweep_runs.json').write_text(json.dumps(records,indent=2))
            print('SWEEP_RUN '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
