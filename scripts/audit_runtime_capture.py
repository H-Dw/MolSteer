"""Audit actual saved fields and all live-capture snapshot bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from molsteer.common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',required=True);p.add_argument('--capture-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.model_root).resolve();capture=Path(a.capture_root).resolve()
    old=root/'output/crossdocked_100target_stage_test/5i0b_A__5vef_M77/ligand_002/t_0.50'
    new=capture/'5i0b_A__5vef_M77/ligand_002/t_0.50'
    paths=[old/'state.pt',old/'structure_affinity_prediction.pt',old/'world_prediction.pt',
        root/'MolSteer/experiments/guidance/resume_start.pt',
        root/'output/molsteer_guidance_20260923_fp32/creativity/resume_t_0.75.pt',new/'state.pt',new/'runtime.pt']
    inventory=[]
    for path in paths:
        data=torch.load(path,weights_only=True,map_location='cpu')
        inventory.append(dict(path=str(path),keys=list(data),self_condition_fields=list(data.get('cond',{})),
            rng_fields=list(data.get('rng',{})),format=data.get('format'),
            provenance=data.get('resume_fidelity','Inspect historical initial-checkpoint provenance if this is a legacy MolExecutor runtime')))
    checks=[]
    for source in sorted(capture.glob('*/ligand_*/*/runtime.pt')):
        meta=json.loads(source.with_suffix('.json').read_text());runtime=torch.load(source,weights_only=True,map_location='cpu')
        original=torch.load(source.parent/'state.pt',weights_only=True,map_location='cpu');index=meta['batch_index']
        exact=all(torch.equal(v,runtime['curr'][k][index:index+1]) for k,v in original.items() if torch.is_tensor(v))
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        assert exact and sha==meta['sha256']
        assert all(k in runtime['rng'] for k in ['python','numpy','torch','cuda']) and runtime['cond']
        assert runtime['cuda_device_index']==1 and runtime['precision']=='highest'
        checks.append(dict(path=str(source),snapshot_matches=True,sha256=sha,self_condition_saved=True,rng_saved=True,
            cuda_device_index=runtime['cuda_device_index'],step_index=runtime['step_index']))
    assert len(checks)==24
    write_json(a.output,dict(inventory=inventory,live_checkpoints=checks,all_24_snapshot_bindings_exact=True,
        interpretation='Original compact state/head files lack SC and RNG. Legacy MolExecutor runtime files include them, but their starting SC/RNG were replay reconstructed. New stage runtime files capture the actual live original trajectory.'))
    print(json.dumps(dict(inventory_count=len(inventory),runtime_checkpoints=len(checks),all_exact=True)))


if __name__=='__main__':main()
