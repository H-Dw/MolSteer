"""Paired chemistry-aware comparison and measured guidance/native dynamics."""
import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from molreader.io import parse_pdb,load_config
from molsteer.molexecutor.program import MolecularReward


def average(xs):
    return statistics.mean(xs) if xs else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--tailoff');args=parser.parse_args()
    root=Path(args.root);manifest=json.loads((root/'sweep_manifest.json').read_text());config=manifest['config']
    runs=json.loads((root/'sweep_runs.json').read_text());quality=json.loads((root/'quality_comparison.json').read_text())
    q={(r['seed_index'],r['strength'],r['stage']):r for r in quality}
    b=torch.load(Path(config['saved_stage'])/'world_prediction.pt',weights_only=True,map_location='cpu')
    baseline={k:v[0] for k,v in b.items() if torch.is_tensor(v)}
    receptor,_=parse_pdb(config['receptor']);periodic=Chem.GetPeriodicTable()
    for atom in receptor:atom['vdw_radius']=periodic.GetRvdw(atom['atomic_number'])
    program=json.loads((root/'RewardProgram.json').read_text());reward=MolecularReward(program,baseline,receptor,load_config())
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    suffix=Path(config['target_id'])/f"ligand_{config['ligand_index']:03d}"/'final/world_prediction.pt'
    summary=[];phases=[]
    for run in runs:
        seed,weight=run['seed_index'],run['strength'];directory=Path(run['run_dir'])/run['arm']
        trace=[json.loads(s) for s in (directory/'guidance_trace.jsonl').read_text().splitlines()]
        valid=[r for r in trace if 'cosine_native_gradient' in r]
        attempts=[a for r in trace for a in r.get('proposal_attempts',[])]
        failures=Counter(f for a in attempts for f in a['failures'])
        current=q[seed,weight,'final'];control=q[seed,0.,'final'];same=current['smiles']==control['smiles']
        world=torch.load(directory/suffix,weights_only=True,map_location='cpu')
        score,detail=reward.evaluate({k:v[0] for k,v in world.items() if torch.is_tensor(v)})
        control_dir=root/f'seed_{seed}'/'eta_0'/'unguided'
        control_world=torch.load(control_dir/suffix,weights_only=True,map_location='cpu')
        coordinate_difference=(world['coords']-control_world['coords']).norm(dim=-1)
        frames=torch.load(directory/'tensor_trace.pt',weights_only=True,map_location='cpu')
        controls=torch.load(control_dir/'tensor_trace.pt',weights_only=True,map_location='cpu')
        raw_differences=[float((a['state_after']-b['state_after']).norm(dim=-1).max()) for a,b in zip(frames,controls)]
        first_class_difference=next((dict(step=a['step'],time_after=(a['step']+1)/100) for a,b in zip(frames,controls)
            if any(not torch.equal(a[k],b[k]) for k in ['atom_classes','charge_classes','bond_classes'])),None)
        change=current['strain']-control['strain']
        record=dict(seed=seed,strength=weight,strain=current['strain'],strain_status=current['strain_status'],
            same_final_graph=same,paired_strain_change=change if same else None,
            paired_strain_reduction_percent=-change/control['strain']*100 if same else None,
            sanitized=current['sanitized'],geometry_outliers=current['geometry_outliers'],
            protein_clashes=current['protein_clashes'],intra_clashes=current['intra_clashes'],pb_failures=current['pb_failures'],
            alert_matches=current['alert_matches'],smiles=current['smiles'],affinity=current['affinity'],
            accepted_steps=run['accepted_steps'],gradient_steps=run['gradient_steps'],
            path_max=run['max_injected_path_angstrom'],clipped_steps=sum(r.get('clipped_atom_count',0)>0 for r in trace),
            backtracking_steps=sum(len(r.get('proposal_attempts',[]))>1 for r in trace),proposal_failure_counts=dict(failures),
            opposing_steps=sum(r['cosine_drift_gradient']<0 for r in valid),
            mean_drift_cosine=average([r['cosine_drift_gradient'] for r in valid]),
            mean_raw_ratio=average([r['raw_guidance_to_native_ratio'] for r in valid]),
            mean_actual_ratio=average([r.get('accepted_to_native_ratio',0.) for r in valid]),
            final_reward=float(score),final_objectives=detail,
            final_slot_distance_max=float(coordinate_difference.max()),
            final_coordinate_rmsd_same_graph=float(coordinate_difference.square().mean().sqrt()) if same else None,
            peak_raw_slot_difference=max(raw_differences),final_raw_slot_difference=raw_differences[-1],
            first_latent_category_difference=first_class_difference,
            accepted_immediate_graph_changes=sum(r.get('candidate_graph_changed_from_base',False) for r in trace))
        summary.append(record)
        for lo,hi in [(.5,.6),(.6,.75),(.75,.9),(.9,1.01)]:
            part=[r for r in valid if lo-1e-6<=r['t']<hi-1e-6]
            phases.append(dict(seed=seed,strength=weight,lo=lo,hi=min(hi,1.),count=len(part),
                opposing_steps=sum(r['cosine_drift_gradient']<0 for r in part),
                **{key:average([r[key] for r in part if r.get(key) is not None]) for key in
                   ['cosine_drift_gradient','raw_guidance_to_native_ratio','accepted_to_native_ratio',
                    'pullback_norm_ratio','local_retention_norm_ratio','local_retention_signed_ratio']},
                geometry_zero_steps=sum(r['reward_detail']['components']['geometry']==0 for r in part),
                clash_zero_steps=sum(r['reward_detail']['components']['clash']==0 for r in part)))
    aggregates=[]
    for weight in manifest['weights']:
        selected=[s for s in summary if s['strength']==weight]
        comparable=[s for s in selected if s['same_final_graph']]
        aggregates.append(dict(strength=weight,comparable_count=len(comparable),graph_changes=len(selected)-len(comparable),
            mean_same_graph_paired_strain_change=average([s['paired_strain_change'] for s in comparable]),
            mean_same_graph_paired_reduction_percent=average([s['paired_strain_reduction_percent'] for s in comparable]),
            improved_same_graph_count=sum(s['paired_strain_change']<0 for s in comparable),
            worst_same_graph_strain_increase=max((s['paired_strain_change'] for s in comparable),default=None)))
    report=dict(records=summary,phases=phases,aggregates=aggregates,
        comparison_scope='Same molecule start with three paired suffix random states; no population significance claim',
        retention_scope='One-step linear FLOWR coordinate response with common noise and fixed self-conditioning; not long-horizon causal attribution')
    (root/'analysis.json').write_text(json.dumps(report,indent=2))
    if args.tailoff:
        tail=Path(args.tailoff);tail_quality=json.loads((tail/'quality_comparison.json').read_text())
        comparisons=[]
        for item in tail_quality:
            if item['stage']!='final':continue
            seed=item['seed_index']
            full=next(s for s in summary if s['seed']==seed and s['strength']==30)
            control=q[seed,0.,'final']
            directory=Path(item['run_dir'])/item['arm']
            world=torch.load(directory/suffix,weights_only=True,map_location='cpu')
            value,detail=reward.evaluate({k:v[0] for k,v in world.items() if torch.is_tensor(v)})
            f=torch.load(root/f'seed_{seed}/eta_30/creativity/tensor_trace.pt',weights_only=True)
            g=torch.load(directory/'tensor_trace.pt',weights_only=True)
            prefix=max(float((x['state_after']-y['state_after']).abs().max()) for x,y in zip(f[:40],g[:40]))
            categories=all(torch.equal(x[k],y[k]) for x,y in zip(f[:40],g[:40]) for k in ['atom_classes','charge_classes','bond_classes'])
            comparisons.append(dict(seed=seed,unguided_strain=control['strain'],constant_30_strain=full['strain'],
                tailoff_strain=item['strain'],tailoff_minus_constant=item['strain']-full['strain'],
                same_graph=item['smiles']==control['smiles'],
                paired_reduction_percent=(control['strain']-item['strain'])/control['strain']*100,
                prefix_coordinates_max_abs=prefix,prefix_categories_equal=categories,
                constant_final_reward=full['final_reward'],tailoff_final_reward=float(value),tailoff_final_objectives=detail))
        (root/'schedule_comparison.json').write_text(json.dumps(dict(records=comparisons,
            mean_paired_reduction_percent=average([x['paired_reduction_percent'] for x in comparisons])),indent=2))
    keys=['seed','strength','strain','same_final_graph','paired_strain_change','paired_strain_reduction_percent','final_reward',
          'accepted_steps','gradient_steps','path_max','clipped_steps','opposing_steps','mean_drift_cosine','mean_actual_ratio','final_slot_distance_max']
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#147D92','#C76135','#6B53A3'];weights=manifest['weights'];positions=list(range(len(weights)))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for seed,color in enumerate(colors):
        selected=sorted([s for s in summary if s['seed']==seed],key=lambda s:s['strength'])
        ys=[s['paired_strain_reduction_percent'] if s['same_final_graph'] else float('nan') for s in selected]
        axes[0].plot(positions,ys,'o-',color=color,label=f'Paired suffix {seed}')
        axes[1].plot(positions,[s['path_max'] for s in selected],'o-',color=color)
    axes[0].axhline(0,color='grey',linewidth=.8)
    axes[0].set_ylabel('Same-graph strain-proxy reduction (%)')
    axes[0].set_title('A  Increasing strength is not uniformly beneficial',fontsize=10)
    axes[0].legend(fontsize=8);axes[0].annotate('Suffix 0: graph changed',xy=(7,0),xytext=(4.3,3.9),fontsize=8,
        arrowprops=dict(arrowstyle='->',color='grey'))
    axes[1].axhline(.5,color='grey',linestyle='--',label='Path cap = 0.5 A')
    axes[1].set_ylabel('Max per-atom injected path (Angstrom)')
    axes[1].set_title('B  Clipping limits effective guidance',fontsize=10);axes[1].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks(positions,[format(w,'g') for w in weights]);ax.set_xlabel('External guidance strength eta');ax.grid(alpha=.2)
    phase=[p for p in phases if p['seed']==0 and p['strength']==1]
    axes[2].bar(range(4),[p['cosine_drift_gradient'] for p in phase],color=colors[0])
    axes[2].axhline(0,color='grey',linewidth=.8);axes[2].set_ylim(-.3,.05)
    axes[2].set_xticks(range(4),['.50-.60','.60-.75','.75-.90','.90-1.0']);axes[2].tick_params(axis='x',labelrotation=20)
    axes[2].set_ylabel('Mean cosine(native drift, reward gradient)')
    axes[2].set_title('C  Weak opposing directions (suffix 0, eta=1)',fontsize=10)
    fig.savefig(root/'strength_analysis.png',dpi=180);fig.savefig(root/'strength_analysis.svg');plt.close(fig)
    print(json.dumps(aggregates,indent=2),flush=True)


if __name__=='__main__':main()
