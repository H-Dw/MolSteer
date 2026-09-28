"""Summarize all monitor outcomes, including rejected and paused continuations."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from molsteer.common import write_json,file_hash


def stats(values):
    return dict(min=float(min(values)),median=float(np.median(values)),max=float(max(values))) if values else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();records=json.loads((root/'all_runs.json').read_text())
    quality=json.loads((root/'final_quality.json').read_text());reference=json.loads((root/'reference.json').read_text())
    control_root=Path(records[0]['config']['control_trajectory']).parent
    control=next(q for q in json.loads((control_root/'final_quality.json').read_text()) if q['label']=='control')
    traces={};summaries=[]
    for r in records:
        trace=Path(r['directory'])/r['arm']/'monitor_trace.jsonl'
        rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
        traces[r['label']]=rows
        selected=[v['selected'] for v in rows if v.get('selected')]
        failures=Counter(f for v in rows for t in v['proposal_attempts'] for f in t['failures'])
        requests=[]
        folder=Path(r['directory'])/r['arm']/'feedback'
        for path in sorted(folder.glob('*.json')):
            if path.name.endswith('.context.json'):continue
            req=json.loads(path.read_text());requests.append(dict(request_id=req['request_id'],event=req['event'],
                sentinel=req['observations'].get('independent_sentinel'),local_geometry=req['observations']['selected'].get('localized_geometry'),
                temporal_evidence=req['observations']['localized_rate_evidence'][:4],source=str(path)))
        summaries.append(dict(label=r['label'],status=r['summary'].get('status','complete'),summary=r['summary'],
            budget=r['config']['budget'],eta=stats([s['eta'] for s in selected]),effective_eta=stats([s['effective_eta'] for s in selected]),
            effective_l2=stats([s['effective_l2'] for s in selected]),trials=sum(len(v['proposal_attempts']) for v in rows),
            rejected_trials=sum(bool(t['failures']) for v in rows for t in v['proposal_attempts']),rejection_reasons=dict(failures),
            clipped_selected=sum(s['effective_eta']<s['eta']*.99 for s in selected),
            graph_changes_selected=sum(s['graph_changed'] for s in selected),requests=requests))
    analysis=dict(control=control,final_quality=quality,runs=summaries,reference=dict(bindings=reference['bindings'],
        coarse=reference['coarse_050_075_100'],frame_count=len(reference['frames']),fidelity=reference.get('fidelity')))
    # Reference builders store auxiliary fields explicitly; retain their names.
    analysis['reference']['metadata']={k:v for k,v in reference.items() if k not in ['frames','bindings']}
    write_json(root/'analysis.json',analysis)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    labels=['adaptive_same_budget','adaptive_wide_review','stress_monitored']
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for index,label in enumerate(labels):
        rows=traces[label]
        axes[index,0].plot([v['next_t'] for v in rows],[v['selected']['eta'] if v['selected'] else np.nan for v in rows],label='Nominal eta',marker='.',ms=3)
        axes[index,0].plot([v['next_t'] for v in rows],[v['selected']['effective_eta'] if v['selected'] else np.nan for v in rows],label='After clipping',ls='--')
        axes[index,0].set_yscale('log');axes[index,0].set_ylabel(label.replace('_',' ')+'\nGuidance coefficient')
        native=reference['frames']
        axes[index,1].plot([v['time'] for v in native],[v.get('geometry_rms_z',np.nan) for v in native],color='0.6',label='Native reference')
        axes[index,1].plot([v['next_t'] for v in rows],[v['committed'].get('geometry_rms_z',np.nan) for v in rows],color='#17847a',label='Committed endpoint')
        axes[index,1].set_ylabel('RMS normalized geometry residual')
        for v in rows:
            if v['decision']['destination']=='MolThinker':
                for ax in axes[index]:ax.axvline(v['next_t'],color='#c14c39',ls=':',label='Revision request')
        for ax in axes[index]:ax.grid(alpha=.2);ax.set_xlim(.5,1.)
    for ax in axes[-1]:ax.set_xlabel('Generation progress t')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('MolMonitor: effective control and endpoint geometry\nOne matched FLOWR.ROOT continuation; pauses remain visible',fontsize=13)
    fig.text(.1,.015,'Gaps: unavailable checks or no accepted control. Residual references rebind after graph changes; jumps are not necessarily coordinate jumps.',fontsize=8)
    fig.tight_layout(rect=[0,.025,1,1]);fig.savefig(root/'guidance_dynamics.png');plt.close(fig)
    points=[control]+quality
    fig,ax=plt.subplots(figsize=(9,5.7));offsets={'control':(6,-17),'fixed_300':(-80,-18),'adaptive_same_budget':(8,-19),
        'adaptive_wide':(7,5),'stress_fixed':(-90,-12),'stress_stopped':(8,4)}
    for q in points:
        if q.get('strain') is None:continue
        changed=q['sdf_isomeric_smiles']!=control['sdf_isomeric_smiles'];label=q['label']
        ax.scatter(q['affinity']['pkd'],q['strain'],s=75,marker='^' if changed else 'o',color='#c14c39' if changed else '#167b80')
        ax.annotate(label.replace('_',' '),(q['affinity']['pkd'],q['strain']),xytext=offsets.get(label,(7,7)),textcoords='offset points',fontsize=9)
    ax.axhline(control['strain'],ls=':',color='0.6');ax.axvline(control['affinity']['pkd'],ls=':',color='0.6')
    ax.set_ylim(min(q['strain'] for q in points if q.get('strain') is not None)-4.,max(q['strain'] for q in points if q.get('strain') is not None)+4.)
    ax.set_xlabel('Predicted pKd (same model head)');ax.set_ylabel('MMFF94s self-strain proxy (kcal/mol)');ax.grid(alpha=.2)
    ax.set_title('Final outcomes: affinity gain can coexist with structural regression')
    fig.text(.1,.025,'Triangle: changed graph, evaluated against its own relaxed minimum.\nThese are surrogate measurements, not experimental affinity or a cross-graph total-energy ranking.',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(root/'final_tradeoff.png');plt.close(fig)
    source_root=Path(__file__).resolve().parents[1]
    write_json(root/'source_manifest_final.json',{str(path.relative_to(source_root)):file_hash(path)
        for folder in ['src/molsteer','scripts','tests','docs'] for path in (source_root/folder).rglob('*')
        if path.is_file() and path.suffix in ['.py','.md']})
    print(json.dumps(dict(reference_keys=list(reference),runs=[{k:v for k,v in s.items() if k not in ['requests','summary']} for s in summaries]),indent=2))


if __name__=='__main__':main()
