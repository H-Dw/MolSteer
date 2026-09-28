"""Independent post-generation evidence, never posthoc pose editing."""
import argparse
import json
from pathlib import Path
import math
import numpy as np
from rdkit import Chem, RDLogger
from molreader.io import load_stage
from molreader.packet import build_packet
from molreader.localized_report import make_localized_report, validate_localized_report
from molsteer.molreader.reporting import render_diagnostic
from molsteer.molreader import enrich_packet
from molsteer.common import write_json


def evaluate(root, config, arms=None, stages=None):
    root=Path(root)
    summaries=[]
    RDLogger.DisableLog('rdApp.warning')
    for arm in arms or ['unguided','selection','creativity']:
        previous=None
        for stage in stages or ['t_0.50','t_0.75','final']:
            source=root/arm/config['target_id']/f"ligand_{config['ligand_index']:03d}"/stage
            if not source.exists():continue
            contexts=[];gaps=[]
            for view in ['state','prediction','sdf']:
                try:contexts.append(load_stage(source,view,receptor=config['receptor'],posebusters=True))
                except Exception as exc:gaps.append(dict(view=view,error=str(exc)))
            packet=build_packet(contexts,history_packet=previous)
            packet=enrich_packet(packet,contexts)
            report=make_localized_report(packet)
            validate_localized_report(report,packet)
            dest=root/'evaluation'/arm/stage
            write_json(dest/'StatePacket.json',packet)
            write_json(dest/'DiagnosticReport.json',report)
            for lang in ['en','zh']:
                (dest/f'DiagnosticReport.{lang}.md').write_text(render_diagnostic(report,lang),encoding='utf-8')
            metrics={view:{m['metric_id']:dict(status=m['status'],values=m['values'],evidence_count=len(m['evidence']))
                     for m in packet['observations'] if m['view']==view} for view in ['state','prediction','sdf']}
            context=next(c for c in contexts if c.view=='prediction')
            result=dict(arm=arm,stage=stage,packet_id=packet['packet_id'],coverage=packet['coverage'],
                sanitized=context.mol is not None,smiles=Chem.MolToSmiles(context.mol) if context.mol is not None else None,
                metrics=metrics,gaps=gaps,risk_count=len(report['findings']))
            region=json.loads(Path(config['reward_programs']['creativity']).read_text())['region_atom_ids']
            result['local_atoms']=[dict(atom_id=i,element=context.atoms[i],formal_charge=context.charges[i]) for i in region]
            result['local_pairs']=[dict(atom_ids=[i,j],bond_order=float(context.orders[i,j]),distance=float(np.linalg.norm(context.coords[i]-context.coords[j])))
                for i in region for j in region if i<j]
            summaries.append(result)
            previous=packet
            print(json.dumps({k:v for k,v in result.items() if k not in ['metrics']}),flush=True)
    write_json(root/'quality_summary.json',dict(records=summaries,
        interpretation='MMFF local relaxation energy drop is a same-graph proxy, not binding energy; affinity heads are predictions. PoseBusters pass is limited to its reported checks.'))
    return summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();evaluate(a.root,json.loads(Path(a.config).read_text()))


if __name__=='__main__':main()
