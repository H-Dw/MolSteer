from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from .core import write_json,clean
from .io import load_stage,load_config
from .metrics import METRICS
from .packet import compute_metric,build_packet,validate_packet
from .diagnostic import make_report,render_markdown,validate_report


def common(parser):
    parser.add_argument('--stage-dir',required=True,type=Path)
    parser.add_argument('--receptor',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--previous-stage',type=Path)
    parser.add_argument('--posebusters',action='store_true')
    parser.add_argument('--prepared-receptor',dest='prepared_receptor_sdf',type=Path)
    parser.add_argument('--vina-result',type=Path)


def context(args,view):
    previous=load_stage(args.previous_stage,view,receptor=args.receptor,config=args.config) if args.previous_stage else None
    return load_stage(args.stage_dir,view,receptor=args.receptor,config=args.config,previous=previous,
                      posebusters=args.posebusters,prepared_receptor_sdf=args.prepared_receptor_sdf,vina_result=args.vina_result)


def metric_main(name=None):
    parser=argparse.ArgumentParser(description='Calculate one MolReader metric and emit a provenance envelope')
    common(parser)
    if name is None:parser.add_argument('metric',choices=METRICS)
    parser.add_argument('--view',choices=['state','prediction','sdf'],default='prediction')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    data=compute_metric(context(args,args.view),name or args.metric)
    if args.output:write_json(args.output,data)
    else:print(json.dumps(clean(data),indent=2,allow_nan=False))
    return 0 if data['status']!='error' else 1


def pack_main():
    parser=argparse.ArgumentParser(description='Pack independently calculated molecular observations into a StatePacket')
    common(parser)
    parser.add_argument('--views',nargs='+',choices=['state','prediction','sdf'],default=['state','prediction','sdf'])
    parser.add_argument('--metrics',nargs='+',choices=METRICS)
    parser.add_argument('--metric-results',nargs='*',type=Path,default=[])
    parser.add_argument('--history-packet',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    contexts=[context(args,v) for v in args.views]
    history=json.loads(args.history_packet.read_text()) if args.history_packet else None
    packet=build_packet(contexts,args.metrics,args.metric_results,history)
    write_json(args.output,packet)
    print(json.dumps({'packet':str(args.output),'coverage':packet['coverage']}))


def batch_main():
    parser=argparse.ArgumentParser(description='Read only existing generation artifacts; write a separate diagnostic output tree')
    parser.add_argument('--input-root',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--posebusters',action='store_true')
    parser.add_argument('--metrics',nargs='+',choices=METRICS)
    parser.add_argument('--views',nargs='+',choices=['state','prediction','sdf'],default=['state','prediction','sdf'])
    parser.add_argument('--layout',choices=['localized','legacy'],default='localized')
    args=parser.parse_args()
    if args.output_root.resolve()==args.input_root.resolve() or args.input_root.resolve() in args.output_root.resolve().parents:
        raise ValueError('Diagnostic output must be outside the immutable input tree')
    stage_dirs=list(args.input_root.glob('*/ligand_*/t_*'))+list(args.input_root.glob('*/ligand_*/final'))
    if not stage_dirs:raise ValueError('No stage directories discovered')
    stage_dirs.sort(key=lambda p:(p.parent.parent.name,p.parent.name,1.0 if p.name=='final' else float(p.name[2:])))
    config=load_config(args.config);records=[];previous={};histories={}
    for stage in stage_dirs:
        key=(stage.parent.parent.name,stage.parent.name);dest=args.output_root/stage.relative_to(args.input_root)
        try:
            contexts=[];view_errors=[]
            for view in args.views:
                try:
                    c=load_stage(stage,view,config=config,previous=previous.get((key,view)),posebusters=args.posebusters)
                    contexts.append(c)
                except Exception as exc:
                    view_errors.append(dict(view=view,error=f'{type(exc).__name__}: {exc}'))
            if not contexts:raise ValueError(str(view_errors))
            packet=build_packet(contexts,args.metrics,history_packet=histories.get(key))
            if view_errors:packet['limitations'].extend('View unavailable: '+str(e) for e in view_errors)
            if args.layout=='localized':
                from .localized_report import make_localized_report,render_localized_markdown
                report=make_localized_report(packet);markdown=render_localized_markdown(report)
            else:report=make_report(packet);markdown=render_markdown(report)
            write_json(dest/'StatePacket.json',packet);write_json(dest/'DiagnosticReport.json',report)
            (dest/'DiagnosticReport.md').write_text(markdown,encoding='utf-8')
            row=dict(target_id=key[0],ligand_id=key[1],stage=stage.name,stage_t=packet['identity']['stage_t'],
                     packet_id=packet['packet_id'],risk_groups=len(report['findings']),view_errors=len(view_errors),**packet['coverage'])
            for view in args.views:
                for m in packet['observations']:
                    if m['view']!=view:continue
                    if m['metric_id']=='affinity' and view=='prediction':row.update(m['values'])
                    if m['metric_id']=='protein_clashes':row[view+'_protein_clashes']=m['values'].get('clash_count')
                    if m['metric_id']=='mmff_strain':row[view+'_strain_kcal_mol']=m['values'].get('strain_proxy_kcal_mol')
                    if m['metric_id']=='valence':row[view+'_sanitized']=m['values'].get('sanitized')
                    if m['metric_id']=='decode_consistency' and view=='prediction':row['decode_difference_count']=m['values'].get('difference_count')
            records.append(row)
            for c in contexts:previous[(key,c.view)]=c
            histories[key]=packet
            print(json.dumps(row),flush=True)
        except Exception as exc:
            error=dict(target_id=key[0],ligand_id=key[1],stage=stage.name,read_error=f'{type(exc).__name__}: {exc}')
            write_json(dest/'read_error.json',error);records.append(error);print(json.dumps(error),flush=True)
    write_json(args.output_root/'run_summary.json',{'stage_count':len(records),'failed_stages':sum('read_error' in r for r in records),'records':records})
    keys=list(dict.fromkeys(k for row in records for k in row))
    with (args.output_root/'stage_summary.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter='\t');w.writeheader();w.writerows(records)
    if any('read_error' in r for r in records):raise SystemExit(1)


def report_main():
    parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--layout',choices=['localized','legacy'],default='localized')
    args=parser.parse_args();packet=json.loads(args.packet.read_text(encoding='utf-8'))
    if args.layout=='localized':
        from .localized_report import make_localized_report,render_localized_markdown
        report=make_localized_report(packet);markdown=render_localized_markdown(report)
    else:report=make_report(packet);markdown=render_markdown(report)
    write_json(args.output,report);args.output.with_suffix('.md').write_text(markdown,encoding='utf-8')


def validate_main():
    parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);parser.add_argument('--report',type=Path)
    args=parser.parse_args();packet=json.loads(args.packet.read_text(encoding='utf-8'));validate_packet(packet)
    if args.report:validate_report(json.loads(args.report.read_text(encoding='utf-8')),packet)
    print('Validated packet identity, metric coverage, atom references and diagnostic evidence.')


def main():
    functions={'metric':metric_main,'pack':pack_main,'batch':batch_main,'report':report_main,'validate':validate_main}
    if len(sys.argv)<2 or sys.argv[1] not in functions:
        print('Usage: molreader {metric,pack,batch,report,validate} --help');return
    command=sys.argv.pop(1);functions[command]()

if __name__=='__main__':main()
