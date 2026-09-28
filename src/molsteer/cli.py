import argparse
import json
from pathlib import Path
from molreader.io import load_stage
from molreader.localized_report import make_localized_report, validate_localized_report
from molsteer.common import write_json, file_hash
from molsteer.molreader import enrich_packet
from molsteer.molreader.reporting import render_diagnostic
from molsteer.molthinker.planner import derive
from molsteer.molthinker.reporting import render_derivation
from molsteer.molexecutor.offline import run_offline_trial
from molsteer.molthinker.creativity import derive as derive_program, render_program


def regenerate(input_root, baseline_root, output_root, knowledge):
    source, baseline, dest = [Path(p).resolve() for p in [input_root,baseline_root,output_root]]
    if dest == source or source in dest.parents or dest == baseline or baseline in dest.parents:
        raise ValueError('Output must be separate from generation inputs and baseline reports')
    records=[]
    selections=['5i0b_A__5vef_M77/ligand_002','2pqw_A__2rhy_MLZ/ligand_001']
    for selection in selections:
        for stage in ['t_0.50','t_0.75']:
            key=Path(selection)/stage
            base=json.loads((baseline/key/'StatePacket.json').read_text(encoding='utf-8'))
            contexts=[load_stage(source/key,view) for view in ['state','prediction','sdf']]
            packet=enrich_packet(base,contexts)
            report=make_localized_report(packet)
            validate_localized_report(report,packet)
            out=dest/key
            write_json(out/'StatePacket.json',packet)
            write_json(out/'DiagnosticReport.json',report)
            for lang in ['en','zh']:
                (out/f'DiagnosticReport.{lang}.md').write_text(render_diagnostic(report,lang),encoding='utf-8')
            record=dict(identity=packet['identity'],packet_id=packet['packet_id'],parent_packet_id=base['packet_id'],
                observation_count=len(packet['observations']),evidence_count=len(report['evidence_index']),
                risk_regions=len(report['findings']),coverage_root_causes=len(report['evidence_gaps']))
            if selection==selections[0] and stage=='t_0.50':
                spec=derive(packet,report,knowledge)
                write_json(out/'RewardSpec.json',spec)
                write_json(out/'RetrievalTrace.json',dict(source=spec['knowledge_source'],results=spec['retrieval']))
                movable=sorted({a for t in spec['terms'] if t['view']=='prediction' for a in t['hypothesis_atom_ids']})
                monitor=run_offline_trial(packet,spec,'prediction',movable)
                source_unchanged=all(file_hash(s['path'])==s['sha256'] for s in packet['provenance']['sources'].values())
                monitor['source_files_unchanged']=source_unchanged
                if not source_unchanged:raise ValueError('Generation source files changed')
                write_json(out/'ExecutionMonitor.json',monitor)
                for lang in ['en','zh']:
                    (out/f'RewardDerivation.{lang}.md').write_text(render_derivation(spec,monitor,lang),encoding='utf-8')
                record['offline_trial']=dict(penalty_before=monitor['penalty_before'],penalty_after=monitor['penalty_after'],
                    gradient_passed=monitor['numerical_gradient']['passed'],source_files_unchanged=source_unchanged)
            records.append(record)
            print(json.dumps(record),flush=True)
    write_json(dest/'run_summary.json',dict(records=records))


def main():
    p=argparse.ArgumentParser(description='MolSteer evidence-bound offline reward reasoning')
    sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo')
    for name in ['input-root','baseline-root','output-root','knowledge']:
        demo.add_argument('--'+name,required=True)
    think=sub.add_parser('think')
    for name in ['packet','report','knowledge','output']:
        think.add_argument('--'+name,required=True)
    think.add_argument('--mode',choices=['creativity','selection'],required=True,
                       help='Explicit historical deterministic mode; use the agents command for dynamic reward design')
    think.add_argument('--native-reference',help='Matched dense outcome reference; enables outcome-aware creativity')
    think.add_argument('--base-program',help='Existing live affinity program whose execution contract is retained')
    think.add_argument('--no-discrete-search',action='store_true')
    think.add_argument('--no-outcome-feedback',action='store_true')
    think.add_argument('--research-packet',help='Optional immutable ResearchPacket; no retrieval when omitted')
    think.add_argument('--research-mode',choices=['off','shadow','active'],default='off')
    think.add_argument('--research-hypothesis',action='append',default=[])
    think.add_argument('--research-allow-exploratory',action='store_true')
    think.add_argument('--research-fixed-influence',action='store_true')
    search=sub.add_parser('research-search',help='Optional literature retrieval; saves raw responses and source identifiers')
    search.add_argument('--request',required=True);search.add_argument('--output',required=True)
    publish=sub.add_parser('research-publish',help='Publish source-bound agent analysis without activating a reward')
    publish.add_argument('--store',required=True);publish.add_argument('--hypotheses',required=True);publish.add_argument('--analysis',required=True)
    agents=sub.add_parser('agents',help='Run the API-default LangGraph agent workflow')
    agents.add_argument('--config',help='Agent config JSON, default configs/agents.json')
    agents.add_argument('--packet',required=True,help='Enriched StatePacket JSON')
    agents.add_argument('--report',help='Optional validated DiagnosticReport JSON')
    agents.add_argument('--knowledge',help='Reviewed reward knowledge file')
    agents.add_argument('--offline',action='store_true',help='Explicit deterministic test mode; no API calls')
    agents.add_argument('--run-id',help='Unique audit run identifier')
    args=p.parse_args()
    if args.command=='agents':
        from molsteer.agents.config import load_config
        from molsteer.agents.runtime import AgentRuntime
        config=load_config(args.config)
        if args.offline:
            config.mode='offline'
        packet=json.loads(Path(args.packet).read_text(encoding='utf-8'))
        report=json.loads(Path(args.report).read_text(encoding='utf-8')) if args.report else None
        runtime=AgentRuntime(config,knowledge_path=args.knowledge)
        result=runtime.run(packet,report,run_id=args.run_id,execute=False)
        print(json.dumps({key:result.get(key) for key in ['run_id','status','route','errors','checkpoint_path']},ensure_ascii=False))
        if result.get('status')=='failed':
            raise SystemExit(1)
    elif args.command=='research-search':
        from molsteer.molthinker.research import ResearchStore,EuropePMC
        request=json.loads(Path(args.request).read_text(encoding='utf-8'));store=ResearchStore(args.output,request)
        for query in request['queries']:store.search(EuropePMC(),query,request.get('page_size',5))
    elif args.command=='research-publish':
        from molsteer.molthinker.research import ResearchStore
        store=ResearchStore.open_pending(args.store)
        store.publish(json.loads(Path(args.hypotheses).read_text(encoding='utf-8')),json.loads(Path(args.analysis).read_text(encoding='utf-8')))
    elif args.command=='demo':
        regenerate(args.input_root,args.baseline_root,args.output_root,args.knowledge)
    else:
        packet=json.loads(Path(args.packet).read_text(encoding='utf-8'))
        report=json.loads(Path(args.report).read_text(encoding='utf-8'))
        if args.mode=='creativity' and ('outcome_context' in packet.get('steering',{}) or args.native_reference):
            if not args.native_reference or not args.base_program:
                raise ValueError('Outcome-aware creativity requires --native-reference and --base-program; do not silently ignore outcome evidence')
            from molsteer.molthinker.outcomes import build_outcome_program
            base=json.loads(Path(args.base_program).read_text(encoding='utf-8'))
            spec=build_outcome_program(base,packet,report,args.knowledge,args.native_reference,
                discrete=not args.no_discrete_search,feedback=not args.no_outcome_feedback)
        else:
            spec=derive_program(packet,report,args.knowledge,args.mode)
        if args.research_mode!='off':
            if not args.research_packet:raise ValueError('Research mode requires --research-packet')
            from molsteer.molthinker.research_rewards import attach_research
            spec=attach_research(spec,args.research_packet,args.research_hypothesis,mode=args.research_mode,
                dynamic=not args.research_fixed_influence,allow_exploratory=args.research_allow_exploratory)
        write_json(args.output,spec)
        for lang in ['en','zh']:
            if spec.get('evaluator')=='outcome_aware':
                from molsteer.molthinker.outcomes import render_outcome_program
                content=render_outcome_program(spec,lang)
            else:content=render_program(spec,language=lang)
            Path(args.output).with_suffix(f'.{lang}.md').write_text(content,encoding='utf-8')


if __name__=='__main__':main()
