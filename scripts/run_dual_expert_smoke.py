#!/usr/bin/env python3
"""Run an explicitly identified live-LLM smoke test, or record missing credentials."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from molsteer.agents.config import load_config, SecretStore
from molsteer.agents.runtime import AgentRuntime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--packet',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    cfg=load_config(args.config);cfg.thinker.architecture='dual_expert';cfg.mode='api'
    example=cfg.repo_root/'examples/5i0b_A__5vef_M77/ligand_002/t_0.50'
    output=args.output or cfg.repo_root/'outputs'/('dual_expert_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    run_id='dual_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    summary={'kind':'DualExpertSmoke','live_llm':'not_run','gpu_continuation':'not_run',
             'research_role_coverage':[],'run_id':run_id}
    secrets=SecretStore(cfg)
    profiles={a.model for a in cfg.agents.values()}|set(cfg.thinker.experts.values())
    missing=[]
    for profile in profiles:
        provider=cfg.models[profile].provider
        try:secrets.get(provider)
        except ValueError:missing.append(provider)
    if missing:
        summary.update(status='skipped',reason='Required provider credentials unavailable',missing_providers=sorted(set(missing)))
    else:
        packet=json.loads((args.packet or example/'StatePacket.json').read_text(encoding='utf-8'))
        report=json.loads((args.report or example/'DiagnosticReport.json').read_text(encoding='utf-8'))
        state=AgentRuntime(cfg).run(packet,report,run_id=run_id)
        summary.update(status=state['status'],live_llm='run',checkpoint_path=state['checkpoint_path'],trace_path=state['trace_path'],
                       research_role_coverage=sorted({p['requested_by'] for p in state.get('research_packets',[])}),
                       version=state.get('reward_spec',{}).get('schema_version'),errors=state.get('errors',[]))
        summary['both_research_roles_exercised']=summary['research_role_coverage']==['biology','mathematics']
        # A smoke test reports actual coverage; it does not force unnecessary research in production.
        for key,name in [('biology_plan','BiologyPlan.json'),('mathematical_design','MathematicalDesign.json'),
                         ('research_packets','ResearchEvidencePackets.json'),('validation','Validation.json')]:
            (output/name).write_text(json.dumps(state.get(key),ensure_ascii=False,indent=2),encoding='utf-8')
        for language,title in [('en','Expert derivation record'),('zh-CN','双专家推导记录')]:
            lines=['# '+title,'',str(summary['status']),'',
                   'GPU continuation / 完整续生成: not_run','']
            for direction in state.get('biology_plan',{}).get('directions',[]):
                lines+=['## '+direction['direction_id'],direction['priority_reason'],direction['repair_predicate'],'']
            for direction in state.get('mathematical_design',{}).get('directions',[]):
                lines+=['## '+direction['direction_id'],direction['formula'],direction['derivation_summary'],
                        direction['gradient_path'],'Sources: '+', '.join(direction['source_ids']),'']
            (output/('Derivation.'+language+'.md')).write_text('\n'.join(lines),encoding='utf-8')
    (output/'SmokeResult.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
    return 1 if summary['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
