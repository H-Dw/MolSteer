"""Read final molecules from a completed live API sweep, without changing them."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def observation(packet, name):
    item = next((o for o in packet['observations']
                 if o['view']=='prediction' and o['metric_id']==name), None)
    return item['values'] if item and item['status']=='ok' else {}


def measure(packet):
    geometry = observation(packet, 'mmff_local_geometry')
    bonds = geometry.get('bonds', [])
    deviations = [b['relative_deviation'] for b in bonds]
    chemistry = observation(packet, 'chemistry_context')
    atoms = {a['atom_id']:a for a in chemistry.get('atoms', [])}
    selected = next((b for b in bonds if sorted(b['atom_ids'])==[4,12]), None)
    return dict(
        sanitized=observation(packet,'valence').get('sanitized'),
        component_count=observation(packet,'connectivity').get('component_count'),
        protein_clash_count=observation(packet,'protein_clashes').get('clash_count'),
        intra_clash_count=observation(packet,'intramolecular_clashes').get('clash_count'),
        qed=observation(packet,'qed').get('qed'),
        sa_score=observation(packet,'sa_score').get('sa_score'),
        strain_proxy_kcal_mol=observation(packet,'mmff_strain').get('strain_proxy_kcal_mol'),
        affinity=observation(packet,'affinity'),
        bond_relative_deviation_rms=math.sqrt(sum(v*v for v in deviations)/len(deviations)) if deviations else None,
        bond_max_abs_relative_deviation=max(map(abs,deviations),default=None),
        alert_count=observation(packet,'structural_alerts').get('alert_count'),
        atom_4=atoms.get(4),atom_12=atoms.get(12),bond_4_12=selected,
        chemistry=dict(atoms=[{key:a.get(key) for key in ('atom_id','element','formal_charge')}
                             for a in chemistry.get('atoms',[])],
                       bonds=[list(b['atom_ids'])+[b['bond_order']]
                              for b in chemistry.get('bonds',[]) if b['bond_order']]))


def main():
    from molreader.io import load_stage
    from molreader.packet import build_packet
    from molreader.localized_report import make_localized_report
    from molsteer.molreader import enrich_packet
    from molsteer.molreader.reporting import render_diagnostic
    from rdkit import Chem
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root',type=Path,default=REPO/'test/5i0b_t050_api_weights_20260930')
    parser.add_argument('--attempt',default='run_02')
    args=parser.parse_args()
    root=args.test_root.resolve()
    if not root.is_relative_to(REPO/'test') or Path(args.attempt).name!=args.attempt:
        parser.error('Analysis outputs must stay under test')
    run=root/args.attempt
    sweep=read(run/'weight_summary.json')
    if sweep['status']!='completed':raise ValueError('The live sweep has not completed')
    api=read(run/'api_status.json')
    if api['mode']!='api' or api['status']!='validated':raise ValueError('Validated actual API workflow required')
    initial=read(run/'StatePacket.json')
    initial_metrics=measure(initial)
    target=initial['identity']['target_id']
    receptor=root/'inputs/5i0b_A_rec_5vef_m77_lig_tt_min_0_pocket10.pdb'
    executions={'unguided':sweep['native'],**{f"weight_{r['weight']}":r for r in sweep['weights']}}
    rows=[]
    for name,execution in executions.items():
        arm=run/name
        stage=(arm if name=='unguided' else arm/'run')/target/'ligand_000/final'
        contexts=[load_stage(stage,v,receptor=receptor) for v in ('state','prediction')]
        packet=enrich_packet(build_packet(contexts),contexts)
        report=make_localized_report(packet)
        metrics=measure(packet)
        write(arm/'StatePacket.final.json',packet)
        write(arm/'DiagnosticReport.final.json',report)
        (arm/'DiagnosticReport.final.zh.md').write_text(render_diagnostic(report,'zh'),encoding='utf-8')
        try:
            mol=Chem.MolFromMolFile(str(stage/'ligand.sdf'),sanitize=False,removeHs=False) if (stage/'ligand.sdf').is_file() else None
        except OSError:
            mol=None
        smiles=None
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                smiles=Chem.MolToSmiles(mol)
            except Exception:pass
        trace_path=(arm if name=='unguided' else arm/'run')/'guidance_trace.jsonl'
        trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
        reasons=Counter(reason for r in trace for attempt in r.get('proposal_attempts',[])
                        for reason in attempt.get('failures',[]))
        events=Counter(r.get('guidance_unavailable') for r in trace if r.get('guidance_unavailable'))
        rows.append(dict(arm=name,weight=execution.get('weight',0),execution=execution,
            final=metrics,smiles=smiles,final_sdf=str(stage/'ligand.sdf'),
            initial_hypothesis_preserved=metrics['chemistry']==initial_metrics['chemistry'],
            gradient_preflight=read(arm/'gradient_preflight.json')['passed'] if name!='unguided' else None,
            first_guidance_time=next((r['t'] for r in trace if r.get('guidance_active')),None),
            first_gradient_time=next((r['t'] for r in trace if 'gradient_norm' in r),None),
            proposal_failure_counts=dict(reasons),guidance_unavailable_counts=dict(events),
            final_assessment=report['assessment']))
        print(json.dumps({'arm':name,'status':'terminal_reader_complete'}),flush=True)
    receipts=[json.loads(line) for line in (run/'api_requests.jsonl').read_text().splitlines()]
    completed=[r for r in receipts if r['event']=='request_completed']
    usage={key:sum(u.get(key,0) for r in completed for u in r.get('usage',[]))
           for key in ('input_tokens','output_tokens','total_tokens')}
    program=read(run/'RewardProgram.api.json')
    design=program.get('expert_spec',{}).get('mathematical_design',{})
    gradients={r['weight']:r['execution'].get('initial_gradient_norm') for r in rows if r['weight']}
    scale_verified=(isinstance(gradients.get(1),(int,float)) and gradients[1]>0
                    and all(isinstance(v,(int,float)) and math.isclose(v/w,gradients[1],rel_tol=1e-5,abs_tol=1e-10)
                            for w,v in gradients.items()))
    summary=dict(status='completed',target=target,molecule_index=0,initial_time=.5,
        initial=initial_metrics,api=api,api_requests_by_agent=dict(Counter(r['agent'] for r in completed)),
        model_profiles=read(run/'agents.api.json')['models'],
        token_usage=usage,program_id=program['program_id'],evaluator=program['evaluator'],
        mathematical_design=design,monitor_enabled=False,graph_review_enabled=False,arms=rows,
        initial_control_weight_scaling_verified=scale_verified,
        data_integrity=read(root/'source_integrity_after.json'),
        scope='One molecule, one checkpoint and matched native randomness; model affinity is not experimental affinity')
    write(run/'experiment_analysis.json',summary)
    lines=['# 5i0b 单分子真实 API 与五档奖励权重实验','',
        f'目标：{target}；分子：molecule_000；t=0.50→1.00。MolMonitor 和图复核关闭。','',
        f"真实 API 完成请求：{len(completed)}；按 Agent 分布：{summary['api_requests_by_agent']}。",
        f"模型：{', '.join(sorted({p['model'] for p in summary['model_profiles'].values()}))}（服务器 OPENROUTER_API_KEY）。",
        f"奖励：{program['program_id']}（{program['evaluator']}）。全局 Rw=wR，固定执行力度与位移预算。",'',
        '| 权重 | t=0.5梯度范数 | 有梯度步 | 有效位移步 | 接受步 | 累计注入最大 Å | 最终化学有效 | 连通分量 | 蛋白碰撞 | MMFF应变代理 kcal/mol | 预测pKd |',
        '|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|']
    def number(v):return f'{v:.6g}' if isinstance(v,(int,float)) else '未测得'
    for row in rows:
        e=row['execution'];m=row['final']
        lines.append('| '+' | '.join(map(str,[row['weight'],number(e.get('initial_gradient_norm')),e['gradient_steps'],e.get('effective_steps',0),e['accepted_steps'],
            number(e['max_injected_path_angstrom']),m['sanitized'],m['component_count'],m['protein_clash_count'],
            number(m['strain_proxy_kcal_mol']),number(m['affinity'].get('pkd'))]))+' |')
    lines+=['','所有引导分支通过 t=0.50 的实时有限差分检查。最终分子由独立 MolReader 重新评价，提交坐标不做事后最小化。',
        '',f't=0.50 控制梯度按全局权重成比例缩放：{scale_verified}。接受步包括零位移提议，因此单列有效位移步。',
        '',f"data 完整性检查：{summary['data_integrity']['passed']}；文件数：{summary['data_integrity']['data_file_count']}。",
        '', '历史局部探针 4–12 与当前图适用性（不代表奖励的全部目标）：',
        '', '| 权重 | 终态slot 4元素 | 4–12实际键长 Å | 当前图MMFF参考 Å | 当前图相对偏差 | QED | SA |',
        '|---:|:---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['final'];bond=m.get('bond_4_12') or {};atom=m.get('atom_4') or {}
        lines.append('| '+' | '.join(map(str,[row['weight'],atom.get('element','未测得'),
            number(bond.get('distance_angstrom')),number(bond.get('reference_angstrom')),
            number(bond.get('relative_deviation')),number(m['qed']),number(m['sa_score'])]))+' |')
    audit=design.get('design_audit',{})
    lines+=['', '奖励以本轮真实 API 提交的表达式、函数谱系与适用性合同为准，完整保存在 RewardProgram.api.json。局部键偏差与全分子应变同时出现，不能据此证明应变由这一条键造成。',
        '', '五档奖励固定不重写。终态必须用当前化学图独立复评，不能把类别改变后的旧代理下降当作原修复假设成立。',
        '', '本表为一个分子的机制测试。不同化学图不能用固定键长参考或 MMFF 总能量直接排名。',
        '亲和力是生成器预测值，未运行实验测定或独立对接。五档参数不能据此得到跨靶点最优权重。']
    if audit:
        lines+=['', f"奖励审计范围：{audit['execution_scope']}；图变化策略：{audit['graph_policy']}。",
                '', '| 生物因素 | 处理方式 | 相关方向 |', '|---|---|---|']
        for factor in program['expert_spec']['biology_plan'].get('factor_assessment',[]):
            lines.append('| '+factor['factor']+' | '+factor['disposition']+' | '+', '.join(factor['direction_ids'])+' |')
    lines+=['', '引导不可用原因（区分零梯度与参考失效）：']
    for row in rows:
        if row['weight']:lines.append(f"- 权重 {row['weight']}：{row['guidance_unavailable_counts']}")
    (run/'experiment_report.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'analysis_completed','output':str(run/'experiment_analysis.json')}),flush=True)


if __name__=='__main__':main()
