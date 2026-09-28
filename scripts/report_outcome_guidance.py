"""Report matched outcome tests, including failed revisions and rejected searches."""
import argparse,csv,json
from collections import Counter
from pathlib import Path
from molsteer.common import write_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();context=json.loads((root/'evaluation/ReaderComparisonContext.json').read_text())
    cases={r['label']:r for r in context['observations']};base=cases['native_final']
    records=[];traces={};searches={};revisions=[]
    for label,r in cases.items():
        ff=r['forcefield'];aff=r['model_predictions']['affinity'];cmp=context['comparisons'][label]
        records.append(dict(label=label,**aff,strain_kcal_mol=ff['strain_kcal_mol'],
            strain_delta=ff['strain_kcal_mol']-base['forcefield']['strain_kcal_mol'],
            pkd_delta=aff['pkd']-base['model_predictions']['affinity']['pkd'],
            vina=r['docking']['vina']['score_only_kcal_mol'],vinardo=r['docking']['vinardo']['score_only_kcal_mol'],
            vina_local=r['docking']['vina']['local_optimized_kcal_mol'],vinardo_local=r['docking']['vinardo']['local_optimized_kcal_mol'],
            same_graph=cmp['same_slot_graph'],same_stereochemistry=cmp['same_isomeric_smiles'],
            rmsd_angstrom=cmp['world_rmsd_angstrom'],posebusters_failed=len(r['metrics']['posebusters']['values']['failed_checks'])))
    for label in ('outcome_continuous','outcome_discrete','outcome_adaptive'):
        path=root/'continuations'/label/'creativity/outcome_trace.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()];traces[label]=rows
        trials=[dict(time=r['next_t'],**q) for r in rows if 'chemical_search' in r for q in r['chemical_search']['trials']]
        searches[label]=dict(trials=trials,trial_count=len(trials),accepted=sum(r['categorical_commit'] for r in rows),
            enumerations=[dict(time=r['next_t'],**r['chemical_search']['enumeration']) for r in rows if 'chemical_search' in r],
            rejection_counts=dict(Counter(f for q in trials for f in q.get('failures',[]))),
            realized_changed_graph_trials=sum('actual_smiles' in q and 'no_realized_endpoint_graph_change' not in q.get('failures',[]) for q in trials))
        revisions += [dict(label=label,time=r['next_t'],**r['reward_revision']) for r in rows if 'reward_revision' in r]
    audit=json.loads((root/'validation/runtime_audit.json').read_text())
    gradient=json.loads((root/'validation/continuous_gradient.json').read_text())
    reference=json.loads((root/'native_reference.json').read_text())
    continuous=cases['outcome_continuous'];adaptive=cases['outcome_adaptive']
    c=next(r for r in records if r['label']=='outcome_continuous');d=next(r for r in records if r['label']=='outcome_adaptive')
    search_count=searches['outcome_discrete']['trial_count']
    search_schedule=', '.join(f"{r['next_t']:.2f}: {len(r['chemical_search']['trials'])}" for r in traces['outcome_discrete'] if 'chemical_search' in r)
    dominated=c['pkd']>d['pkd'] and c['strain_kcal_mol']<d['strain_kcal_mol'] and c['vina']<d['vina'] and c['vinardo']<d['vinardo']
    feedback=dict(kind='MolThinkerOutcomeFeedback',context_id=context['packet_id'],
        candidate='outcome_adaptive',comparison='outcome_continuous',route='revise_objective_policy' if dominated else 'review_tradeoff',
        promote_automatically=False,eta_escalation_requested=False,
        reason='Revision underperformed the fixed continuous program on primary head, strain, Vina and Vinardo' if dominated else 'Inspect paired objective tradeoffs',
        comparison_values=dict(candidate=d,reference=c),actual_revision_events=revisions,
        failed_search_summary={k:{f:v[f] for f in ('trial_count','accepted','realized_changed_graph_trials','rejection_counts')} for k,v in searches.items()},
        local_residuals=dict(atom_force_norms=continuous['forcefield']['atom_force_norms'],
            polar_context=continuous['polar_context'],mmff_local_geometry=continuous['metrics']['mmff_local_geometry']['values']),
        reasoning_requests=['Distinguish transient graph-conditioned strain regression from persistent terminal damage before reweighting.',
            'Retain head-independent interface quality when resolving gradient conflict; inspect projection suppression instead of increasing eta.',
            'Compare revised objective policies on matched suffixes and preserve a nondominated fixed-policy fallback.',
            'Investigate why valid categorical hypotheses produce invalid or inferior model endpoints; do not count rejected trials as new molecules.'],
        limitations=['Single checkpoint experiment; no seed sweep or general success-rate estimate',
            'Vina is used for control and is not held out; Vinardo is related',
            'No calibrated binding or pH population estimate'],
        provenance=dict(context_sha256=file_hash(root/'evaluation/ReaderComparisonContext.json'),
            traces={k:file_hash(root/'continuations'/k/'creativity/outcome_trace.jsonl') for k in traces}))
    write_json(root/'MolThinkerFeedback.json',feedback);write_json(root/'chemical_search_audit.json',searches)
    write_json(root/'comparison.json',dict(records=records,revision_events=revisions,runtime_audit=audit,gradient_check=gradient))
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    labels=dict(native_final=('无引导','Unguided'),legacy_fixed300=('旧奖励，固定 η=300','Legacy fixed η=300'),
        outcome_continuous=('新连续奖励','Continuous outcomes'),outcome_discrete=('新连续奖励 + 图搜索','Continuous + graph search'),
        outcome_adaptive=('新奖励 + 图搜索 + 动态修订','Continuous + graph + revision'),previous_adaptive=('此前自适应力度','Previous adaptive strength'))
    for lang in ('zh-CN','en'):
        zh=lang=='zh-CN';choose=lambda cn,en:cn if zh else en
        lines=['# '+choose('MolThinker 改进与实际续生成验证','MolThinker outcome-aware implementation and real continuation tests'),'',
            choose('新连续奖励改善了结构应变，但没有全面超过旧奖励。主动图搜索确实进入了模型推理；本例未有候选通过验收。动态修订发生两次，终态反而不如固定新奖励，因此本轮不将该修订策略提升为优选方案。',
            'Continuous outcome rewards improved strain but did not outperform the legacy reward on every objective. Active graph trials entered the live model; none passed acceptance in this case. Two objective revisions ran, but their final result underperformed the fixed new reward and is not promoted.'),'',
            choose('对象：5i0b_A__5vef_M77 / ligand_002；原始 t=0.50 完整检查点，续生成至 1.0。诊断仍只来自 t=0.50，额外使用配对无引导轨迹与终态证据。所有本轮引导分支 η=300，单步每原子上限 0.04 Å、累计注入路径上限 2 Å；没有随机种子扫描。',
            'Subject: 5i0b_A__5vef_M77 / ligand_002, authentic full t=0.50 runtime continued to 1.0. Diagnosis remains at t=0.50; matched native trajectories and terminal evidence are additional inputs. All new guided arms use η=300, 0.04 Å per-atom step cap and 2 Å cumulative injected path cap. No seed sweep.'),'',
            '## '+choose('配对终态结果','Matched terminal outcomes'),'',
            '| '+choose('方案','Arm')+' | pKd ↑ | MMFF strain ↓ (kcal/mol) | Vina ↓ | Vinardo ↓ |',
            '|---|---:|---:|---:|---:|']
        for row in records:lines.append(f"| {labels[row['label']][0 if zh else 1]} | {row['pkd']:.5f} | {row['strain_kcal_mol']:.4f} | {row['vina']:.3f} | {row['vinardo']:.3f} |")
        pct=100*(base['forcefield']['strain_kcal_mol']-c['strain_kcal_mol'])/base['forcefield']['strain_kcal_mol']
        lines+=['',choose(f"新连续奖励相对无引导：应变下降 {pct:.2f}%，pKd 增加 {c['pkd_delta']:.5f}。Vina 改善仅 0.019 kcal/mol、Vinardo 仅 0.028 kcal/mol，不能据此宣称结合力有实质提升。相对旧固定奖励，应变进一步降低，但 pKd 和原位经验评分较差，属于目标取舍。",
            f"Continuous reward versus unguided: strain decreases {pct:.2f}%, pKd increases {c['pkd_delta']:.5f}. Vina and Vinardo gains are only 0.019 and 0.028 kcal/mol; these do not establish meaningful binding improvement. Compared with the legacy fixed reward, strain is lower but pKd and original-pose empirical scores are worse: an objective tradeoff."),'',
            choose('表中应变来自保存的 SDF：固定重原子优化氢，与各自局部最小值比较。实时奖励使用高精度张量和冻结参照，因此其终态应变 13.330555 与 SDF 的 13.323016 有小差异。Vina/Vinardo 为原位评分；单独优化副本的结果不替代生成分子。Vina 已参与引导验收，不能再称作留出验证；Vinardo 与之相关。',
                'Table strain uses saved SDFs, hydrogen-only relaxation and a local reference. Live reward uses high-precision tensors and frozen references, giving 13.330555 versus SDF 13.323016. Vina/Vinardo are score-only. Optimized copies do not replace generated molecules. Vina is part of control, not held-out validation; Vinardo is related.'),'',
            '![Matched outcome comparison](comparison.png)','',
            '## '+choose('四项代码改动','Four implemented changes'),'',
            choose('1. 同时间原生基线与自然修复账本。旧图绑定诊断只作历史证据；例如原始 [1,10,14] 的 N/C/C 演化为终态 N/N/N 已由无引导产生，不能全部归功于奖励。纯原生演化的增量奖励为零。',
                '1. Same-time native references and a self-correction ledger. Obsolete graph-bound findings remain historical; the original [1,10,14] N/C/C to terminal N/N/N change already occurs unguided and is not credited to guidance. Pure native evolution receives zero incremental reward.'),
            choose('2. 完整连续 MMFF 应变梯度、方向接触和平滑埋藏未满足极性代价。保留硬化学与碰撞检查，不将通过告警阈值视为充分合理。',
                '2. Continuous full-MMFF strain derivatives, directional contacts and smooth buried-unsatisfied-polar cost. Hard chemistry/clash checks remain; passing an alert threshold does not establish physical plausibility.'),
            choose('3. 实际枚举、映射并注入当前态分类张量，重新执行模型，再根据预测终点验收；保留拒绝原因。微观状态不是 pH 优势态预测。',
                '3. Enumerate, map and inject categorical current-state hypotheses, rerun the model, and accept only on actual predicted endpoints. Rejection reasons are retained; microstate hypotheses are not pH population predictions.'),
            choose('4. 测量真实梯度冲突，使用独立于 head 的 Vina 反证，保存父子奖励程序。反馈修改目标权重而不增加 η，并在终态评估中保留失败修订。',
                '4. Measure actual gradient conflict, use head-independent Vina counterevidence, and retain parent/child reward programs. Feedback changes objective weights without increasing η; failed revisions remain visible.'),'',
            '## '+choose('实际图搜索与修订','Actual chemical search and revision'),'',
            choose(f'两个图搜索分支分别执行 {search_schedule} 次模型候选试探，每个分支共 {search_count} 次，包含互变异构体、元素替换和键级候选。均为 0 次提交。部分候选确实改变模型预测的化学图，但未通过奖励/应变/Vina/有效性联合检查；有些候选被原生模型映回同一化学图。已剔除仅切换芳香/Kekule 表示的伪图变化。',
                f'Both search arms executed live-model trials at time:count {search_schedule}, {search_count} total per arm, including tautomers, atom substitutions and bond changes. Neither committed a candidate. Some trials changed the predicted graph but failed joint reward/strain/Vina/validity checks; others mapped back to the original endpoint graph. Aromatic/Kekule representation-only changes are excluded.'),
            choose('因此连续奖励与加图搜索分支的最终张量和分子相同。不能声称本轮主动搜索已经找到了更优的新分子。完整候选、实际预测结构和拒绝理由见 chemical_search_audit.json。',
                'The continuous and graph-search arms therefore end identically. This run does not demonstrate a better new chemical identity. Full candidates, actual endpoint identities and rejection reasons are in chemical_search_audit.json.'),'']
        for event in revisions:
            evidence=event['evidence']
            if 'oracle' in evidence:
                current,native=evidence['oracle'],evidence['native_oracle']
                lines.append(choose(
                    f"- t={event['time']:.2f}：相对同时间无引导，pKd 增加 {evidence['affinity_delta']:.5f}，但应变为 {current['strain']:.4f} 对 {native['strain']:.4f} kcal/mol（恶化 {evidence['strain_delta']:.4f}）。Vina 为 {current['vina']:.3f} 对 {native['vina']:.3f}；本次触发原因是应变反证。",
                    f"- t={event['time']:.2f}: pKd gains {evidence['affinity_delta']:.5f}, but strain is {current['strain']:.4f} versus matched-native {native['strain']:.4f} kcal/mol (regression {evidence['strain_delta']:.4f}). Vina is {current['vina']:.3f} versus {native['vina']:.3f}; strain counterevidence triggers this revision."))
            else:
                recent=[v for v in traces[event['label']] if v['next_t']<=event['time'] and 'gradient' in v][-3:]
                cosines=', '.join(f"{v['gradient']['affinity_structure_cosine']:.4f}" for v in recent)
                lines.append(choose(f"- t={event['time']:.2f}：连续 3 步 head/应变梯度冲突，余弦依次为 {cosines}，均低于 −0.25。",
                    f"- t={event['time']:.2f}: head/strain gradient conflict persists for three steps; cosines are {cosines}, all below −0.25."))
        lines+=['',choose('两次修订将 affinity 权重从 1 降到 0.5，再降到 0.25；应变权重从 1 升至 1.25，再至 1.5625。终态 pKd=5.41624、Vina=−5.744，均不如新固定连续奖励，应变也没有额外改善。该规则对瞬时应变反证和末段梯度冲突反应过强的可能性需要后续成对验证，不能由本次单例确定因果。',
            'The two revisions reduce affinity weight 1→0.5→0.25 and increase strain weight 1→1.25→1.5625. Final pKd=5.41624 and Vina=−5.744 are worse than the fixed continuous arm, without additional strain improvement. Overreaction to transient strain counterevidence and late gradient conflict is a hypothesis for further paired testing, not a causal conclusion from one case.'),'',
            choose('MolThinkerFeedback.json 已汇总失败修订、原始目标数值、搜索拒绝原因和局部残余证据，路由为 revise_objective_policy；不请求提高 η，也不自动提升该候选。',
                'MolThinkerFeedback.json summarizes failed revisions, original objective values, rejected hypotheses and local residual evidence. Its route is revise_objective_policy; it requests neither η escalation nor automatic candidate promotion.'),'',
            '## '+choose('局部改善与残余问题','Local changes and residual concerns'),'',
            choose('所有终态保持相同化学图及立体结构；MW、logP、TPSA、QED 不变。新连续奖励的重原子原位 RMSD 为 0.16854 Å。N1–N14 从 1.04286 Å 增至 1.07465 Å，MMFF 参照为 1.140 Å，仍压缩约 5.73%；N14/N1 残余力从 227.70/199.72 降至 138.78/119.39 kcal/mol/Å。此类残余可低于旧的 10% 键长告警阈值，但仍有连续改善空间。',
                'All finals retain the same graph and stereochemistry; MW, logP, TPSA and QED are unchanged. Continuous guidance gives 0.16854 Å heavy-atom world RMSD. N1–N14 length increases 1.04286→1.07465 Å against the 1.140 Å MMFF reference, still about 5.73% compressed. N14/N1 forces decrease 227.70/199.72→138.78/119.39 kcal/mol/Å. Residual pressure persists below the old 10% bond-alert threshold.'),
            choose('ProLIF 仍识别 1 个配体受体氢键和 1 个配体供体氢键。N13–LEU398 距离由 3.11045 降至 3.00325 Å，DHA 角由 161.77° 增至 164.92°；N3–GLU396 由 3.11286 降至 2.98854 Å，角由 140.64° 增至 142.94°。接触几何改善没有产生新的氢键类型。',
                'ProLIF still detects one ligand-acceptor and one ligand-donor hydrogen bond. N13–LEU398 distance changes 3.11045→3.00325 Å and DHA 161.77→164.92°; N3–GLU396 changes 3.11286→2.98854 Å and 140.64→142.94°. Geometry improves without adding a hydrogen-bond type.'),
            choose('连续方向满足度由 1.24859 升至 1.36223；埋藏未满足极性代理由 6.18584 降至 6.07366，改善约 1.81%，仍有限。O2、O19、N15 等残留埋藏且缺直接极性接触的疑点；缺少水桥与微观状态集合，不能直接认定为真实未满足氢键。',
                'Continuous directional satisfaction rises 1.24859→1.36223; unsatisfied-burial proxy falls 6.18584→6.07366, only about 1.81%. O2, O19 and N15 retain buried-polar concerns without direct detected polar contacts. Missing water and microstate ensembles prevent treating them as confirmed unsatisfied hydrogen bonds.'),'',
            '## '+choose('验证与边界','Validation and limits'),'',
            choose('- 50 项单元测试通过；包括 MMFF/界面数值梯度、芳香/互变异构编码、离散编辑隔离、自然修复归因和反证修订。实时模型奖励梯度有限差分检验通过。',
                '- 50 unit tests pass, covering numerical derivatives, aromatic/tautomer encoding, isolated categorical trials, native attribution and counterevidence revisions. The live-model raw reward gradient passes finite differences.'),
            choose('- 无引导及旧固定奖励均精确重现历史配对结果。新自动修订分支从 t=0.75 重启后，终态张量、自条件、时间、RNG、奖励状态和预算均逐值一致。所有分支未引导的两个批次分子及其自条件保持一致，RNG 一致；原始 runtime 哈希未变。',
                '- Unguided and legacy fixed branches exactly reproduce previous paired results. Restarting the adaptive branch at t=0.75 reproduces current tensors, SC, times, RNG, reward state and budget exactly. Other two batch molecules and SC remain identical across arms; RNG matches and original runtime hashes are unchanged.'),
            choose('- 无引导 51 个时点中 45 个可评估，6 个明确不可用。三个新分支各接受 41/50 个坐标步骤，最大注入路径 1.64 Å。早期部分步因化学图/参照不可用而回退原生更新，未虚构奖励值。',
                '- 45 of 51 native frames are evaluable; six are explicitly unavailable. Each new arm accepts 41/50 coordinate steps with a maximum 1.64 Å injected path. Some early steps fall back to native updates because chemistry/reference is unavailable; no reward value is fabricated.'),
            choose('- 所有评估终态通过本次 PoseBusters 所报告检查，未检出严重内碰撞或蛋白碰撞；这不能抹去连续应变和界面缺口。末端没有进行坐标最小化来替代生成。',
                '- All assessed finals pass reported PoseBusters checks, with no detected severe internal/protein clashes. Continuous strain and interface gaps remain. No posthoc coordinate minimization replaces the generated final.'),
            choose('- Vina 单一准备状态、Vinardo 相关评分、局部 MMFF 极小值均有限制。本例不能外推为统计性能提升或实验亲和力提高。',
                '- Single-state Vina, correlated Vinardo and local MMFF minima limit interpretation. This case does not establish statistical performance or experimental affinity improvement.'),'',
            '## '+choose('产物与复现','Artifacts and reproducibility'),'',
            '- [Comparison table](comparison.csv), [machine-readable result](comparison.json), [feedback](MolThinkerFeedback.json)',
            '- [Chemical trials](chemical_search_audit.json), [runtime audit](validation/runtime_audit.json), [gradient audit](validation/continuous_gradient.json)',
            '- [Native reference](native_reference.json), [planned runs](planned_runs.json), [provenance](provenance.json)',
            '- [Full new continuous risk report](evaluation/cases/outcome_continuous/DiagnosticReport.'+('zh' if zh else 'en')+'.md)',
            '- [Full revised risk report](evaluation/cases/outcome_adaptive/DiagnosticReport.'+('zh' if zh else 'en')+'.md)','',
            choose('开发过程中发现并修复了外部评分消耗 RNG、芳香/Kekule 比较不一致及将表示切换误算为改图的问题。早期开发运行保留在父目录、verified 和 final_validation；本报告仅采用当前 release 目录中测试通过后的完整运行，不混合结果。',
                'Development uncovered and fixed external-scoring RNG side effects, aromatic/Kekule comparison rejection and representation-only pseudo-edits. Earlier development runs remain in the parent directory, verified and final_validation. This report uses only completed release results after tests passed.'),'',
            'Remote root: `'+str(root)+'`','',
            choose('实现依据：','Implementation references: ')+
            '[RDKit force-field gradients](https://rdkit.org/docs/source/rdkit.ForceField.rdForceField.html), '+
            '[RDKit tautomer API](https://www.rdkit.org/docs/cppapi/MolStandardize_2Tautomer_8h_source.html), '+
            '[AutoDock Vina](https://autodock-vina.readthedocs.io/en/latest/vina.html)']
        (root/f'Results.{lang}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selection=[r for r in records if r['label'] in ('native_final','legacy_fixed300','outcome_continuous','outcome_adaptive')]
    fig,axes=plt.subplots(1,3,figsize=(11,4.5),layout='constrained')
    names=['Native','Legacy','Continuous','Revised'];colors=['#8a9ba8','#6e95bb','#268f76','#c7886a']
    for ax,key,title in zip(axes,['pkd','strain_kcal_mol','vina'],['Predicted pKd (higher)','MMFF strain, kcal/mol (lower)','Vina score, kcal/mol (lower)']):
        vals=[r[key] for r in selection];ax.bar(names,vals,color=colors)
        if key=='pkd':ax.set_ylim(5.35,5.6)
        if key=='vina':ax.set_ylim(-6.1,-5.5);ax.invert_yaxis()
        ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=25);ax.spines[['top','right']].set_visible(False)
        for i,v in enumerate(vals):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle('Matched t=0.50 continuation: structural improvement and objective tradeoffs',fontsize=12)
    fig.savefig(root/'comparison.png',dpi=190);fig.savefig(root/'comparison.svg');plt.close(fig)
    print(json.dumps(dict(records=records,feedback_route=feedback['route'],runtime_passed=audit['all_passed'])),flush=True)


if __name__=='__main__':main()
