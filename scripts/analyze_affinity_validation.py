"""Audit paired suffixes and produce bilingual result artifacts."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw,AllChem
from molsteer.common import write_json,file_hash
from run_exact_continuations import unequal

NAMES={
 'control':('无引导','Unguided'), 'legacy':('旧结构奖励','Previous structural reward'),
 'affinity_retained':('增加 pKd，保留初始保持','Add pKd; retain initial restraints'),
 'affinity_released':('解除初始保持，局部几何','Release initial restraints; local geometry'),
 'affinity_global':('全分子几何，小预算','Global geometry; small budget'),
 'affinity_global_large':('全分子几何，大预算','Global geometry; larger budget'),
 'affinity_global_large_a3':('大预算，affinity 权重 3','Larger budget; affinity weight 3'),
 'branch_cool':('离散分支 T=0.7','Categorical branch T=0.7'),
 'branch_warm':('离散分支 T=1.5','Categorical branch T=1.5'),
 'branch_hot':('离散分支 T=2.0','Categorical branch T=2.0'),
 'structure_weight_3':('结构权重 3','Structure weight 3'),
 'structure_weight_10':('结构权重 10','Structure weight 10'),
 'structure_weight_30':('结构权重 30','Structure weight 30')}


def floatsum(rows,key):return sum(float(r.get(key,0)) for r in rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();quality=json.loads((root/'final_quality.json').read_text())
    runs={r['label']:r for r in json.loads((root/'runs.json').read_text())}
    bylabel={r['label']:r for r in quality};control=bylabel['control']
    load=lambda path:torch.load(path,weights_only=True,map_location='cpu')
    control_cp=load(Path(control['directory'])/control['arm']/'resume_final.pt')
    control_pred=load(Path(control['source'])/'world_prediction.pt')
    control_trace=load(Path(control['directory'])/control['arm']/'tensor_trace.pt')
    records=[];same_rng=True;other_batch_exact=True
    for i,r in enumerate(quality):
        directory=Path(r['directory']);arm=r['arm'];run=runs[r['label']]
        cp=load(directory/arm/'resume_final.pt');pred=load(Path(r['source'])/'world_prediction.pt')
        displacement=(pred['coords'][0]-control_pred['coords'][0]).norm(dim=-1)
        trace=[json.loads(line) for line in (directory/arm/'guidance_trace.jsonl').read_text().splitlines()]
        tensors=load(directory/arm/'tensor_trace.pt')
        changed=[x['step'] for x,y in zip(tensors,control_trace)
                 if any(not torch.equal(x[k],y[k]) for k in ['atom_classes','charge_classes','bond_classes'])]
        rgdiff=unequal(control_cp['rng'],cp['rng']);same_rng &= not rgdiff
        non_target=[]
        for section in ['curr','cond']:
            for key,value in cp[section].items():
                if torch.is_tensor(value) and value.ndim and value.shape[0]==3:
                    if not torch.equal(value[:2],control_cp[section][key][:2]):non_target.append(section+'/'+key)
        other_batch_exact &= not non_target
        rejected=Counter(reason for t in trace for attempt in t.get('proposal_attempts',[]) for reason in attempt['failures'])
        unavailable=Counter(t['guidance_unavailable'] for t in trace if 'guidance_unavailable' in t)
        cfg=run['config'];spec=json.loads(Path(cfg['reward_programs']['creativity']).read_text())
        records.append(dict(id=f'A{i}',label=r['label'],name_en=NAMES[r['label']][1],name_zh=NAMES[r['label']][0],
            affinity=r['affinity'],delta_affinity={k:r['affinity'][k]-control['affinity'][k] for k in r['affinity']},
            strain=r['strain'],delta_strain=r['strain']-control['strain'],strain_reduction_percent=100*(control['strain']-r['strain'])/control['strain'],
            same_isomeric_smiles=r['sdf_isomeric_smiles']==control['sdf_isomeric_smiles'],
            pose_rms_displacement_angstrom=float(displacement.square().mean().sqrt()),
            max_pose_displacement_angstrom=float(displacement.max()),raw_coordinate_alignment='fixed receptor frame; identical atom slots, no fit',
            categorical_diverged_steps_from_control=changed,final_rng_exact=not rgdiff,other_batch_difference_paths=non_target,
            budget=cfg['budget'],categorical_proposal=cfg.get('categorical_proposal'),program_id=spec['program_id'],
            affinity_weight=spec.get('affinity_weight'),structure_weight=spec.get('structure_weight'),
            accepted_nonzero_steps=sum(t.get('injected_max_angstrom',0)>1e-9 for t in trace),
            max_injected_path_angstrom=run['summary']['max_injected_path_angstrom'],
            unavailable=dict(unavailable),rejected_proposals=dict(rejected),
            sanitized=r['sanitized'],geometry_outliers=r['geometry_outliers'],protein_clashes=r['protein_clashes'],
            intra_clashes=r['intra_clashes'],pb_failed=r['pb_failed'],pb_status=r['pb_status'],risk_count=r['risk_count'],
            final_relative=str(Path(r['source']).relative_to(root)),
            diagnostic_relative=str((directory/'evaluation'/arm/'final/DiagnosticReport.en.md').relative_to(root))))
    # Reproduce the historical structural-guidance arm, beyond matching a score.
    old=root.parent/'molsteer_exact_restart_20260923/continuations/t_0.50/eta_300/creativity/resume_final.pt'
    legacy=load(Path(bylabel['legacy']['directory'])/'creativity/resume_final.pt');oldcp=load(old)
    legacy_diff={k:unequal(oldcp[k],legacy[k]) for k in ['curr','cond','times','rng']}
    best=next(r for r in records if r['label']=='structure_weight_30')
    second=next(r for r in records if r['label']=='structure_weight_10')
    highest=max(records,key=lambda r:r['affinity']['pkd'])
    finite_checks={x.stem:json.loads(x.read_text())['passed'] for x in (root/'validation').glob('*_gradient.json')}
    result=dict(records=records,primary='pkd',control_exact=json.loads((root/'validation/control_exact.json').read_text())['all_exact'],
        legacy_exact=not any(legacy_diff.values()),legacy_differences=legacy_diff,all_final_rng_exact=same_rng,
        other_batch_items_bitwise_preserved=other_batch_exact,gradient_checks=finite_checks,
        all_same_final_isomeric_smiles=all(r['same_isomeric_smiles'] for r in records),
        selected_tradeoffs=['structure_weight_10','structure_weight_30'],highest_pkd=highest['label'],
        limitations=['one checkpoint; no seed sweep','optimized and evaluated affinity from same uncalibrated model',
            'no independent binding energy or prepared ProLIF','geometry is not full force-field strain',
            '13 tested settings do not establish global optimality','categorical tempering did not change final identity'])
    write_json(root/'analysis.json',result)
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['id','label','pKd','delta_pKd','pKi','pIC50','pEC50','strain_kcal_mol','strain_reduction_percent','pose_RMS_displacement_A','max_pose_displacement_A','accepted_nonzero_steps','max_injected_path_A'])
        for r in records:writer.writerow([r['id'],r['label'],r['affinity']['pkd'],r['delta_affinity']['pkd'],r['affinity']['pki'],r['affinity']['pic50'],r['affinity']['pec50'],r['strain'],r['strain_reduction_percent'],r['pose_rms_displacement_angstrom'],r['max_pose_displacement_angstrom'],r['accepted_nonzero_steps'],r['max_injected_path_angstrom']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    y=np.arange(len(records));colors=['#2b8c63' if r['label'].startswith('structure_weight') else '#4378ad' for r in records]
    axes[0].barh(y,[r['delta_affinity']['pkd'] for r in records],color=colors)
    axes[0].axvline(0,color='#777777',lw=.8);axes[0].set_xlabel('Predicted pKd gain over matched control')
    axes[0].set_yticks(y,[r['id']+'  '+r['label'] for r in records]);axes[0].invert_yaxis()
    axes[1].barh(y,[r['strain'] for r in records],color=colors)
    axes[1].axvline(control['strain'],color='#a34737',ls='--',label='Unguided: 19.557')
    axes[1].set_xlabel('MMFF local relaxation proxy (kcal/mol)');axes[1].legend(loc='lower right')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Same t=0.50 runtime -> t=1.00 | same RNG | all 13 final chemical identities equal')
    fig.tight_layout();fig.savefig(root/'affinity_structure_comparison.png',dpi=180);fig.savefig(root/'affinity_structure_comparison.svg');plt.close(fig)
    mol=Chem.MolFromMolFile(str(Path(control['source'])/'ligand.sdf'),removeHs=False)
    AllChem.Compute2DCoords(mol)
    for atom in mol.GetAtoms():atom.SetProp('atomNote',str(atom.GetIdx()))
    Draw.MolToFile(mol,str(root/'shared_final_structure.png'),size=(900,650),legend='Shared final identity; labels are original zero-based atom slots')
    for zh in [True,False]:
        lang='zh-CN' if zh else 'en'
        title='生成效果验证：t=0.50 → 1.00' if zh else 'Generation validation: t=0.50 to 1.00'
        lines=['# '+title,'']
        lines += ([f"完成 13 个设置的真实后缀生成。结构权重 30 的方案同时将 pKd 从 {control['affinity']['pkd']:.6f} 提高到 {best['affinity']['pkd']:.6f}（+{best['delta_affinity']['pkd']:.6f}），应变从 {control['strain']:.6f} 降到 {best['strain']:.6f} kcal/mol（降低 {best['strain_reduction_percent']:.2f}%）。这是本次检查点上的代理指标改善，未证明真实结合亲和力提高。",'',
            f"结构权重 10 提供另一种取舍：pKd={second['affinity']['pkd']:.6f}、应变={second['strain']:.6f}。最高 pKd 出现在 {highest['id']}（{highest['affinity']['pkd']:.6f}），但应变升至 {highest['strain']:.6f}，不作为兼顾结构的优选。",''] if zh else
            [f"Thirteen authentic suffix continuations were evaluated. Structure weight 30 raised pKd from {control['affinity']['pkd']:.6f} to {best['affinity']['pkd']:.6f} (+{best['delta_affinity']['pkd']:.6f}), while reducing the relaxation proxy from {control['strain']:.6f} to {best['strain']:.6f} kcal/mol ({best['strain_reduction_percent']:.2f}%). These are surrogate improvements for this checkpoint, not evidence of increased experimental affinity.",'',
             f"Structure weight 10 is another tradeoff: pKd={second['affinity']['pkd']:.6f}, strain={second['strain']:.6f}. The highest pKd was {highest['id']} ({highest['affinity']['pkd']:.6f}), but its strain rose to {highest['strain']:.6f}; it is not the preferred structural compromise.",''])
        lines += ['![Matched comparison](affinity_structure_comparison.png)','', '## '+('实验控制与真实性' if zh else 'Experimental controls and fidelity'),'']
        lines += (['- 目标仅为 5i0b_A__5vef_M77 / ligand_002；只使用原 t=0.50 的诊断与 StatePacket 形成奖励，所有生成区间均为 0.50–1.00。终态 Reader 用于质量评价，不反向改变初始诊断。',
            '- 每组恢复同一份真实 runtime.pt，包括当前状态、自条件缓存、prior、时间、口袋状态及 CPU/CUDA/Python/NumPy RNG。没有更换随机种子，没有测试 t=0.25。',
            f"- 无引导与原始 clean 的状态、head 输出、自条件缓存和随机数均逐张量一致：{result['control_exact']}；旧奖励结果与此前 η=300 的运行一致：{result['legacy_exact']}。",
            f"- 所有组最终 RNG 相同：{same_rng}；另外两个 batch 分子的状态与缓存逐张量未变：{other_batch_exact}。所有 {len(finite_checks)} 种新奖励有限差分检查通过。",
            '- 首轮预声明 10 个设置。看到大预算导致应变升高后，追加结构权重 3、10、30 三个设置；这是根据初步结果进行的补充实验，未伪装为事先固定的设计。',
            '- 输出均为原生最终 head 产生的分子，没有后处理修复或最小化。应变评估在分子副本上进行。',''] if zh else
            ['- Target: 5i0b_A__5vef_M77 / ligand_002. Only the original t=0.50 diagnosis and StatePacket informed rewards. Every generation interval was 0.50–1.00. Terminal Reader assessments did not redefine the initial diagnosis.',
            '- Every arm restored the same authentic runtime, including state, self-conditioning, prior, time, pocket state and CPU/CUDA/Python/NumPy RNG. No seed sweep or t=0.25 run.',
            f"- Unguided tensors, heads, conditioning and RNG reproduce native clean output exactly: {result['control_exact']}. Previous η=300 structural guidance also reproduces exactly: {result['legacy_exact']}.",
            f"- All final RNG states match: {same_rng}; both other batch items are bitwise preserved: {other_batch_exact}. All {len(finite_checks)} new reward derivative screens passed.",
            '- Ten initial settings were declared before generation. Three stronger structural weights were added after observing strain regression; this adaptive follow-up is recorded separately.',
            '- Submitted molecules are native final-head outputs, with no post-generation minimization. Strain evaluation minimizes copies only.',''])
        lines+=['## '+('全部结果' if zh else 'All results'),'','| ID | '+('设置' if zh else 'Setting')+' | pKd | ΔpKd | pKi | pIC50 | pEC50 | '+('应变 kcal/mol' if zh else 'Strain kcal/mol')+' |','|---|---|---:|---:|---:|---:|---:|---:|']
        for r in records:
            aff=r['affinity'];name=r['name_zh' if zh else 'name_en']
            lines.append(f"| {r['id']} | {name} | {aff['pkd']:.6f} | {r['delta_affinity']['pkd']:+.6f} | {aff['pki']:.6f} | {aff['pic50']:.6f} | {aff['pec50']:.6f} | {r['strain']:.6f} |")
        lines += ['', '## '+('实际奖励与控制参数' if zh else 'Executed reward and control parameters'),'']
        lines += (['最终结构权重 30 方案实际使用：','',r'\[R_t=2\tanh\left[\frac{pK_d(t)-pK_{d,ctl}(t)}{2}\right]-30\mathcal L_S-F_{pocket}.\]',
            r'\[\mathcal L_S=0.1\log\left(\frac{e^{F_{geom}/0.1}+e^{F_{clash}/0.1}}{2}\right)+0.05(F_{geom}+F_{clash}).\]',
            r'\[F_{geom}=\sqrt{1+\operatorname{mean}[\max(|z|-1,0)^2]}-1+0.1\operatorname{mean}(z^2).\]','',
            'z 是相对当前化学图 MMFF 键长/键角参照的偏差，以 10% 键长或 30° 角度归一化。F_clash 为最大正穿透量的平方，尺度 1 Å。F_pocket 是最近蛋白距离超出 4.5 Å 的平方均值，尺度 1 Å。几何项覆盖全分子，并随图重绑参照。1 个 pK 单位、饱和尺度 2、全部权重均未做统计校准。',
            '- η=300；单原子单步注入上限 0.04 Å，累计注入路径上限 2 Å。结构权重 10 只把上述 30 改为 10。η 是梯度注入系数，不能与结构权重混为一谈。',
            '- 初始坐标保持和初始图偏好均移除。化学有效性、断连、新增/恶化严重碰撞与口袋范围仍作为新增引导的不可补偿检查。每步使用同一下一时刻和缓存比较候选。',
            '- A0–A4 的步长/路径预算为 0.02/0.5 Å；A5–A12 为 0.04/2 Å。A2 保留原局部结构奖励，A3 移除保持约束，A4 同时扩大几何范围并改变其聚合，不能将 A4 的收益只归因于扩大范围。',
            '- A6–A9 affinity 权重为 3，其余新增方案为 1。A7–A9 在 0.50–0.70 对最大概率低于 0.95 的类别提案使用温度变换，原生采样器完成状态转移，同时同步对应自条件字段。它是小规模候选搜索，未实现多轮束重采样，也不属于随机种子扫描。',''] if zh else
            ['The structure-weight-30 candidate uses:', '',r'\[R_t=2\tanh\left[\frac{pK_d(t)-pK_{d,ctl}(t)}{2}\right]-30\mathcal L_S-F_{pocket}.\]',
            r'\[\mathcal L_S=0.1\log\left(\frac{e^{F_{geom}/0.1}+e^{F_{clash}/0.1}}{2}\right)+0.05(F_{geom}+F_{clash}).\]',
            r'\[F_{geom}=\sqrt{1+\operatorname{mean}[\max(|z|-1,0)^2]}-1+0.1\operatorname{mean}(z^2).\]','',
            'z is deviation from the current graph MMFF bond/angle reference, divided by 10% of bond length or 30 degrees. Clash cost is squared maximum positive penetration, scaled by 1 Å. Pocket cost is mean squared excess of nearest protein distance over 4.5 Å, scaled by 1 Å. References rebind to the whole current graph. All weights and the one-pK-unit normalization/two-unit saturation are uncalibrated.',
            '- η=300, per-atom step cap 0.04 Å and cumulative injected-path cap 2 Å. Structure weight 10 changes only 30 to 10 above. η controls gradient injection and is distinct from the structural weight.',
            '- Initial-coordinate retention and frozen initial graph preference are removed. Chemical validity, connectivity, new/worsened severe clashes and pocket occupancy remain noncompensable proposal checks. Proposals use matched next times and caches.',
            '- A0–A4 use 0.02/0.5 Å step/path budgets; A5–A12 use 0.04/2 Å. A2 retains the original structural reward, A3 releases restraints, and A4 changes both geometry coverage and aggregation. Those A4 contributions are not individually isolated.',
            '- A6–A9 use affinity weight 3; other new arms use 1. A7–A9 temper categorical slots below 0.95 confidence during 0.50–0.70. Native transitions update the state and matching conditioning. This fixed candidate search is neither a seed sweep nor iterative beam resampling.',''])
        lines += ['## '+('结构变化与质量' if zh else 'Structural changes and quality'),'','![Shared chemical identity](shared_final_structure.png)','',
                  '`'+control['sdf_isomeric_smiles']+'`','']
        lines += ([f"13 组均为相同化学图和 SDF 立体构型，重原子槽位 20。分子式 {control['descriptors']['formula']}，MW={control['descriptors']['mw']:.3f}、logP={control['descriptors']['logp']:.4f}、TPSA={control['descriptors']['tpsa']:.2f}、QED={control['descriptors']['qed']:.6f}；这些性质没有改变。许可改变分子身份并未保证实际跨越到新化学图。",'',
            f"A12 相对无引导终态在受体固定坐标系中的重原子 RMS 位移为 {best['pose_rms_displacement_angstrom']:.6f} Å，最大原子位移 {best['max_pose_displacement_angstrom']:.6f} Å；这是不做叠合的坐标差。累计注入路径 {best['max_injected_path_angstrom']:.6f} Å 不等于终态位移，原生 flow 会继续改变轨迹。",'',
            '所有终态均可 sanitize、连通，已执行的 PoseBusters 检查无失败，键长/键角异常数、蛋白碰撞与分子内碰撞均为 0。A6–A9 的应变超过 20 kcal/mol 筛查阈值，诊断增加一个全分子应变风险。其余组仍有合并后的叠氮相关 PAINS/BRENK 筛选警示，不能声称风险全部消失。', '',
            '应变是原始重原子构象（先弛豫氢）与同图局部 MMFF94s 最小值的能差代理。这里最终图与质子化相同，比较比跨图更直接；仍不是全局最小应变或结合自由能。所有这些弛豫计算均报告收敛。',''] if zh else
            [f"All 13 final graphs and SDF stereochemical identities are equal, with 20 active heavy-atom slots. Formula {control['descriptors']['formula']}, MW={control['descriptors']['mw']:.3f}, logP={control['descriptors']['logp']:.4f}, TPSA={control['descriptors']['tpsa']:.2f}, QED={control['descriptors']['qed']:.6f}; these descriptors are unchanged. Permission to change identity did not yield a new final graph.",'',
            f"A12's heavy-atom RMS displacement from the control in the fixed receptor frame is {best['pose_rms_displacement_angstrom']:.6f} Å, maximum {best['max_pose_displacement_angstrom']:.6f} Å, without alignment. Its cumulative injection {best['max_injected_path_angstrom']:.6f} Å is not the terminal displacement because native flow continues to evolve the trajectory.",'',
            'All final molecules sanitize and are connected; executed PoseBusters checks have no failures, with zero reported bond/angle outliers, protein clashes and intramolecular clashes. A6–A9 exceed the 20 kcal/mol strain screening threshold, adding a molecule-level strain risk. Every final candidate retains a merged azido-related PAINS/BRENK screening concern; risks have not all disappeared.','',
            'The strain proxy is the energy drop from the submitted heavy-atom pose with relaxed hydrogens to a same-graph local MMFF94s minimum. Final graph and protonation are equal here, making this a more direct comparison than across graphs. It is still neither global-minimum strain nor binding free energy. All relaxation calculations report convergence.',''])
        lines += ['## '+('结果解释与限制' if zh else 'Interpretation and limits'),'']
        lines += (['加入显式 pKd 目标确实改变了优化方向。仅扩大预算或增强 affinity 权重会推高 head 数值，同时可能损害结构。全分子几何均值较小，权重 1 时结构项不足以抑制这种权衡；在相同预算下提高结构权重得到 A11/A12 的共同改善。这个结果支持“重新平衡奖励”而非单纯继续提高 η。', '',
            'A11/A12 是本次已测试方案中的不同取舍，没有全局最优证明。未校准的同一 FLOWR head 同时参与优化与评价，不能作为独立亲和力验证；未执行准备完善的独立 docking/实验评价。没有同终点不确定性估计，不能把这些微小头分数差异解释为统计显著结果。', '',
            '温度分支没有带来最终身份变化。中间类别轨迹的变化步数记录于 analysis.json；这种简单分支未证明足以实现较大骨架搜索。当前固定槽位适配器也不支持增删重原子。', '',
            '覆盖限制：受体质子化/显式氢/化学类型未验证，ProLIF 未准备；无来源核验的 Vina 分数；最终 Reader 未绑定相邻阶段历史，因此其 trajectory_change 不可用，轨迹变化另由保存的状态张量检查。可微奖励未覆盖完整扭转、共轭平面或全力场能。', '',
            '记录说明：首轮 affinity trace 的 pullback_norm_ratio 混合了直接 affinity 分支与结构坐标分支，不用于结论；后续记录已将梯度路径分开标注。A2 初始 JSON 的通用约束文字未准确列出保留的 1 Å 初始终点限制；实际计算调用原结构 guard，本文按实际执行解释，后续构建器已修正该描述。',''] if zh else
            ['An explicit pKd objective changed optimization behavior. Larger budgets or affinity weights increased the head score but could worsen structural strain. The global mean geometric penalty was too weak at structural weight 1; increasing its weight under the same budget produced the joint improvements in A11/A12. This supports rebalancing the reward instead of merely increasing η.','',
            'A11/A12 are observed tradeoffs among tested settings, not proven global optima. The same uncalibrated FLOWR head was optimized and evaluated, so this is not independent affinity validation. No prepared independent docking or experimental assay was performed. No same-endpoint uncertainty estimate supports statistical significance claims for these differences.','',
            'Temperature branches did not change final identity. Intermediate categorical divergence steps are preserved in analysis.json; this simple search has not demonstrated large scaffold exploration. The adapter also cannot add or remove heavy-atom slots.','',
            'Coverage: receptor protonation, explicit hydrogens and chemical typing are unverified; ProLIF is not prepared; no provenance-verified Vina score; terminal Reader lacks paired history for trajectory_change, which is instead audited from saved tensors. Differentiable rewards do not cover full torsional, conjugated-plane or force-field energy.','',
            'Record notes: the initial affinity traces contain a pullback_norm_ratio mixing a direct affinity path with output-coordinate derivatives; it is not used for conclusions and subsequent logging separates the scopes. A2 initial generic constraint prose omitted the retained 1 Å endpoint-from-start bound. Its executed guard was the original structural guard; this report follows actual execution and the builder description has been corrected.',''])
        lines += ['## '+('可复核文件' if zh else 'Auditable artifacts'),'','- [CSV](comparison.csv) · [Analysis JSON](analysis.json) · [DesignIntent](DesignIntent.json) · [Initial design](planned_ablations.json) · [Adaptive follow-up](adaptive_followup.json)',
            '- [Original t=0.50 StatePacket](reasoning/t_0.50/StatePacket.json) · [Original t=0.50 diagnosis](reasoning/t_0.50/DiagnosticReport.en.md)',
            '- [Control exactness](validation/control_exact.json) · [Runs and executable configurations](runs.json) · [Source and checkpoint hashes](provenance.json)','']
        for label in ['control','structure_weight_10','structure_weight_30','branch_cool']:
            r=next(r for r in records if r['label']==label)
            report=r['diagnostic_relative'].replace('.en.md','.zh.md') if zh else r['diagnostic_relative']
            lines.append(f"- {r['id']}: [SDF]({r['final_relative']}/ligand.sdf) · [DiagnosticReport]({report}) · [RewardProgram](programs/{label}.json)")
        (root/f'Results.{lang}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(root/'artifact_manifest.json',{str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=file_hash(path))
        for path in root.rglob('*') if path.is_file() and path.name!='artifact_manifest.json'})
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
