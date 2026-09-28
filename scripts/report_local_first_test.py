"""Independent evaluation and standalone HTML for local-first tests."""
import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from molsteer.common import write_json


def svg(mol,highlight=(1,10,14)):
    mol=Chem.Mol(mol);rdDepictor.Compute2DCoords(mol);draw=rdMolDraw2D.MolDraw2DSVG(650,360)
    draw.drawOptions().addAtomIndices=True;draw.drawOptions().clearBackground=False
    draw.DrawMolecule(mol,highlightAtoms=[i for i in highlight if i<mol.GetNumAtoms()]);draw.FinishDrawing();return draw.GetDrawingText()


def delta(value,reference,lower_better=False):
    if value is None or reference is None:return 'n/a'
    d=value-reference;good=d<0 if lower_better else d>0
    return f'<span class="{("good" if good else "bad" if d else "neutral")}">{d:+.3f}</span>'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();source=Path(a.source_root).resolve();evaluation=root/'evaluation'
    prep=source/'output/molsteer_researcher_integrated_20260924/evaluation/preparation'
    shutil.copytree(prep,evaluation/'preparation',dirs_exist_ok=True)
    target=Path('5i0b_A__5vef_M77/ligand_002/final')
    cases=[dict(label='native_final',path=str(source/'output/crossdocked_100target_stage_test_exact_20260923'/target)),
        dict(label='previous_direct_sum_eta300',path=str(source/'output/molsteer_researcher_integrated_20260924/continuations/research_off/creativity'/target))]
    for label in ['local_first_eta100','local_first_eta300']:
        cases.append(dict(label=label,path=str(root/'continuations'/label/'local_first'/target)))
    write_json(evaluation/'cases.json',cases)
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_guidance_independently.py')),
        '--model-root',str(source),'--output',str(evaluation),'--cases-json',str(evaluation/'cases.json')],check=True)
    context=json.loads((evaluation/'ReaderComparisonContext.json').read_text());rows=[]
    for row in context['observations']:
        m=Chem.MolFromMolFile(row['source_sdf'],removeHs=True);ff=row['forcefield'];forces={v['atom_id']:v['force_kcal_mol_angstrom'] for v in ff['atom_force_norms']}
        rows.append(dict(label=row['label'],smiles=row['smiles'],pkd=row.get('model_predictions',{}).get('affinity',{}).get('pkd'),
            vina=row['docking']['vina']['score_only_kcal_mol'],vinardo=row['docking']['vinardo']['score_only_kcal_mol'],
            strain=ff['strain_kcal_mol'],force1=forces.get(1),force10=forces.get(10),force14=forces.get(14),
            posebusters=row['metrics']['posebusters']['values']['all_reported_checks_pass'],svg=svg(m)))
    write_json(root/'comparison.json',[{k:v for k,v in row.items() if k!='svg'} for row in rows])
    native=next(v for v in rows if v['label']=='native_final');runs=json.loads((root/'runs.json').read_text());trace={}
    for run in runs:
        pth=root/'continuations'/run['label']/'local_first/local_first_trace.jsonl';trace[run['label']]=[json.loads(v) for v in pth.read_text().splitlines()]
    diagnosis=(source/'output/molsteer_researcher_integrated_20260924/input/DiagnosticReport.zh.md').read_text()
    derivation=(root/'Simulated_MolThinker_LocalReward.zh-CN.md').read_text() if (root/'Simulated_MolThinker_LocalReward.zh-CN.md').exists() else ''
    md=MarkdownIt('commonmark',{'html':False}).enable('table')
    table=''.join(f'''<tr><td>{html.escape(r['label'])}</td><td>{r['pkd']:.3f}<br>{delta(r['pkd'],native['pkd'])}</td>
      <td>{r['vina']:.3f}<br>{delta(r['vina'],native['vina'],True)}</td><td>{r['vinardo']:.3f}<br>{delta(r['vinardo'],native['vinardo'],True)}</td>
      <td>{r['strain']:.3f}<br>{delta(r['strain'],native['strain'],True)}</td><td>{r['force1']:.1f} / {r['force10']:.1f} / {r['force14']:.1f}</td><td>{'pass' if r['posebusters'] else 'fail'}</td></tr>''' for r in rows)
    cards=''.join(f'''<article><h3>{html.escape(r['label'])}</h3>{r['svg']}<p><code>{html.escape(r['smiles'])}</code></p></article>''' for r in rows)
    monitor=[]
    for run in runs:
        rr=trace[run['label']];accepted=sum(v['accepted'] for v in rr);stages=run['summary']['stage_counts']
        last=next((v['review_detail'] for v in reversed(rr) if v.get('review_detail')),None)
        last_strain=f"{last['local_strain']:.3f}" if last else 'n/a'
        monitor.append(f'''<tr><td>{run['label']}</td><td>{accepted}/{len(rr)}</td><td>{html.escape(json.dumps(stages))}</td>
          <td>{run['summary']['max_injected_path_angstrom']:.3f}</td><td>{html.escape(last['stage'] if last else 'unavailable')}</td>
          <td>{last_strain}</td></tr>''')
    source_binding=json.loads((root/'source_binding.json').read_text());preflight=json.loads((root/'gradient_preflight.json').read_text())
    fidelity=json.loads((root/'runtime_fidelity.json').read_text());tests=(source/'output/molsteer_local_first_20260924_tests.log').read_text()
    test_count=sum(int(v) for v in re.findall(r'Ran (\d+) tests',tests));tests_passed=tests.count('\nOK\n')>=2
    eta100=next(v for v in rows if v['label']=='local_first_eta100');eta300=next(v for v in rows if v['label']=='local_first_eta300')
    direct=next(v for v in rows if v['label']=='previous_direct_sum_eta300')
    conclusion=f'''<p>局部优先奖励产生了可测但有限的改善。η=100/300 的 pKd 分别提高 {eta100['pkd']-native['pkd']:+.3f}/{eta300['pkd']-native['pkd']:+.3f}，全分子 MMFF strain 分别改变 {eta100['strain']-native['strain']:+.3f}/{eta300['strain']-native['strain']:+.3f} kcal/mol；两者保持原化学图并通过 PoseBusters。</p>
    <p>独立对接反证没有确认这项收益：Vina 分别改变 {eta100['vina']-native['vina']:+.3f}/{eta300['vina']-native['vina']:+.3f} kcal/mol，Vinardo 分别改变 {eta100['vinardo']-native['vinardo']:+.3f}/{eta300['vinardo']-native['vinardo']:+.3f} kcal/mol，方向均为变差。η=300 的 pKd 略高，但 strain 和 Vina 比 η=100 更差，增加 η 没有解决目标错配。</p>
    <p>上一版直接求和 η=300 在本样本上仍占优：pKd={direct['pkd']:.3f}、Vina={direct['vina']:.3f}、strain={direct['strain']:.3f}。这不否定局部优先架构，而是说明当前“局部松弛总能量”仍过于粗糙：它降低了 atom 1/14 的力，却把部分负担转移到 atom 10；终点局部应变也没有超过预设的 2 kcal/mol 额外改善门槛，因此控制器最终退回 strain 阶段。</p>'''
    report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>MolSteer 局部优先奖励测试</title>
    <style>body{{font:15px/1.6 system-ui;margin:0;background:#f4f7f8;color:#17242b}}main{{max-width:1180px;margin:auto;padding:30px}}section,article{{background:white;border:1px solid #dce5e8;border-radius:12px;padding:20px;margin:16px 0}}h1,h2{{color:#164e63}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #dde6e8;text-align:left}}.good{{color:#087f5b;font-weight:700}}.bad{{color:#c92a2a;font-weight:700}}.neutral{{color:#59666d}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}}svg{{max-width:100%;height:auto}}code{{white-space:normal}}.note{{border-left:4px solid #d97706;padding-left:12px}}</style></head><body><main>
    <h1>MolSteer 局部缺陷优先奖励：远端生成测试</h1>
    <section><h2>结论概览</h2><p>本报告比较无引导终态、上一版五项直接求和 η=300，以及局部优先奖励 η=100/300。所有差值均相对同一完整 t=0.50 检查点的无引导终态。</p>{conclusion}</section>
    <section><h2>来源与可重复性</h2><p class="note">请求目录的 state.pt 不含 RNG 和自条件缓存。续跑使用内容绑定的精确 runtime：<code>{html.escape(source_binding['continuation_runtime'])}</code>。原始 state 与 SDF 哈希仍记录在 source_binding.json。</p>
    <p>局部 MMFF 梯度有限差分预检：{'通过' if preflight['passed'] else '失败'}；scaled relative error={preflight['relative_scaled_error']:.4g}。</p></section>
    <section><h2>终态定量结果</h2><table><thead><tr><th>方案</th><th>pKd / Δ</th><th>Vina / Δ</th><th>Vinardo / Δ</th><th>MMFF strain / Δ</th><th>局部力 atom 1/10/14</th><th>PoseBusters</th></tr></thead><tbody>{table}</tbody></table></section>
    <section><h2>MolMonitor 过程</h2><table><thead><tr><th>运行</th><th>接受步</th><th>梯度阶段计数</th><th>最大路径 Å</th><th>终止阶段</th><th>终止局部应变</th></tr></thead><tbody>{''.join(monitor)}</tbody></table></section>
    <section><h2>机制分析</h2><ul>
      <li>两条轨迹均有 23 个 affinity 梯度阶段，但在 t=0.99–1.00 重新进入 strain 阶段。局部约束投影只保护当前一步的一阶方向，后续非线性模型前向和原生积分仍可重新增加局部应变。</li>
      <li>终点 flat-bottom 几何损失为零，但局部力仍高，说明宽容差区间适合作为异常门槛，不能替代连续力或分解能量目标。</li>
      <li>Vina 在若干中期复核点曾改善，但终点回落。当前 Vina 是同一步候选门和评估 oracle，不是可微目标，也没有终端保持约束。</li>
      <li>局部松弛能量聚合了 halo 内多个自由度，允许力从 atom 1/14 转移到 atom 10。后续应把最坏原子力、键角/扭转能分量或 robust-tail 作为局部阶段的完成条件。</li>
      <li>η=300 的有效路径为 1.600 Å，η=100 为 1.242 Å，但阶段序列几乎一致；更大的名义强度主要增加位移，没有改变可达的奖励阶段。</li>
    </ul></section>
    <section><h2>分子结构</h2><div class="grid">{cards}</div></section>
    <section><h2>输入诊断报告</h2>{md.render(diagnosis)}</section>
    <section><h2>MolThinker 推导记录</h2>{md.render(derivation)}</section>
    <section><h2>验证记录</h2><p>共运行 {test_count} 个单元/回归测试：{'全部通过' if tests_passed else '存在失败'}。两条轨迹的 RNG、时间、自条件相关 cond 张量及非目标 batch 均与无引导终态逐位一致；目标化学图也保持一致。</p><pre>{html.escape(json.dumps(fidelity,ensure_ascii=False,indent=2))}</pre></section>
    <section><h2>解释限制</h2><ul><li>MMFF、Vina、Vinardo 与模型 affinity head 均为计算代理，不是实验结合自由能。</li><li>本次不做随机种子扫描；各方案共享保存的 RNG、自条件缓存和生成前缀。</li><li>τ=0.05、局部应变尺度 5 kcal/mol 和额外改善门槛 2 kcal/mol 是明示的试验参数，需要后续跨任务校准。</li></ul></section>
    </main></body></html>'''
    (root/'report.html').write_text(report,encoding='utf-8')
    audit=dict(html_bytes=len(report.encode()),case_count=len(rows),has_diagnosis='输入诊断报告' in report,
        has_derivation='MolThinker 推导记录' in report,has_molecules=report.count('<svg')>=len(rows),preflight_passed=preflight['passed'])
    audit.update(test_count=test_count,tests_passed=tests_passed,runtime_fidelity_passed=all(all(v[k] for k in ['non_target_curr_exact','non_target_cond_exact','rng_exact','times_exact','target_graph_same']) for v in fidelity.values()))
    write_json(root/'report_audit.json',audit)
    if not all([audit['has_diagnosis'],audit['has_derivation'],audit['has_molecules'],audit['preflight_passed'],audit['tests_passed'],audit['runtime_fidelity_passed']]):raise ValueError('Incomplete report')
    print(json.dumps([{k:v for k,v in r.items() if k!='svg'} for r in rows],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
