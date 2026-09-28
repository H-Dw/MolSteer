"""Render the updated-skill exact-dataset retest report."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from molsteer.common import file_hash, write_json


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def molecule_svg(path, legend):
    mol = Chem.MolFromMolFile(str(path), removeHs=True)
    if mol is None:
        return "<p>Structure unavailable</p>"
    mol = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol)
    svg = Draw.MolsToGridImage([mol], legends=[legend], molsPerRow=1, subImgSize=(440, 260), useSVG=True)
    return str(svg).replace("svg:", "")


def case_table(case):
    rows = []
    for stage in case["stages"]:
        cats = ", ".join(f"{f['category']} {f.get('atom_ids', [])}" for f in stage["findings"]) or "none"
        affinity = stage.get("affinity") or {}
        strain = stage.get("mmff_strain") or {}
        pb = stage.get("posebusters") or {}
        rows.append(
            f"<tr><td>{escape(stage['stage'])}</td><td>{escape(cats)}</td>"
            f"<td>{fmt(affinity.get('pkd'))}</td><td>{fmt(strain.get('strain_proxy_kcal_mol'))}</td>"
            f"<td>{escape(', '.join(pb.get('failed_checks', [])) or 'pass')}</td></tr>"
        )
    return "".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prior-comparison", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    dataset = Path(args.dataset).resolve()
    manifest = json.loads((root / "collection_manifest.json").read_text())
    routing = json.loads((root / "reward_design_summary.json").read_text())
    route_by = {(r["target_id"], r["ligand_id"]): r for r in routing}
    native_repair_count = sum(r["status"] == "native_self_repair" for r in routing)
    screen_only_count = sum(r["status"] == "screen_only" for r in routing)
    guided_count = sum(r["status"] == "coordinate_actionable" for r in routing)
    comparison_context = json.loads((root / "evaluation/ReaderComparisonContext.json").read_text())
    observations = {row["label"]: row for row in comparison_context["observations"]}
    native = observations["native_final"]
    guided = observations["updated_skill_staged_eta100"]
    guided_sdf = root / "continuations/5i0b_A__5vef_M77/ligand_002/staged_lexicographic_eta100/creativity/5i0b_A__5vef_M77/ligand_002/final/ligand.sdf"
    prior = json.loads(Path(args.prior_comparison).read_text())
    execution = json.loads((root / "runs.json").read_text())[0]["summary"]
    fidelity = json.loads((root / "runtime_fidelity.json").read_text())
    trace_path = root / "continuations/5i0b_A__5vef_M77/ligand_002/staged_lexicographic_eta100/creativity/local_first_trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    rejection_counts = Counter(
        reason
        for row in trace
        for attempt in row.get("proposal_attempts", [])
        for reason in attempt.get("failures", [])
    )
    accepted_by_stage = Counter(
        row.get("gradient", {}).get("stage", "unavailable") for row in trace if row.get("accepted")
    )
    monitor = {
        "accepted_steps": execution["accepted_steps"],
        "stage_counts": execution["stage_counts"],
        "accepted_by_stage": dict(accepted_by_stage),
        "rejection_counts": dict(rejection_counts),
        "oracle_calls": execution["oracle_calls"],
        "max_injected_path_angstrom": execution["max_injected_path_angstrom"],
        "eta_changed": False,
        "interpretation": "Same-time feasibility and independent-oracle review controlled proposal acceptance; eta was not escalated.",
    }
    write_json(root / "monitor_summary.json", monitor)

    def force_map(row):
        return {x["atom_id"]: x["force_kcal_mol_angstrom"] for x in row["forcefield"]["atom_force_norms"]}

    nf, gf = force_map(native), force_map(guided)
    result = {
        "dataset_cases": len(manifest),
        "guided_cases": guided_count,
        "native_self_repair_cases": native_repair_count,
        "screen_only_cases": screen_only_count,
        "selected_case": "5i0b_A__5vef_M77/ligand_002",
        "final": {
            "native": {
                "pkd": native["model_predictions"]["affinity"]["pkd"],
                "strain": native["forcefield"]["strain_kcal_mol"],
                "vina": native["docking"]["vina"]["score_only_kcal_mol"],
                "vinardo": native["docking"]["vinardo"]["score_only_kcal_mol"],
                "smiles": native["smiles"],
            },
            "guided": {
                "pkd": guided["model_predictions"]["affinity"]["pkd"],
                "strain": guided["forcefield"]["strain_kcal_mol"],
                "vina": guided["docking"]["vina"]["score_only_kcal_mol"],
                "vinardo": guided["docking"]["vinardo"]["score_only_kcal_mol"],
                "smiles": guided["smiles"],
            },
        },
        "deltas_guided_minus_native": {
            "pkd": guided["model_predictions"]["affinity"]["pkd"] - native["model_predictions"]["affinity"]["pkd"],
            "strain_kcal_mol": guided["forcefield"]["strain_kcal_mol"] - native["forcefield"]["strain_kcal_mol"],
            "vina_kcal_mol": guided["docking"]["vina"]["score_only_kcal_mol"] - native["docking"]["vina"]["score_only_kcal_mol"],
            "vinardo_kcal_mol": guided["docking"]["vinardo"]["score_only_kcal_mol"] - native["docking"]["vinardo"]["score_only_kcal_mol"],
            "force_atom_1": gf[1] - nf[1],
            "force_atom_10": gf[10] - nf[10],
            "force_atom_14": gf[14] - nf[14],
        },
        "monitor": monitor,
        "runtime_fidelity": fidelity,
        "historical_exact_comparators": prior,
    }
    write_json(root / "result_summary.json", result)

    routing_rows = []
    case_sections = []
    for case in manifest:
        key = (case["target_id"], case["ligand_id"])
        route = route_by[key]
        routing_rows.append(
            f"<tr><td>{escape(case['target_id'])}</td><td>{escape(case['ligand_id'])}</td>"
            f"<td>{escape(route['status'])}</td><td>{escape(route['selected_architecture'])}</td></tr>"
        )
        final_sdf = dataset / case["target_id"] / case["ligand_id"] / "final/ligand.sdf"
        case_sections.append(f"""
        <section class='case'>
          <div class='case-head'><div><h3>{escape(case['target_id'])} / {escape(case['ligand_id'])}</h3>
          <p><b>Routing:</b> {escape(route['status'])} → {escape(route['selected_architecture'])}</p></div>
          <div class='mol'>{molecule_svg(final_sdf, 'native final')}</div></div>
          <table><thead><tr><th>stage</th><th>localized risks</th><th>pKd</th><th>MMFF strain</th><th>PoseBusters</th></tr></thead>
          <tbody>{case_table(case)}</tbody></table>
          <p class='links'>Artifacts: <code>RewardDesignIR.json</code>, <code>ArchitectureComparison.json</code>, <code>EvidenceLedger.json</code>, bilingual derivation and diagnostics.</p>
        </section>""")

    deltas = result["deltas_guided_minus_native"]
    prior_rows = "".join(
        f"<tr><td>{escape(row['label'])}</td><td>{fmt(row['pkd'])}</td><td>{fmt(row['strain'])}</td><td>{fmt(row['vina'])}</td><td>{fmt(row['vinardo'])}</td></tr>"
        for row in prior
    )
    rejected = "".join(f"<li><code>{escape(k)}</code>: {v}</li>" for k, v in rejection_counts.most_common()) or "<li>none</li>"
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>MolSteer RewardDesignIR Retest</title>
    <style>
    :root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#63708a;--blue:#315efb;--green:#127a55;--orange:#b95d00;--red:#bd2d3a;}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}}
    main{{max-width:1240px;margin:auto;padding:36px}}h1{{font-size:34px;margin:0 0 8px}}h2{{margin-top:0}}h3{{margin:0 0 8px}}
    .lead{{color:var(--muted);font-size:17px;max-width:920px}}section{{background:var(--card);border:1px solid #dfe4ef;border-radius:14px;padding:24px;margin:20px 0;box-shadow:0 4px 18px #17305b0b}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}.metric{{padding:16px;border-radius:10px;background:#f2f5fb}}
    .metric b{{display:block;font-size:22px}}.good{{color:var(--green)}}.bad{{color:var(--red)}}.warn{{color:var(--orange)}}
    table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #e5e9f2;padding:9px 10px;text-align:left;vertical-align:top}}th{{background:#f7f9fd}}
    .case-head,.pair{{display:grid;grid-template-columns:1fr 460px;gap:18px;align-items:center}}.mol svg{{width:100%;height:auto}}code{{font-size:13px}}.links{{color:var(--muted)}}
    .tag{{display:inline-block;padding:3px 9px;border-radius:99px;background:#e9efff;color:#234ec4;font-weight:600}}details{{margin-top:12px}}
    @media(max-width:800px){{main{{padding:18px}}.case-head,.pair{{grid-template-columns:1fr}}}}
    </style></head><body><main>
    <h1>MolSteer 更新版 MolThinker Skill：精确检查点复测</h1>
    <p class='lead'>数据集 <code>crossdocked_100target_stage_test_exact_20260923</code>，共 2 个靶点、6 个 ligand。每个 t=0.50 决策独立生成 StatePacket、DiagnosticReport、EvidenceLedger、RewardDesignIR 和五类架构比较；只有通过证据门控的样本进入真实 t=0.50→1.0 续推。</p>
    <section><h2>路由结论</h2><div class='grid'>
      <div class='metric'><span>样本</span><b>{len(manifest)}</b></div><div class='metric'><span>无引导自行修复且无终态风险</span><b>{native_repair_count}</b></div>
      <div class='metric'><span>终态筛选警示、无可信坐标目标</span><b>{screen_only_count}</b></div><div class='metric'><span>执行局部引导</span><b>{guided_count}</b></div>
    </div><p>该路由避免把无引导轨迹本来会完成的修复计入奖励收益，也避免把 PAINS/BRENK 类警示直接伪装成坐标能量。</p>
    <table><thead><tr><th>target</th><th>ligand</th><th>decision</th><th>selected architecture</th></tr></thead><tbody>{''.join(routing_rows)}</tbody></table></section>
    <section><h2>执行奖励</h2><p><span class='tag'>staged / lexicographic</span></p>
      <p><code>local geometry → local MMFF relaxation strain → matched-control pKd</code></p>
      <p>区域绑定为原子槽位 <code>[1,10,14]</code> 及一键 halo。只有当前优先级目标达标后才允许优化下一目标；化学有效性、新增严重碰撞、区域外位移、路径预算以及定期 Vina/MMFF 反证均不可由 pKd 增益抵消。程序中的三个单位权重仅满足执行器接口校验，分阶段执行器不会对三项求和。</p>
    </section>
    <section><h2>5i0b / ligand_002 终态结果</h2><div class='grid'>
      <div class='metric'><span>Δ pKd</span><b class='good'>+{fmt(deltas['pkd'])}</b></div>
      <div class='metric'><span>Δ MMFF strain</span><b class='good'>{fmt(deltas['strain_kcal_mol'])}</b><small>kcal/mol</small></div>
      <div class='metric'><span>Δ Vina</span><b class='bad'>+{fmt(deltas['vina_kcal_mol'])}</b><small>kcal/mol (lower is better)</small></div>
      <div class='metric'><span>Δ Vinardo</span><b class='bad'>+{fmt(deltas['vinardo_kcal_mol'])}</b><small>kcal/mol</small></div>
    </div>
    <div class='pair'><div><table><thead><tr><th>metric</th><th>native</th><th>guided</th></tr></thead><tbody>
      <tr><td>pKd</td><td>{fmt(result['final']['native']['pkd'])}</td><td>{fmt(result['final']['guided']['pkd'])}</td></tr>
      <tr><td>MMFF strain</td><td>{fmt(result['final']['native']['strain'])}</td><td>{fmt(result['final']['guided']['strain'])}</td></tr>
      <tr><td>Vina</td><td>{fmt(result['final']['native']['vina'])}</td><td>{fmt(result['final']['guided']['vina'])}</td></tr>
      <tr><td>Vinardo</td><td>{fmt(result['final']['native']['vinardo'])}</td><td>{fmt(result['final']['guided']['vinardo'])}</td></tr>
      <tr><td>force atom 1</td><td>{fmt(nf[1])}</td><td>{fmt(gf[1])}</td></tr>
      <tr><td>force atom 10</td><td>{fmt(nf[10])}</td><td>{fmt(gf[10])}</td></tr>
      <tr><td>force atom 14</td><td>{fmt(nf[14])}</td><td>{fmt(gf[14])}</td></tr>
    </tbody></table><p><b>判定：</b>pKd 和总应变得到小幅改善，化学图、SMILES、界面接触类别及 PoseBusters 通过状态保持。Vina/Vinardo 略退化，且原子 10 的局部力上升，因此是部分改善，不能报告为局部缺陷已完全修复。</p></div>
    <div>{molecule_svg(guided_sdf, 'updated skill guided final')}</div></div>
    <p>坐标差异集中在诊断区域：全分子 world RMSD {fmt(comparison_context['comparisons']['updated_skill_staged_eta100']['world_rmsd_angstrom'])} Å，局部 `[1,10,14]` RMSD {fmt(next(r['rmsd_angstrom'] for r in comparison_context['comparisons']['updated_skill_staged_eta100']['regions'] if r['atom_ids']==[1,10,14]))} Å。</p></section>
    <section><h2>MolMonitor 记录</h2><div class='grid'>
      <div class='metric'><span>accepted steps</span><b>{execution['accepted_steps']} / {execution['steps']}</b></div>
      <div class='metric'><span>max injected path</span><b>{fmt(execution['max_injected_path_angstrom'])} Å</b></div>
      <div class='metric'><span>independent oracle calls</span><b>{execution['oracle_calls']}</b></div>
      <div class='metric'><span>eta</span><b>100</b><small>未动态升高</small></div>
    </div><p>阶段计数：<code>{escape(json.dumps(execution['stage_counts']))}</code>；按同时间优先级比较接受提案，定期审查独立 Vina 与全局应变。</p>
    <details><summary>提案拒绝原因</summary><ul>{rejected}</ul></details></section>
    <section><h2>历史精确续推对照</h2><p>以下三项来自上一轮同一检查点的已验证实验，仅用于解释新版架构的取舍，本轮没有把它们重新计作新实验。</p>
      <table><thead><tr><th>arm</th><th>pKd</th><th>strain</th><th>Vina</th><th>Vinardo</th></tr></thead><tbody>{prior_rows}</tbody></table>
      <p>旧 direct-sum η=300 在该单例上仍给出更好的四项终态数值；新版 skill 拒绝把这项单例结果泛化成默认架构，因为其目标可相互补偿、区域归因较弱。新版结果的价值是把可解释的局部优先级和不可补偿约束落实到执行中，但本次独立打分也证明它尚未解决区域内力重分配。</p></section>
    <section><h2>逐样本证据</h2>{''.join(case_sections)}</section>
    <section><h2>完整性与限制</h2><ul>
      <li>RNG 与时间网格同无引导终态完全一致；非目标 batch 的 coords/atomics/bonds/charges/mask 均逐张量相等。</li>
      <li>没有随机种子扫描；只有一个样本通过执行门控，结论不能解释为跨靶点效果估计。</li>
      <li>MMFF、Vina/Vinardo 和 affinity head 都是模型或经验代理；当前证据不等于实验结合自由能。</li>
      <li>Researcher 未激活：已有证据足以完成坐标路由，而持久 iodine 筛选警示缺少可验证的替代化学图，外部检索不能代替候选生成与验证。</li>
    </ul></section>
    <footer><p>Report SHA-256 is recorded in <code>artifact_manifest.json</code> after rendering.</p></footer>
    </main></body></html>"""
    report = root / "report.html"
    report.write_text(html, encoding="utf-8")
    files = [
        root / "collection_manifest.json", root / "reward_design_summary.json", root / "result_summary.json",
        root / "monitor_summary.json", root / "runtime_fidelity.json", root / "evaluation/ReaderComparisonContext.json", report,
    ]
    write_json(root / "artifact_manifest.json", {"files": [{"path": str(p.relative_to(root)), "sha256": file_hash(p)} for p in files]})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
