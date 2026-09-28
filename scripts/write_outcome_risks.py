"""Produce a risk-only supplement from independent outcome evidence."""
import argparse
import json
from pathlib import Path
from rdkit import Chem
from molsteer.common import write_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    context=json.loads((root/'ReaderComparisonContext.json').read_text())
    row=next(x for x in context['observations'] if x['label']=='adaptive_same_budget')
    mol=Chem.MolFromMolFile(row['source_sdf'],removeHs=True)
    atoms=[dict(atom_id=i,element=mol.GetAtomWithIdx(i).GetSymbol(),formal_charge=mol.GetAtomWithIdx(i).GetFormalCharge(),
        coords_angstrom=list(mol.GetConformer().GetAtomPosition(i))) for i in [1,10,14,2,15,17,19]]
    geometry=row['metrics']['mmff_local_geometry']['values']
    bond=next(x for x in geometry['bonds'] if x['atom_ids']==[1,14])
    risks=[dict(id='residual_geometry',priority='high_review',atom_ids=[1,10,14],measurement=bond,
        forces=[v for v in row['forcefield']['atom_force_norms'] if v['atom_id'] in [1,10,14]],
        certainty='Observed model-relative deviation; chemical applicability of MMFF is not independently validated'),
        dict(id='buried_polar_candidates',priority='conditional',
            locations=[v for v in row['polar_context'] if v['buried_without_detected_polar_contact'] is True],
            certainty='Direct-contact and single-preparation evidence; water-mediated satisfaction is unknown'),
        dict(id='screening_alert',priority='screening',atom_ids=[1,10,14],
            certainty='Consolidated local alert, not a chemical invalidity or toxicity verdict'),
        dict(id='binding_improvement_unconfirmed',priority='evaluation_limit',
            certainty='Small score-only gain disappears after separate local optimization; no experimental binding result')]
    report=dict(kind='OutcomeRiskSupplement',subject=context['subject'],stage='final',candidate='adaptive_same_budget',
        comparison_packet_id=context['packet_id'],source=dict(path=str(root/'ReaderComparisonContext.json'),sha256=file_hash(root/'ReaderComparisonContext.json')),
        atom_indexing='original zero-based tensor slots',atoms=atoms,risks=risks,
        observed_passes=['RDKit sanitization','connected graph','no detected intramolecular or protein clashes','all reported PoseBusters checks'],
        coverage_limits=context['coverage_limits']+['Water-bridge and dynamic-contact coverage unavailable','ADME and experimental affinity unavailable'])
    write_json(root/'ResidualRisks.json',report)
    for lang in ['en','zh-CN']:
        zh=lang=='zh-CN'
        text=['# '+('终态风险补充报告' if zh else 'Final-state risk supplement'),'','`5i0b_A__5vef_M77 / ligand_002 / adaptive_same_budget / final`','',
            ('本报告仅描述风险。补充包身份：' if zh else 'Risk-only report. Comparison packet: ')+context['packet_id'],
            '',('原子使用原始 0-based tensor slot；坐标为保存的世界坐标，单位 Å。' if zh else 'Original zero-based tensor slots; saved world coordinates in Å.'),'',
            '| ID | Element | Formal charge | x | y | z |','|---|---|---:|---:|---:|---:|']
        for v in atoms:text.append(f"| {v['atom_id']} | {v['element']} | {v['formal_charge']} | "+' | '.join(f'{x:.4f}' for x in v['coords_angstrom'])+' |')
        text+=['','## '+('局部几何及应变' if zh else 'Local geometry and strain'),'',
            ('N1–N14=1.071098 Å，MMFF 参照 1.140 Å，偏短 6.04%。N14/N1 分子内力范数为 147.94/123.81 kcal/mol/Å。局部压力持续存在，但未越过原 10% 键长筛查线。N1–N10–C6=123.2097°，参照 113.995°。自应变 15.0768 kcal/mol。参照的适用性尚未独立验证。' if zh else
             'N1–N14=1.071098 Å versus MMFF reference 1.140 Å, 6.04% compressed. Intramolecular force norms on N14/N1 are 147.94/123.81 kcal/mol/Å. Residual pressure persists below the original 10% bond screening threshold. N1–N10–C6=123.2097° versus reference 113.995°. Self-strain is 15.0768 kcal/mol. Reference applicability has not been independently validated.'),
            '', '## '+('埋藏极性候选风险' if zh else 'Candidate buried polar concerns'),'']
        for v in risks[1]['locations']:
            text.append(f"- {v['element']}{v['atom_id']}: {100*v['buried_fraction']:.2f}% "+('埋藏；当前准备条件下未检测到直接极性配对。' if zh else 'buried; no direct polar partner detected under this preparation.'))
        text+=['',('N17 仅略超过 80% 筛查阈值。水桥、微观电离状态及 H 朝向覆盖不足，因此上述结果不是确定的结合缺陷。' if zh else
            'N17 only marginally exceeds the exploratory 80% threshold. Water bridges, microstates and hydrogen orientations are incompletely covered, so these are not established binding defects.'),
            '', '## '+('筛选警示与结论限制' if zh else 'Screening and interpretation limits'),'',
            ('[1,10,14] 的叠氮基命中合并为一个筛选警示，不作为无效结构或已证实毒性的判断。Vina 原姿势小幅获益在独立局部优化后不再保持；真实亲和力改善未被证实。RDKit、连通性、碰撞及已报告的 PoseBusters 检查通过，不能代替前述局部与结合风险判断。' if zh else
            'The azido group [1,10,14] is one consolidated screening warning, not a verdict of invalidity or demonstrated toxicity. The small Vina score-only advantage is not retained after separate local optimization; actual affinity improvement is unconfirmed. RDKit, connectivity, clash and reported PoseBusters passes do not resolve these local or binding concerns.'),
            '', '[Evidence](ResidualRisks.json) · [Full comparison](ReaderComparisonContext.json)']
        (root/f'ResidualRisks.{lang}.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Risk-only supplement written')


if __name__=='__main__':main()
