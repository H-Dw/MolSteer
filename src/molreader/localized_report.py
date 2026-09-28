"""Version 2: local risk cards, source-preserving aggregation and compact gaps."""
from collections import defaultdict
from itertools import combinations
from .packet import validate_packet,canonical_hash

GEOMETRY={'bond_lengths','bond_angles','ring_planarity','double_bond_planarity','mmff_local_geometry','stereochemistry'}
UNCERTAINTY={'atom_confidence','bond_confidence','charge_confidence'}
CATEGORIES={'valence':'chemical_validity','atom_inventory':'chemical_identity','formal_charge':'chemical_identity',
            'connectivity':'graph_connectivity','intramolecular_clashes':'intramolecular_sterics',
            'protein_clashes':'binding_interface_sterics','protein_contacts':'binding_interface_contact',
            'tensor_integrity':'evidence_integrity','decode_consistency':'evidence_integrity',
            'affinity':'prediction_integrity'}
GLOBAL={'mmff_strain','posebusters'}
LABELS={'local_geometry':'局部几何与化学图一致性','chemical_validity':'价态与声明成键冲突',
        'chemical_identity':'未确定的元素或电荷','graph_connectivity':'分子图断连',
        'intramolecular_sterics':'分子内部空间冲突','binding_interface_sterics':'蛋白界面空间冲突',
        'binding_interface_contact':'口袋接触覆盖','evidence_integrity':'输入或表示一致性风险',
        'prediction_integrity':'预测值有效性','structural_screening':'局部结构筛选警示',
        'model_uncertainty':'局部类别不确定性','conformational_strain':'全分子构象应变',
        'pose_plausibility':'分子级结构检查','raw_graph_validity':'当前噪声图的有效性'}


def index_packet(packet):
    metrics={(m['view'],m['metric_id']):m for m in packet['observations']}
    evidence={e['evidence_id']:dict(view=m['view'],metric_id=m['metric_id'],status=m['status'],method=m['method'],
                                  interpretation_class='parameter_dependent_screen' if m['metric_id'] in ('mmff_local_geometry','mmff_strain') else 'catalog_screen' if m['metric_id']=='structural_alerts' else 'model_category_uncertainty' if m['metric_id'] in UNCERTAINTY else 'partial_computational_check' if m['status']=='partial' else 'direct_computational_check',
                                  thresholds=m['thresholds'],units=m['units'],evidence=e)
              for m in packet['observations'] for e in m['evidence']}
    return metrics,evidence


def rank(item):
    e=item['evidence'];score=0.
    if 'vdw_ratio' in e:score=1-e['vdw_ratio']
    elif 'conservative_valence_lower_bound' in e:score=e['conservative_valence_lower_bound']/e['ordinary_valence_cap']
    elif 'deviation_degrees' in e:score=abs(e['deviation_degrees'])/30.
    elif 'relative_deviation' in e:score=abs(e['relative_deviation'])/.1
    elif 'endpoint_lower_angstrom' in e and e['endpoint_lower_angstrom']:
        score=max(0,(e['endpoint_lower_angstrom']-e['endpoint_distance_angstrom'])/e['endpoint_lower_angstrom'])
    elif 'lower_angstrom' in e and e['lower_angstrom']:
        d=e['distance_angstrom'];score=max(0,(e['lower_angstrom']-d)/e['lower_angstrom'],(d-e['upper_angstrom'])/e['upper_angstrom'])
    return (e['severity']=='high',score,bool(e['atom_ids']),item['view']=='prediction',e['evidence_id'])


def family(metric):
    if metric in GEOMETRY:return 'local_geometry'
    if metric in UNCERTAINTY:return 'model_uncertainty'
    return CATEGORIES.get(metric,{'structural_alerts':'structural_screening','mmff_strain':'conformational_strain','posebusters':'pose_plausibility'}.get(metric,'measurement_risk'))


def cluster(items):
    groups=[]
    for item in items:
        ids=set(item['evidence']['atom_ids'])
        anchors={('ligand',a) for a in ids}
        e=item['evidence']
        if 'receptor_serial' in e:anchors.add(('receptor',e.get('residue_id'),e['receptor_serial']))
        hits=[g for g in groups if anchors and anchors&g['anchors']]
        if not hits:groups.append(dict(atoms=ids,anchors=anchors,items=[item]));continue
        g=hits[0];g['atoms']|=ids;g['anchors']|=anchors;g['items'].append(item)
        for other in hits[1:]:g['atoms']|=other['atoms'];g['anchors']|=other['anchors'];g['items']+=other['items'];groups.remove(other)
    return groups


def matching_context(metrics,view,ids):
    m=metrics.get((view,'chemistry_context'))
    if not m:return dict(atoms=[],bonds=[],previous=None,source_metric=None)
    values=m['values'];ids=set(ids)
    return dict(atoms=[a for a in values['atoms'] if a['atom_id'] in ids],
        bonds=[b for b in values['bonds'] if set(b['atom_ids'])<=ids],
        previous=values.get('previous'),source_metric={'view':view,'metric_id':'chemistry_context'})


def make_card(packet,metrics,category,items,scope):
    primary=[x for x in items if x.get('role','primary')=='primary']
    ids=sorted({a for x in items if x['metric_id'] not in GLOBAL for a in x['evidence']['atom_ids']})
    if category in ('conformational_strain','pose_plausibility'):ids=[]
    views=sorted({x['view'] for x in items})
    view='state' if scope=='observed_noisy_state' else 'prediction' if 'prediction' in views else views[0]
    ctx=matching_context(metrics,view,ids)
    # Prefer original head values when mapped SDF merely rounds coordinates.
    representatives=[];seen=set()
    for x in sorted(primary,key=rank,reverse=True):
        e=x['evidence'];key=(x['metric_id'],tuple(e['atom_ids']),e.get('kind'),e.get('catalog'),e.get('alert'),e.get('problem_type'))
        if key in seen:continue
        seen.add(key)
        equivalent=next((y for y in primary if y['view']=='prediction' and y['metric_id']==x['metric_id'] and
                         tuple(y['evidence']['atom_ids'])==tuple(e['atom_ids']) and
                         all(y['evidence'].get(k)==e.get(k) for k in ('kind','catalog','alert','problem_type'))),x)
        representatives.append(equivalent['evidence']['evidence_id'])
    unique=[x for x in primary if x['view']==view]
    priority='contextual' if category in ('structural_screening','model_uncertainty') else 'high' if category in ('chemical_validity','raw_graph_validity') or any(x['evidence']['severity']=='high' for x in items) else 'elevated'
    previous=ctx.pop('previous')
    trajectory=None
    if previous and ids:
        old={a['atom_id']:a for a in previous['atoms']};new={a['atom_id']:a for a in ctx['atoms']}
        oldb={tuple(b['atom_ids']):b for b in previous['bonds']}
        now=metrics[(view,'chemistry_context')]['values']['bonds']
        changes=[dict(atom_id=a,previous_element=old[a]['element'],element=new[a]['element'],previous_charge=old[a]['formal_charge'],formal_charge=new[a]['formal_charge']) for a in ids if a in old and a in new and any(old[a][k]!=new[a][k] for k in ('element','formal_charge'))]
        edges=[dict(atom_ids=b['atom_ids'],previous_order=oldb[tuple(b['atom_ids'])]['bond_order'],bond_order=b['bond_order']) for b in now if set(b['atom_ids'])<=set(ids) and tuple(b['atom_ids']) in oldb and oldb[tuple(b['atom_ids'])]['bond_order']!=b['bond_order']]
        trajectory=dict(previous_t=previous['identity']['stage_t'],atom_changes=changes,bond_changes=edges,
                        interpretation='Local chemical identity changed; do not claim persistence of the same chemical constraint.' if changes or edges else 'Same local identity in listed atoms and internal bonds; this alone does not prove persistent risk.')
    source_ids=list(dict.fromkeys(x['evidence']['evidence_id'] for x in items))
    molecule_context=[]
    if scope!='observed_noisy_state':
        for name,keys in [('valence',['sanitized']),('mmff_strain',['strain_proxy_kcal_mol','hydrogen_relaxation_status','minimization_status']),('posebusters',['all_reported_checks_pass'])]:
            m=metrics.get((view,name))
            if m:molecule_context.append(dict(view=view,metric_id=name,status=m['status'],values={k:m['values'].get(k) for k in keys},thresholds=m['thresholds']))
    return dict(finding_id='risk_'+canonical_hash([packet['packet_id'],scope,category,source_ids])[:16],
        category=category,scope=scope,priority=priority,views=views,atom_ids=ids,chemical_context=ctx,
        primary_evidence_ids=[x['evidence']['evidence_id'] for x in primary],
        supporting_evidence_ids=[x['evidence']['evidence_id'] for x in items if x.get('role')=='support'],
        evidence_ids=source_ids,representative_evidence_ids=representatives[:4],
        unique_primary_observation_count=len(unique),source_observation_count=len(items),
        trajectory=trajectory,molecule_context=molecule_context,calibrated_probability=None,
        interpretation_limit='Current-state measurements do not establish final persistence. MMFF and substructure references are screening models, not experimental chemical feasibility. Global energy and PoseBusters findings are molecule-level support, not localized energy attribution.')


def gap_groups(packet):
    groups={}
    for m in packet['observations']:
        if m['status'] not in ('partial','unavailable','error'):continue
        name=m['metric_id'];view=m['view'];rep=packet['representations'][view]
        if name in ('hydrogen_bond_candidates','hydrophobic_contacts','salt_bridge_candidates','prolif'):key='receptor_preparation'
        elif name=='vina_score':key='vina_not_available'
        elif name=='intramolecular_clashes' and not m['values'].get('checked_pair_count'):key='no_evaluable_nonbonded_pairs'
        elif name in ('bond_lengths','bond_angles') and not rep['sanitized']:key='raw_geometry_without_valid_reference'
        elif not rep['sanitized']:key='unresolved_chemical_graph'
        else:key='metric_'+name
        g=groups.setdefault(key,dict(reason_code=key,affected_observations=[],details=[]))
        g['affected_observations'].append(dict(view=view,metric_id=name,status=m['status']))
        for note in m['notes']:
            if note not in g['details']:g['details'].append(note)
    return list(groups.values())


def make_localized_report(packet):
    validate_packet(packet);metrics,index=index_packet(packet)
    decode=metrics.get(('prediction','decode_consistency'),{})
    mapped=decode.get('values',{}).get('mapping_verified') is True and decode.get('values',{}).get('difference_count')==0
    cards=[];raw=[];appendix=[]
    endpoint_groups=[['prediction','sdf']] if mapped else [['prediction'],['sdf']]
    for views in endpoint_groups:
        available=[x for x in index.values() if x['view'] in views]
        if not available:continue
        initial=[];remaining=[]
        for x in available:
            m=x['metric_id']
            if m in UNCERTAINTY|GLOBAL|{'structural_alerts'}:remaining.append(x)
            else:initial.append(x)
        local=[]
        by_family=defaultdict(list)
        for x in initial:by_family[family(x['metric_id'])].append(x)
        for cat,items in by_family.items():
            for g in cluster(items):local.append(dict(category=cat,items=g['items'],atoms=g['atoms']))
        for cat in ('structural_screening','model_uncertainty'):
            for group in cluster([x for x in remaining if family(x['metric_id'])==cat]):
                hits=[g for g in local if group['atoms']&g['atoms']]
                if hits:
                    g=max(hits,key=lambda g:len(g['atoms']&group['atoms']))
                    g['items'] += [dict(x,role='support') for x in group['items']]
                    g['atoms'] |= group['atoms']
                else:local.append(dict(category=cat,items=group['items'],atoms=group['atoms']))
        global_groups=defaultdict(list)
        for x in remaining:
            if x['metric_id'] not in GLOBAL:continue
            related={'bond_angles':'local_geometry','bond_lengths':'local_geometry','internal_steric_clash':'intramolecular_sterics','all_atoms_connected':'graph_connectivity'}.get(x['evidence'].get('check')) if x['metric_id']=='posebusters' else 'local_geometry'
            hits=[g for g in local if g['category']==related]
            if hits:
                # Preserve molecular scope, and never use displacement atom IDs
                # to attribute global strain to an individual region.
                hits[0]['items'].append(dict(x,role='support'))
            else:global_groups[family(x['metric_id'])].append(x)
        for cat,items in global_groups.items():local.append(dict(category=cat,items=items,atoms=set()))
        cards += [make_card(packet,metrics,g['category'],g['items'],'predicted_endpoint' if 'prediction' in views else 'decoded_endpoint') for g in local]
    states=[x for x in index.values() if x['view']=='state']
    graph=[x for x in states if x['metric_id'] in ('valence','atom_inventory','formal_charge')]
    if graph:raw.append(make_card(packet,metrics,'raw_graph_validity',graph,'observed_noisy_state'))
    for cat in ('binding_interface_sterics','intramolecular_sterics','evidence_integrity'):
        items=[x for x in states if family(x['metric_id'])==cat]
        for g in cluster(items):raw.append(make_card(packet,metrics,cat,g['items'],'observed_noisy_state'))
    used={e for c in cards+raw for e in c['evidence_ids']}
    appendix=[eid for eid in index if eid not in used]
    priorities={'high':0,'elevated':1,'contextual':2}
    sort_key=lambda c:(priorities[c['priority']],c['category'],-max(rank(index[eid])[1] for eid in c['primary_evidence_ids']),c['atom_ids'])
    cards.sort(key=sort_key)
    raw.sort(key=sort_key)
    evaluated=any(m['status']=='ok' and m['metric_id'] in GEOMETRY|{'valence','connectivity','protein_clashes'} for m in packet['observations'])
    assessment='risks_observed' if cards or raw or appendix else 'no_flagged_risks_in_evaluated_metrics' if evaluated else 'insufficient_evidence'
    report=dict(schema_version='2.0.0',kind='DiagnosticReport',packet_id=packet['packet_id'],identity=packet['identity'],
        assessment=assessment,summary=f'{len(cards)} endpoint risk regions; {len(raw)} raw-state groups; {len(gap_groups(packet))} coverage root causes.',
        findings=cards,raw_state_findings=raw,evidence_gaps=gap_groups(packet),coverage=packet['coverage'],
        history=packet['history'],evidence_index=index,
        appendix=dict(evidence_ids=appendix,reason='Raw-state derived or remaining evidence retained without dominating endpoint diagnosis.'),
        representation_merging=dict(prediction_sdf_merged=mapped,rule='Verified atom mapping, coordinates, formal charges and graph identity; SDF is not independent corroboration.'),
        limitations=['Atom IDs are zero-based tensor slots; element and bond roles can change across stages.',
                    'Raw X_t observations and head endpoint estimates describe different representations.',
                    'MMFF local thresholds and global strain thresholds are uncalibrated prioritization screens.',
                    'Missing or zero-evaluable measurements are not passes. Optional tool gaps do not invalidate completed core checks.',
                    'Structural filter names are catalog labels; matches do not establish toxicity, chemical instability or poor experimental affinity.'])
    validate_localized_report(report,packet)
    return report


def validate_localized_report(report,packet):
    allowed={'schema_version','kind','packet_id','identity','assessment','summary','findings','raw_state_findings','evidence_gaps','coverage','history','evidence_index','appendix','representation_merging','limitations'}
    if set(report)!=allowed or report['schema_version']!='2.0.0' or report['kind']!='DiagnosticReport':raise ValueError('Invalid risk-only v2 report fields')
    if report['packet_id']!=packet['packet_id'] or report['identity']!=packet['identity']:raise ValueError('Report identity mismatch')
    metrics,index=index_packet(packet)
    decode=metrics.get(('prediction','decode_consistency'),{}).get('values',{})
    mapped=decode.get('mapping_verified') is True and decode.get('difference_count')==0
    if report['representation_merging']['prediction_sdf_merged']!=mapped:raise ValueError('Unverified representation merging')
    if report['evidence_index']!=index:raise ValueError('Evidence or metric metadata altered')
    if report['coverage']!=packet['coverage'] or report['history']!=packet['history']:raise ValueError('Coverage/history altered')
    assigned=[]
    for c in report['findings']+report['raw_state_findings']:
        if set(c)!={'finding_id','category','scope','priority','views','atom_ids','chemical_context','primary_evidence_ids','supporting_evidence_ids','evidence_ids','representative_evidence_ids','unique_primary_observation_count','source_observation_count','trajectory','molecule_context','calibrated_probability','interpretation_limit'}:raise ValueError('Invalid risk-only card fields')
        if c['calibrated_probability'] is not None:raise ValueError('Uncalibrated probability')
        if set(c['evidence_ids'])!=set(c['primary_evidence_ids']+c['supporting_evidence_ids']):raise ValueError('Evidence role mismatch')
        if not set(c['representative_evidence_ids'])<=set(c['primary_evidence_ids']):raise ValueError('Representative lacks primary evidence')
        if not c['evidence_ids']:raise ValueError('Empty risk card')
        if any(eid not in index for eid in c['evidence_ids']):raise ValueError('Unresolvable evidence')
        if c['scope']=='observed_noisy_state' and any(index[eid]['view']!='state' for eid in c['evidence_ids']):raise ValueError('Mixed raw and endpoint evidence')
        if c['scope']!='observed_noisy_state' and any(index[eid]['view']=='state' for eid in c['evidence_ids']):raise ValueError('Mixed raw and endpoint evidence')
        if set(c['views'])!={index[eid]['view'] for eid in c['evidence_ids']}:raise ValueError('Incorrect source views')
        if {'prediction','sdf'}<=set(c['views']) and not mapped:raise ValueError('Unverified representation merging in a card')
        view='state' if c['scope']=='observed_noisy_state' else 'prediction' if 'prediction' in c['views'] else 'sdf'
        ctx=matching_context(metrics,view,c['atom_ids']);ctx.pop('previous')
        if c['chemical_context']!=ctx:raise ValueError('Chemical identity or numeric context altered')
        items=[dict(index[eid],role='support' if eid in c['supporting_evidence_ids'] else 'primary') for eid in c['evidence_ids']]
        expected=make_card(packet,metrics,c['category'],items,c['scope'])
        for key in ('atom_ids','trajectory','molecule_context','source_observation_count','unique_primary_observation_count'):
            if c[key]!=expected[key]:raise ValueError('Risk location, trajectory or source count altered')
        assigned+=c['evidence_ids']
    assigned+=report['appendix']['evidence_ids']
    if len(assigned)!=len(set(assigned)) or set(assigned)!=set(index):raise ValueError('Evidence duplicated across cards or lost')
    if report['evidence_gaps']!=gap_groups(packet):raise ValueError('Coverage gaps changed')
    return True


def atom_label(a):
    symbol=a['element'] or 'PAD';q=a['formal_charge']
    charge='?' if q is None else f'{q:+d}' if q else '0'
    extra=f",隐式H{a['implicit_hydrogens']}" if symbol=='N' and q and 'implicit_hydrogens' in a else ''
    return f"{symbol}({a['atom_id']},电荷{charge}{extra})"


def measurement_text(item):
    e=item['evidence'];ids=','.join(map(str,e['atom_ids']));m=item['metric_id'];prefix=f'原子 [{ids}]'
    if m=='mmff_local_geometry':
        if e['kind']=='bond_angle':return f"{prefix}，中心 {e['center_atom_id']}：角度 **{e['angle_degrees']:.3f}°**；MMFF参考 {e['reference_degrees']:.3f}°，偏差 {e['deviation_degrees']:+.3f}°。"
        return f"{prefix}：键级 {e['bond_order']:g}，键长 **{e['distance_angstrom']:.4f} Å**；MMFF参考 {e['reference_angstrom']:.4f} Å，相对偏差 {e['relative_deviation']:+.1%}。"
    if m=='bond_angles':return f"{prefix}，中心 {e['atom_ids'][1]}：角度 **{e['angle_degrees']:.3f}°**；端点距离 {e['endpoint_distance_angstrom']:.4f} Å，筛查范围 {e['endpoint_lower_angstrom']:.4f}–{e['endpoint_upper_angstrom']:.4f} Å。"
    if m=='bond_lengths':return f"{prefix}：键长 **{e['distance_angstrom']:.4f} Å**，筛查范围 {e['lower_angstrom']:.4f}–{e['upper_angstrom']:.4f} Å。"
    if 'vdw_ratio' in e:return f"{prefix}"+(f" ↔ {e['residue_id']}:{e['receptor_atom']}" if 'residue_id' in e else '')+f"：距离 **{e['distance_angstrom']:.4f} Å**，vdW比 {e['vdw_ratio']:.4f}，阈值 {item['thresholds'].get('distance_over_vdw_sum_below')}。"
    if e.get('problem_type')=='local_valence_lower_bound':return f"{prefix}，{e['element']}电荷0：声明键级和 **{e['declared_bond_order_sum']:g}**，保守价态需求下界 {e['conservative_valence_lower_bound']:g}，普通价态上限 {e['ordinary_valence_cap']}。"
    if m=='structural_alerts':return f"{prefix}：{e['catalog']} / `{e['alert']}`（子结构筛选匹配）。"
    if m=='mmff_strain':return f"全分子局部弛豫能量下降 **{e['strain_proxy_kcal_mol']:.3f} kcal/mol**，筛查阈值 {item['thresholds']['screening_strain_kcal_mol']:g}；不作局部能量归因。"
    if m in UNCERTAINTY:return f"{prefix}：最大类别概率 {e['confidence']:.3f}，前两类间隔 {e['margin']:.3f}；不是缺陷概率。"
    if m=='posebusters':return f"PoseBusters：`{e['check']}` 未通过（分子级）。"
    if e.get('problem_type')=='graph_unavailable':return '完整化学图不可构建；局部已知原子的矛盾仍单独保留。'
    if m=='formal_charge':return f'{prefix}：形式电荷包含未确定的PAD类别。'
    if m=='atom_inventory':return f'{prefix}：元素身份未确定（PAD类别）。'
    return f"{prefix}：{e['message']}。"


def render_localized_markdown(report):
    i=report['identity'];idx=report['evidence_index'];limits=report['evidence_gaps']
    lines=[f"# DiagnosticReport v2 — {i['target_id']} / {i['ligand_id']} / {i['stage']}",'',
           f"当前评估：**{report['assessment']}**。预测终点 {len(report['findings'])} 个风险区域；原状态 {len(report['raw_state_findings'])} 组风险；覆盖限制按 {len(limits)} 个根因汇总。",'',
           f"证据包：`{report['packet_id']}`。原子编号为原始0-based张量位置。",'',
           '## 预测终点的局部风险','']
    def card(c):
        out=[f"### {LABELS.get(c['category'],c['category'])} · {c['priority']}",'']
        if c['chemical_context']['atoms']:
            displayed=c['chemical_context']['atoms']
            if c['scope']=='observed_noisy_state':
                selected={a for eid in c['representative_evidence_ids'] for a in idx[eid]['evidence']['atom_ids']}
                displayed=[a for a in displayed if a['atom_id'] in selected or a['element'] is None or a['formal_charge'] is None]
            out+=['位置：'+'；'.join(atom_label(a) for a in displayed)+'。','']
            if c['chemical_context']['bonds'] and c['scope']!='observed_noisy_state':
                out+=['当前局部连接：'+'；'.join(f"{b['atom_ids'][0]}–{b['atom_ids'][1]}（键级{b['bond_order']:g}）" for b in c['chemical_context']['bonds'] if b['bond_order'])+'。','']
        for eid in c['representative_evidence_ids']:out+=['- '+measurement_text(idx[eid])]
        support=[idx[eid] for eid in c['supporting_evidence_ids']]
        # Present each related concept once; full source measurements remain JSON.
        shown=set()
        for x in support:
            e=x['evidence'];key=(x['metric_id'],tuple(e['atom_ids']),e.get('alert'),e.get('check'))
            if key in shown:continue
            shown.add(key)
            out+=['- 支持证据：'+measurement_text(x)]
        alerts=[idx[eid] for eid in c['evidence_ids'] if idx[eid]['metric_id']=='structural_alerts']
        if alerts:
            out+=['','重叠规则作为同一局部筛选警示，不作独立缺陷计数；规则名不能直接等同于化学不合理或实验活性差。']
            if any(x['evidence']['alert']=='Oxygen-nitrogen_single_bond' for x in alerts):
                out+=['','Oxygen-nitrogen_single_bond 规则也可匹配 N–N，实际元素和电荷以上述化学上下文为准。']
        if c['scope']!='observed_noisy_state':
            checks={m['metric_id']:m for m in c['molecule_context']}
            facts=[]
            if checks.get('valence',{}).get('values',{}).get('sanitized') is True:facts.append('当前预测图通过RDKit价态/芳香性检查')
            if checks.get('posebusters',{}).get('values',{}).get('all_reported_checks_pass') is True:facts.append('PoseBusters所返回的dock检查通过')
            strain=checks.get('mmff_strain')
            if strain and strain['status']=='ok' and not any(x['metric_id']=='mmff_strain' for x in support) and c['category']!='conformational_strain':
                v=strain['values']['strain_proxy_kcal_mol']
                if v is not None:facts.append(f"全分子局部弛豫能量下降 {v:.3f} kcal/mol（阈值 {strain['thresholds']['screening_strain_kcal_mol']:g}）")
            if facts:out+=['','相关检查背景：'+'；'.join(facts)+'。这些结果不抵消上面的局部筛查信号。']
        relevant_unc=[idx[eid] for eid in c['evidence_ids'] if idx[eid]['metric_id'] in UNCERTAINTY]
        if relevant_unc:
            for a in c['chemical_context']['atoms']:
                if any(x['metric_id']=='atom_confidence' and a['atom_id'] in x['evidence']['atom_ids'] for x in relevant_unc):
                    out+=['',f"原子 {a['atom_id']} 的元素候选："+'；'.join(f"{p['category'] or 'PAD'}={p['probability']:.3f}" for p in a.get('atomics_top3',[])[:2])+'。']
            pairs={tuple(x['evidence']['atom_ids']) for x in relevant_unc if x['metric_id']=='bond_confidence'}
            for b in c['chemical_context']['bonds']:
                if tuple(b['atom_ids']) in pairs:out+=['',f"键 {b['atom_ids']} 的类别候选："+'；'.join(f"键级{p['category']:g}={p['probability']:.3f}" for p in b.get('top3',[])[:2])+'。']
        if c['trajectory'] and c['scope']!='observed_noisy_state':
            t=c['trajectory'];parts=[]
            for a in t['atom_changes']:parts.append(f"原子{a['atom_id']}：{a['previous_element']}({a['previous_charge']})→{a['element']}({a['formal_charge']})")
            for b in t['bond_changes']:parts.append(f"{b['atom_ids']}键级：{b['previous_order']:g}→{b['bond_order']:g}")
            if parts:out+=['',f"相对 t={t['previous_t']:.2f}："+'；'.join(parts)+'。局部化学身份发生变化，不称为同一化学约束持续违反。']
        local=next((idx[eid] for eid in c['evidence_ids'] if idx[eid]['metric_id']=='mmff_local_geometry'),None)
        if local:
            th=local['thresholds']
            out+=['',f"方法限定：局部MMFF筛查采用键长相对偏差>{th['bond_relative_deviation']:.0%}、角度偏差>{th['angle_absolute_deviation_degrees']:g}°的未校准展示阈值；参数可获得不等于适用性已验证，尤其是带电氮环境。"]
        out+=['',f"来源：{' + '.join(c['views'])}；{c['source_observation_count']} 条来源记录已合并，完整证据见 JSON 的 `{c['finding_id']}`。",'']
        return out
    for c in report['findings']:lines+=card(c)
    if not report['findings']:lines+=['当前已评估指标没有建立预测终点风险；覆盖限制见下文。','']
    lines+=['## 当前原始噪声状态 X_t','',
            '以下仅描述当前原状态，不与预测终点合并计数，也不直接推断 final 失败。','']
    for c in report['raw_state_findings']:lines+=card(c)
    omitted=defaultdict(int)
    for eid in report['appendix']['evidence_ids']:
        x=idx[eid];omitted[(x['view'],x['metric_id'])]+=1
    lines+=['## 派生证据附录','',
            '完整数值和全部证据ID保存在 DiagnosticReport.json / StatePacket.json。正文示例按严重性及偏差排序。','']
    for (view,metric),count in omitted.items():lines.append(f'- {view}.{metric}：{count} 条记录；不额外当作独立的预测终点缺陷。')
    lines+=['','## 覆盖限制','']
    names={'receptor_preparation':'受体化学类型、显式氢及质子化未验证；氢键/盐桥等仍为候选，ProLIF未完成',
           'vina_not_available':'没有经过来源验证的Vina结果；亲和力head不能替代对接评分',
           'unresolved_chemical_graph':'当前噪声图包含未知身份或不合理声明成键，依赖完整化学图的指标无法完整计算',
           'raw_geometry_without_valid_reference':'原状态缺少有效拓扑参考；键长使用不区分键级的半径回退，角度不作已通过判断',
           'no_evaluable_nonbonded_pairs':'拓扑排除后没有可评估的非键原子对；碰撞结果未判定'}
    for g in limits:lines.append(f"- **{names.get(g['reason_code'],g['reason_code'])}**。涉及 {len(g['affected_observations'])} 个视图/指标组合，合并记录一次。")
    lines+=['','## 解释边界','',
            '- prediction与SDF仅在身份、化学图和坐标映射已核验时合并；它们不是独立实验复现。',
            '- 局部测量、模型类别歧义、力场筛查与目录规则分别解释；不生成已校准的失败概率。',
            '- 全分子应变和PoseBusters为分子级支持，不将最大位移原子当作能量来源。',
            '- 常规描述符及运行信息留在StatePacket；没有目标区间时不列为分子风险。',
            '- 本报告仅陈述当前风险、证据和限制，不包含分子修改或后续操作建议。','']
    return '\n'.join(lines)
