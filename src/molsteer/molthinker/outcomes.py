"""Outcome-aware reward composition and deterministic evidence-based revision."""
from copy import deepcopy
import json
from pathlib import Path
from molsteer.common import digest,file_hash
from molsteer.contracts import validate_enriched
from .knowledge import KnowledgeBase


def load_context(packet):
    validate_enriched(packet)
    binding=packet['steering'].get('outcome_context')
    if not binding or binding['status']!='observed':raise ValueError('Bound outcome evidence required')
    path=Path(binding['source']['path'])
    if file_hash(path)!=binding['source']['sha256']:raise ValueError('Outcome evidence file changed')
    data=json.loads(path.read_text())
    expected='rc_'+digest({k:v for k,v in data.items() if k!='packet_id'})[:24]
    if data['packet_id']!=expected or data['packet_id']!=binding['value']['comparison_packet_id']:
        raise ValueError('Outcome evidence identity mismatch')
    if any(data['subject'][k]!=packet['identity'][k] for k in ('target_id','ligand_id')):
        raise ValueError('Outcome molecular identity mismatch')
    if data['reference_label']!='native_final':raise ValueError('Matched unguided outcome must be the primary reference')
    return data


def evidence_ledger(report,context):
    native=next(v for v in context['observations'] if v['label']==context['reference_label'])
    from rdkit import Chem
    path=Path(native['source_sdf'])
    if file_hash(path)!=native['source_sha256']:raise ValueError('Native comparison structure changed')
    mol=Chem.MolFromMolFile(str(path),removeHs=True)
    rows=[]
    for finding in report['findings']:
        old=finding.get('chemical_context',{}).get('atoms',[])
        changed=[a['atom_id'] for a in old if (a['element'],a['formal_charge'])!=(mol.GetAtomWithIdx(a['atom_id']).GetSymbol(),mol.GetAtomWithIdx(a['atom_id']).GetFormalCharge())]
        status='chemical_hypothesis_replaced_by_native_generation' if changed else 'requires_matched_terminal_check'
        ids=finding.get('atom_ids',[])
        rows.append(dict(finding_id=finding['finding_id'],category=finding['category'],atom_ids=ids,
            native_evolution=status,identity_changed_slots=changed,
            carry_original_graph_bound_targets=not bool(changed),
            terminal_local_forces=[v for v in native['forcefield']['atom_force_norms'] if v['atom_id'] in ids],
            interpretation='Old graph-bound defects cannot be credited to guidance when the native generator replaces that graph. Continuous residual pressure is measured separately.'))
    return dict(original_findings=rows,reference_role='matched_unguided_final',
        strain_native_final=native['forcefield']['strain_kcal_mol'],
        terminal_polar_evidence=native.get('polar_context',[]),
        attribution_rule='Compare candidate to the matched native state at the same time, and compare final to final. Do not use intermediate-to-final change as treatment benefit.',
        gradient_caveat='Subtracting a fixed native reference alone changes attribution, not the coordinate derivative. Replacement of obsolete targets, new continuous objectives and counterevidence routing change control.')


def build_outcome_program(base,packet,report,knowledge,reference_path,*,discrete=True,feedback=True):
    context=load_context(packet)
    if report['packet_id'] not in (packet['packet_id'],packet['parent_packet_id']):raise ValueError('Diagnosis lineage mismatch')
    reference_path=Path(reference_path)
    reference=json.loads(reference_path.read_text())
    if reference['subject']!=context['subject'] or not reference['fidelity']['all_exact']:
        raise ValueError('Complete matched native reference required')
    ledger=evidence_ledger(report,context)
    kb=KnowledgeBase(knowledge)
    relevant={'P01','G04','P06','P05','S05'}
    retrieval=[{k:r[k] for k in ('function_id','name_en','formula','role','prerequisites','source')}
        for r in kb.entries if r['function_id'] in relevant]
    p=deepcopy(base);parent=p.pop('program_id')
    p.update(parent_program_id=parent,evaluator='outcome_aware',packet_id=packet['packet_id'],
        original_diagnosis_packet_id=report['packet_id'],native_reference=dict(path=str(reference_path),sha256=file_hash(reference_path)),
        outcome_context=deepcopy(packet['steering']['outcome_context']),evidence_ledger=ledger,retrieval= retrieval,
        retain_initial=False,geometry_scope='all',lambda_graph=0.,
        outcome_weights=dict(affinity=1.,strain=1.,contact=.5,desolvation=.25,pocket=1.),
        outcome_scales=dict(strain_kcal_mol=5.,contact=2.,desolvation=2.),
        active_objectives=['affinity','continuous_mmff_strain','directional_contacts','buried_unsatisfied_polar_proxy'],
        weights=[1.,1.,.5,.25],
        reward='wA*2*tanh((pKd-pKd_native(t))/2) + wS*log((1+S_native/5)/(1+S/5)) + wI*(I-I_native)/2 + wD*(D_native-D)/2 - wP*(P-P_native)',
        derivative_contract='MMFF94s envelope derivative after converged H-only relaxation; Torch directional contact and smooth SASA proxies; live affinity head; discrete graph search and Vina are not autograd terms.',
        gradient_policy=dict(protect='strain',project_conflicting_components=True),
        discrete_search=dict(enabled=discrete,steps=[60,75,85],max_candidates=16,max_commits=3,max_changed_slots=6,
            families=['tautomer','uncharging','reionization','atom_substitution','formal_charge','bond_order'],
            model_probability_is_calibrated=False,ph_population_known=False),
        feedback_policy=dict(enabled=feedback,review_every=10,affinity_gain_trigger=.03,score_worsening_kcal_mol=.15,
            strain_worsening_kcal_mol=2.,conflict_cosine=-.25,conflict_streak=3,max_revisions=3),
        oracle_guard=dict(vina_regression_allowance=.15,strain_regression_allowance=2.),
        constraints=['sanitized_connected_guidance_proposal','no_new_or_worsened_severe_clash','pocket_occupancy',
            'per_step_and_cumulative_injection_budget','independent_score_counterevidence_at_reviews'],
        normalization_origin='Explicit experimental scales, not calibrated energies or uncertainty. Directional and desolvation terms are geometry proxies.',
        inactive_objectives=[dict(name='water_bridges',reason='No water/protonation ensemble'),
            dict(name='atom_count_change',reason='Fixed-slot adapter'),dict(name='calibrated_affinity_uncertainty',reason='Unavailable')])
    p['design_intent']=dict(p.get('design_intent',{}),evidence_policy='Keep the original t=0.50 diagnosis; use provenance-bound matched native outcomes and independent assessments as additional reasoning evidence.')
    p['inherited_diagnostic_terms']=p.pop('terms',[])
    p['region_atom_ids']=packet['representations']['prediction']['original_atom_ids']
    for unused in ('structure_weight','geometry_soft_weight'):p.pop(unused,None)
    p['control_contract']=dict(p['control_contract'],categorical_control='Bounded, explicit current-state hypothesis search; evaluate the actual live predicted graph with matched SC and no sampling RNG consumption',
        microstates='Only hypotheses that round-trip through element, bond and charge tensors; pH populations unknown')
    p['program_id']='rp_'+digest(p)[:24]
    return p


def revise_from_evidence(program,event):
    """A transparent policy, not a fabricated LLM call or automatic eta escalation."""
    reasons=[];policy=program['feedback_policy']
    if event.get('conflict_streak',0)>=policy['conflict_streak']:reasons.append('persistent_affinity_structure_gradient_conflict')
    if event.get('affinity_delta',0)>policy['affinity_gain_trigger']:
        if event.get('vina_delta',0)>policy['score_worsening_kcal_mol']:reasons.append('head_gain_with_independent_score_regression')
        if event.get('strain_delta',0)>policy['strain_worsening_kcal_mol']:reasons.append('head_gain_with_self_strain_regression')
    if not reasons:return None
    child=deepcopy(program);parent=child.pop('program_id')
    weights=child['outcome_weights'];weights['affinity']=max(.125,weights['affinity']*.5)
    weights['strain']=min(4.,weights['strain']*1.25)
    if 'head_gain_with_independent_score_regression' in reasons:
        weights['contact']=min(2.,weights['contact']*1.25);weights['desolvation']=min(1.,weights['desolvation']*1.25)
    child.update(parent_program_id=parent,revision=dict(reasons=reasons,evidence=deepcopy(event),
        operator='deterministic evidence-based MolThinker policy',eta_changed=False,
        validation='Same-time proposal checks, oracle review, final independent assessment; preserve runtime/RNG/budgets'))
    child['weights']=[weights[k] for k in ('affinity','strain','contact','desolvation')]
    child['program_id']='rp_'+digest(child)[:24]
    return child


def render_outcome_program(program,language='en'):
    zh=language=='zh';ledger=program['evidence_ledger']
    lines=['# '+('基于原生终态对照的奖励推导' if zh else 'Outcome-aware reward composition'),'',
        f"Program: {program['program_id']}",f"StatePacket: {program['packet_id']}",'',program['reward'],'',
        ('原生轨迹自身的改善不记作引导收益；所有连续目标按同时间参照计算增量。减去固定基线本身不改变梯度，真正的控制变化来自连续应变、新界面项、图搜索和反馈修订。' if zh else
         'Native self-correction is not attributed to guidance. All continuous objectives report same-time incremental benefit. Subtracting a fixed baseline alone does not change gradients; continuous strain, interface terms, graph search and feedback change control.'),'',
        ('MMFF 使用固定重原子优化氢后的包络梯度；方向接触与埋藏极性为可微几何代理，不能解释为结合自由能。Vina 只用于独立反证与离散候选筛选。' if zh else
         'MMFF uses an envelope derivative after hydrogen-only minimization. Directional contacts and buried polarity are differentiable geometric proxies, not binding free energy. Vina is an independent counterevidence and discrete-selection oracle.'),'',
        ('连续目标权重：' if zh else 'Continuous objective weights: ')+json.dumps(program['outcome_weights']),
        ('离散搜索：' if zh else 'Discrete search: ')+json.dumps(program['discrete_search']),
        ('反馈修订：' if zh else 'Feedback revision: ')+json.dumps(program['feedback_policy']),'',
        ('原生演化证据：' if zh else 'Native-evolution evidence: ')]
    for row in ledger['original_findings']:
        lines.append(f"- {row['finding_id']}: {row['native_evolution']}; atoms={row['atom_ids']}; carry original targets={row['carry_original_graph_bound_targets']}")
    lines+=['',('知识库机制：' if zh else 'Knowledge mechanisms: ')+'; '.join(r['function_id']+' '+r['name_en'] for r in program['retrieval']),
        '',('权重和归一化为显式实验设置；微观状态仅为可表示的候选，不宣称是特定 pH 下的优势态。缺少水桥、受体 ensemble 和校准 affinity 不确定性。需以同检查点续推及独立终态评估验证。' if zh else
             'Weights and scales are explicit experimental settings. Microstates are representable hypotheses, not claimed dominant populations at a pH. Water bridges, receptor ensembles and calibrated affinity uncertainty remain unavailable. Validate matched continuation and independent final outcomes.')]
    return '\n'.join(lines)+'\n'
