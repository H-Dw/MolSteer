"""Reproducible baseline report; an agent may refine wording using the skill."""
from .packet import validate_packet,canonical_hash
from .core import write_json

CATEGORIES={
 'tensor_integrity':'evidence_integrity','decode_consistency':'evidence_integrity',
 'atom_confidence':'model_uncertainty','bond_confidence':'model_uncertainty','charge_confidence':'model_uncertainty',
 'atom_inventory':'chemical_identity','formal_charge':'chemical_identity','connectivity':'graph_connectivity','valence':'chemical_validity',
 'bond_lengths':'local_geometry','bond_angles':'local_geometry','ring_planarity':'local_geometry','double_bond_planarity':'local_geometry',
 'intramolecular_clashes':'intramolecular_sterics','stereochemistry':'stereochemical_consistency','structural_alerts':'physicochemical_screening',
 'mmff_strain':'conformational_strain','protein_clashes':'binding_interface_sterics','protein_contacts':'binding_interface_contact',
 'affinity':'prediction_integrity','posebusters':'pose_plausibility'
}


def make_report(packet):
    validate_packet(packet)
    findings=[];gaps=[]
    for m in packet['observations']:
        if m['status'] in ('unavailable','error','partial'):
            gaps.append(dict(view=m['view'],metric_id=m['metric_id'],status=m['status'],reason='; '.join(m['notes']) or 'Partial coverage'))
        if not m['evidence'] or m['status'] in ('unavailable','error'):continue
        severity='high' if any(e['severity']=='high' for e in m['evidence']) else 'warning'
        scope={'state':'observed_noisy_state','prediction':'predicted_endpoint','sdf':'decoded_endpoint'}[m['view']]
        provisional=packet['identity']['stage_t']<1 or m['status']=='partial' or m['metric_id'] in ('structural_alerts','atom_confidence','bond_confidence','charge_confidence')
        finding=dict(finding_id='risk_'+canonical_hash([packet['packet_id'],m['view'],m['metric_id']])[:16],
                     category=CATEGORIES.get(m['metric_id'],'measurement_risk'),view=m['view'],scope=scope,severity=severity,
                     evidence_strength='provisional_screening' if provisional else 'direct_computational_check',
                     calibrated_probability=None,metric_id=m['metric_id'],observation_count=len(m['evidence']),
                     summary=m['evidence'][0]['message'],evidence_ids=[e['evidence_id'] for e in m['evidence']],
                     localized_examples=m['evidence'][:5],thresholds=m['thresholds'],units=m['units'],
                     interpretation_limit=('Current-stage evidence does not determine the eventual clean molecule.' if packet['identity']['stage_t']<1 else
                                           'This is a computational screening observation, not experimental validation.'))
        findings.append(finding)
    coverage=packet['coverage']
    state='risks_observed' if findings else 'insufficient_evidence' if gaps else 'no_flagged_risks_in_evaluated_metrics'
    report=dict(schema_version='1.0.0',kind='DiagnosticReport',packet_id=packet['packet_id'],identity=packet['identity'],
                assessment=state,summary=f"{len(findings)} representation-specific risk groups; {len(gaps)} incomplete or unavailable observations.",
                findings=findings,evidence_gaps=gaps,coverage=coverage,history=packet['history'],limitations=packet['limitations'])
    validate_report(report,packet)
    return report


def validate_report(report,packet):
    if report.get('schema_version')=='2.0.0':
        from .localized_report import validate_localized_report
        return validate_localized_report(report,packet)
    allowed={'schema_version','kind','packet_id','identity','assessment','summary','findings','evidence_gaps','coverage','history','limitations'}
    if set(report)!=allowed:raise ValueError('DiagnosticReport fields must be risk-only contract fields')
    if report['kind']!='DiagnosticReport' or report['packet_id']!=packet['packet_id'] or report['identity']!=packet['identity']:raise ValueError('Report/packet mismatch')
    lookup={e['evidence_id']:(m,e) for m in packet['observations'] for e in m['evidence']}
    for finding in report['findings']:
        if finding.get('calibrated_probability') is not None:raise ValueError('No calibrated defect-probability model is registered')
        if not finding['evidence_ids']:raise ValueError('A finding requires evidence')
        for eid in finding['evidence_ids']:
            if eid not in lookup:raise ValueError('Unresolvable diagnostic evidence')
            m,_=lookup[eid]
            if m['view']!=finding['view'] or m['metric_id']!=finding['metric_id']:raise ValueError('Evidence is assigned to the wrong view/metric')
        for e in finding['localized_examples']:
            if e['evidence_id'] not in finding['evidence_ids'] or e!=lookup[e['evidence_id']][1]:raise ValueError('Evidence values were altered')
    return True


def render_markdown(report):
    def display(value):
        if isinstance(value,float):return f'{value:.5g}'
        if isinstance(value,list):return ', '.join(display(v) for v in value)
        return str(value)
    def label(key):
        return key.replace('_angstrom2',' (Å²)').replace('_angstrom',' (Å)').replace('_kcal_mol',' (kcal/mol)').replace('_degrees',' (°)').replace('_',' ')
    identity=report['identity']
    lines=[f"# DiagnosticReport: {identity['target_id']} / {identity['ligand_id']} / {identity['stage']}",
           '',f"Assessment: **{report['assessment']}**. {report['summary']}",'',f"Source packet: `{report['packet_id']}`. Time: {identity['stage_t']}.",'',
           '## Current risks','']
    for f in report['findings']:
        lines += [f"### {f['category']} — {f['view']} ({f['severity']})",'',f"{f['summary']}. Observations: {f['observation_count']}. Evidence strength: {f['evidence_strength']}.",'']
        for e in f['localized_examples']:
            details={k:v for k,v in e.items() if k not in ('message','severity','evidence_id','atom_ids')}
            measured='; '.join(f'{label(k)}: {display(v)}' for k,v in details.items())
            location=f"Ligand atoms {', '.join(map(str,e['atom_ids']))}" if e['atom_ids'] else 'Molecule-level evidence'
            lines.append(f"- {location}. {measured + '. ' if measured else ''}Evidence: `{e['evidence_id']}`.")
        if f['thresholds']:
            lines += ['', 'Screening settings: '+ '; '.join(f'{label(k)}: {display(v)}' for k,v in f['thresholds'].items())+'.']
        lines += ['',f['interpretation_limit'],'']
    if not report['findings']:lines+=['No flagged risk is established by the evaluated evidence.','']
    lines += ['## Evidence gaps','']
    for g in report['evidence_gaps']:
        lines.append(f"- {g['view']}.{g['metric_id']}: {g['status']} — {g['reason']}")
    lines += ['','## Interpretation limits','']+['- '+s for s in report['limitations']]
    return '\n'.join(lines)+'\n'
