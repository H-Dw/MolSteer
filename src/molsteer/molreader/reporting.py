"""Parallel language renderings of one evidence-bound risk report."""
import json
from collections import Counter
from molreader.localized_report import render_localized_markdown


def measurement_text(item):
    e=item['evidence'];m=item['metric_id'];prefix=f"Atoms {e['atom_ids']}"
    if m=='mmff_local_geometry':
        if e['kind']=='bond_angle':
            return f"{prefix}, center {e['center_atom_id']}: angle **{e['angle_degrees']:.3f}°**; MMFF reference {e['reference_degrees']:.3f}°, deviation {e['deviation_degrees']:+.3f}°."
        return f"{prefix}, order {e['bond_order']:g}: length **{e['distance_angstrom']:.4f} Å**; MMFF reference {e['reference_angstrom']:.4f} Å, relative deviation {e['relative_deviation']:+.1%}."
    if m=='bond_angles':
        return f"{prefix}, center {e['atom_ids'][1]}: angle **{e['angle_degrees']:.3f}°**; endpoint distance {e['endpoint_distance_angstrom']:.4f} Å versus {e['endpoint_lower_angstrom']:.4f}–{e['endpoint_upper_angstrom']:.4f} Å."
    if m=='bond_lengths':
        return f"{prefix}: length **{e['distance_angstrom']:.4f} Å** versus {e['lower_angstrom']:.4f}–{e['upper_angstrom']:.4f} Å."
    if 'vdw_ratio' in e:
        partner=f" to {e['residue_id']}:{e['receptor_atom']}" if 'residue_id' in e else ''
        return f"{prefix}{partner}: distance **{e['distance_angstrom']:.4f} Å**, vdW ratio {e['vdw_ratio']:.4f}; threshold {item['thresholds'].get('distance_over_vdw_sum_below')}."
    if e.get('problem_type')=='local_valence_lower_bound':
        return f"{prefix}, neutral {e['element']}: declared bond-order sum **{e['declared_bond_order_sum']:g}**; conservative valence-demand lower bound {e['conservative_valence_lower_bound']:g}, ordinary cap {e['ordinary_valence_cap']}."
    if m=='mmff_strain':
        return f"Molecule-level relaxation energy drop **{e['strain_proxy_kcal_mol']:.3f} kcal/mol**; screen threshold {item['thresholds']['screening_strain_kcal_mol']:g}. This is not local energy attribution."
    if m=='structural_alerts':return f"{prefix}: {e['catalog']} / `{e['alert']}` substructure screening match."
    if m.endswith('_confidence'):return f"{prefix}: maximum category probability {e['confidence']:.3f}, top-two margin {e['margin']:.3f}; not a defect probability."
    if m=='posebusters':return f"Molecule-level PoseBusters check `{e['check']}` failed."
    return f"{prefix}: {e['message']}."


def render_diagnostic(report, language='en'):
    if language == 'zh':
        return render_localized_markdown(report).replace('DiagnosticReport v2', 'DiagnosticReport')
    i = report['identity']; index = report['evidence_index']
    lines = [f"# DiagnosticReport — {i['target_id']} / {i['ligand_id']} / {i['stage']}", '',
        f"Assessment: **{report['assessment']}**. {len(report['findings'])} endpoint risk regions; "
        f"{len(report['raw_state_findings'])} raw-state groups; {len(report['evidence_gaps'])} coverage root causes.", '',
        f"StatePacket: `{report['packet_id']}`. Atom IDs are original zero-based tensor slots.", '']
    for title, cards in [('Predicted endpoint risks', report['findings']), ('Current noisy state X_t', report['raw_state_findings'])]:
        lines += ['## '+title, '']
        if title.startswith('Current'):
            lines += ['These observations describe X_t and do not establish final failure.', '']
        if not cards:
            lines += ['No risk established by evaluated observations; unmeasured coverage remains unknown.', '']
        for card in cards:
            lines += [f"### {card['category']} · {card['priority']}", '', f"Location: {card['atom_ids']}; scope: {card['scope']}.", '']
            selected = {a for eid in card['representative_evidence_ids'] for a in index[eid]['evidence']['atom_ids']}
            atoms = card['chemical_context']['atoms']
            if card['scope']=='observed_noisy_state':
                atoms = [a for a in atoms if a['atom_id'] in selected or a['element'] is None]
            if atoms:
                lines += ['| Atom | Element | Formal charge | Declared bond-order sum |', '|---|---|---|---|']
                for a in atoms:
                    lines.append(f"| {a['atom_id']} | {a['element']} | {a['formal_charge']} | {a.get('declared_bond_order_sum')} |")
                lines.append('')
            seen = set()
            for eid in card['representative_evidence_ids'] + card['supporting_evidence_ids']:
                x = index[eid]; e=x['evidence']
                key = x['metric_id'],tuple(e['atom_ids']),e.get('kind'),e.get('alert'),e.get('check')
                if key in seen:
                    continue
                seen.add(key)
                role='Support: ' if eid in card['supporting_evidence_ids'] else ''
                lines += [f"- {role}{measurement_text(x)} Evidence: `{eid}`."]
            if card['scope'] != 'observed_noisy_state':
                lines += ['', 'Local categorical alternatives (probabilities are not defect probabilities):', '']
                for b in card['chemical_context']['bonds']:
                    alternatives='; '.join(f"order {p['category']:g}: {p['probability']:.3f}" for p in b.get('top3',[]))
                    lines.append(f"- Pair {b['atom_ids']}: declared order {b['bond_order']:g}; {b['distance_angstrom']:.6f} Å. Alternatives: {alternatives}.")
                if card['trajectory']:
                    t=card['trajectory'];changes=[]
                    for a in t['atom_changes']:changes.append(f"atom {a['atom_id']}: {a['previous_element']}({a['previous_charge']}) → {a['element']}({a['formal_charge']})")
                    for b in t['bond_changes']:changes.append(f"pair {b['atom_ids']}: order {b['previous_order']:g} → {b['bond_order']:g}")
                    if changes:lines += ['', f"Relative to t={t['previous_t']:.2f}: "+'; '.join(changes)+'. Chemical identity changed; this does not establish persistence of the same constraint.']
                local=next((index[eid] for eid in card['evidence_ids'] if index[eid]['metric_id']=='mmff_local_geometry'),None)
                if local:
                    th=local['thresholds']
                    lines += ['',f"Method limits: MMFF screening thresholds are >{th['bond_relative_deviation']:.0%} bond-length deviation and >{th['angle_absolute_deviation_degrees']:g}° angle deviation. These are uncalibrated screens; parameter applicability, especially for charged nitrogen environments, is not established."]
            if any(index[eid]['metric_id']=='structural_alerts' for eid in card['evidence_ids']):
                lines += ['', 'Overlapping PAINS/BRENK matches are one screening concern, not independent defects. Catalog labels do not establish chemical impossibility or measured activity.']
            lines += ['', card['interpretation_limit'], '', f"Merged sources: {' + '.join(card['views'])}; "
                f"{card['source_observation_count']} source observations. Full card: `{card['finding_id']}`.", '']
    lines += ['## Derived-evidence appendix', '', 'All source observations and evidence IDs remain in DiagnosticReport.json and StatePacket.json.', '']
    counts = Counter((index[e]['view'],index[e]['metric_id']) for e in report['appendix']['evidence_ids'])
    lines += [f'- {view}.{metric}: {count} observations, not additional independent endpoint defects.' for (view,metric),count in counts.items()]
    lines += ['', '## Coverage limitations', '']
    reasons={'receptor_preparation':'Receptor chemical types, explicit hydrogens and protonation unverified; interactions remain candidates and ProLIF is not prepared',
             'vina_not_available':'No provenance-verified Vina score; affinity-head predictions are not docking scores',
             'unresolved_chemical_graph':'Unresolved identities or incompatible declared bonds prevent complete graph-dependent evaluation',
             'raw_geometry_without_valid_reference':'Raw state lacks a valid topology reference; radius fallback is not bond-order specific and angle correctness is unassessed',
             'no_evaluable_nonbonded_pairs':'No nonbonded pairs remain after topology exclusions; absence of evaluated pairs is not a pass'}
    for g in report['evidence_gaps']:
        lines.append(f"- **{reasons.get(g['reason_code'],g['reason_code'])}**: {len(g['affected_observations'])} affected view/metric combinations, consolidated once.")
    lines += ['', '## Interpretation limits', '',
        '- Prediction/SDF merge only after graph, identity and coordinate mapping checks; they are not independent experiments.',
        '- Global relaxation strain and PoseBusters are molecular support, not local energy attribution.',
        '- Routine properties without supplied targets remain in StatePacket. Missing coverage is not a pass.',
        '- This report states risks and evidence only; no molecular edits or subsequent intervention recommendations.', '']
    return '\n'.join(lines)
