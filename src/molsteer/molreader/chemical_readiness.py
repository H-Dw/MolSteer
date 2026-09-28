"""Separate chemical, coordinate and preparation readiness per representation."""
from molsteer.common import fact, observation


def compute(packet, contexts):
    result = {}
    for c in contexts:
        row = {}
        valid = c.mol is not None
        row['graph_valid'] = fact(valid, 'observed', f'representations.{c.view}.sanitized', c.sanitize_error)
        # Sanitization establishes consistency for this declared graph, not its physical correctness.
        valence=observation(packet,'valence',c.view)
        local_conflict=valence is not None and any(e.get('problem_type')=='local_valence_lower_bound' for e in valence['evidence'])
        row['valence_ok'] = fact(False, 'derived', f'{c.view}.valence', 'At least one known atom has excessive conservative valence demand') if local_conflict else (
            fact(True, 'derived', f'{c.view}.valence') if valid else fact(reason='Whole graph unresolved and no independently established local valence contradiction'))
        row['aromaticity_consistent'] = fact(True, 'derived', f'{c.view}.valence') if valid else fact(reason='No sanitized graph')
        row['formal_charge_complete'] = fact(all(q is not None for q in c.charges), 'observed', f'{c.view}.formal_charge')
        for name, reason in [('protonation_validated', 'Implicit-H assumptions are not validated protonation'),
                             ('partial_charges_available', 'Formal charges are not partial charges'),
                             ('pocket_forcefield_prepared', 'No verified receptor types, protonation or charge model'),
                             ('reference_stereochemistry', 'No required reference handedness or E/Z assignment')]:
            row[name] = fact(reason=reason)
        row['pocket_coordinates_available'] = fact(bool(c.protein), 'observed', str(c.receptor_path))
        row['pocket_frame_aligned'] = fact(c.world_valid, 'derived', c.transform)
        for name, metric, key in [('component_count', 'connectivity', 'component_count'),
                                  ('bond_len_viol', 'bond_lengths', 'outlier_count'),
                                  ('angle_viol', 'bond_angles', 'outlier_count'),
                                  ('clash_count', 'protein_clashes', 'clash_count')]:
            m = observation(packet, metric, c.view)
            value = m['values'].get(key) if m and m['status'] in ('ok', 'partial') else None
            row[name] = fact(value, 'derived' if value is not None else 'unavailable', f'{c.view}.{metric}')
        clash = observation(packet, 'protein_clashes', c.view)
        checked = clash['values'].get('checked_pair_count', 0) if clash else 0
        row['checked_clash_pair_count'] = fact(checked, 'derived', f'{c.view}.protein_clashes')
        row['max_penetration_angstrom'] = fact(max((e['penetration_angstrom'] for e in clash['evidence']), default=0.),
            'derived', f'{c.view}.protein_clashes') if checked else fact(reason='No evaluated pairs')
        if not checked:
            row['clash_count'] = fact(reason='No evaluated pairs; zero would imply false coverage')
        mmff = observation(packet, 'mmff_local_geometry', c.view)
        row['mmff_local_parameters_available'] = fact(mmff is not None and mmff['status'] == 'ok', 'derived', f'{c.view}.mmff_local_geometry')
        for name, kind in [('mmff_bond_len_viol','bond_length'),('mmff_angle_viol','bond_angle')]:
            row[name]=fact(sum(e.get('kind')==kind for e in mmff['evidence']), 'derived', f'{c.view}.mmff_local_geometry') if mmff and mmff['status']=='ok' else fact(reason='Local MMFF parameter coverage incomplete')
        intra=observation(packet,'intramolecular_clashes',c.view)
        val=intra['values'].get('clash_count') if intra else None
        row['intramolecular_clash_count']=fact(val,'derived' if val is not None else 'unavailable',f'{c.view}.intramolecular_clashes')
        planes=[]
        for metric, collection in [('ring_planarity','rings'),('double_bond_planarity','systems')]:
            m=observation(packet,metric,c.view)
            if m and m['status']=='ok':planes += [r['max_plane_distance_angstrom'] for r in m['values'].get(collection,[])]
        row['plane_dev_max_angstrom']=fact(max(planes),'derived',f'{c.view}.ring_planarity + double_bond_planarity') if planes else fact(reason='No evaluated aromatic-ring or SP2 double-bond planes')
        row['plane_dev_scope']=fact('Evaluated aromatic rings and SP2 double bonds; not a complete amide/conjugation inventory','derived','planarity metric definitions')
        for name, reason in [('chirality_ok','No supplied target handedness'),('sasa_dev','No group area target'),
                             ('shape_coverage','No supplied target point cloud'),('rmsd_to_ref','No designated atom-mapped reference')]:
            row[name]=fact(reason=reason)
        row['mmff_applicability_validated'] = fact(reason='Parameter lookup does not validate the chemical hypothesis')
        result[c.view] = row
    return result
