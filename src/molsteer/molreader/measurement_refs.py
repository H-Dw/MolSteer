"""Content-bound references to measurements, without manufacturing risk flags.

The packet stays unchanged. References identify existing metric summaries and
localized rows so preservation directions need not invent diagnostic findings.
"""
from molsteer.common import digest
import numpy as np

ROW_KEYS = ('atoms', 'bonds', 'angles', 'torsions', 'per_atom', 'contacts',
            'per_atom_nearest', 'candidates', 'geometry_assigned_centers', 'declared_centers',
            'rings', 'systems')


def measurement_references(packet):
    references = {}
    chemistry = {m['view']:m['values'] for m in packet['observations'] if m['metric_id']=='chemistry_context'}
    for metric in packet['observations']:
        def add(path, values):
            identity = dict(packet_id=packet['packet_id'], view=metric['view'],
                            metric_id=metric['metric_id'], path=path, values=values)
            ident = 'me_' + digest(identity)[:24]
            # A summary is not localized support for an arbitrary coordinate.
            atoms = values.get('atom_ids', []) if path else []
            if path and not atoms and type(values.get('atom_id')) is int:
                atoms = [values['atom_id']]
            references[ident] = dict(evidence_id=ident, view=metric['view'],
                metric_id=metric['metric_id'], status=metric['status'], path=path,
                kind='measured_row' if path else 'measured_summary',
                method=metric['method'], units=metric['units'], atom_ids=atoms,
                values=values)
            # Verified receptor pair fields are required by the observable API.
            if path:
                for key in ('receptor_serial', 'residue_id'):
                    if key in values:
                        references[ident][key] = values[key]
            return ident
        if metric['status'] not in ('ok', 'partial'):
            continue
        add([], metric['values'])
        for key in ROW_KEYS:
            rows = metric['values'].get(key, [])
            if not isinstance(rows, list):
                continue
            for index, row in enumerate(rows):
                if isinstance(row, dict) and ('atom_ids' in row or 'atom_id' in row):
                    add([key, index], row)
        if metric['metric_id']=='stereochemistry' and metric['view'] in chemistry:
            snapshot=packet['steering']['coordinate_snapshots'][metric['view']]
            coordinates=dict(zip(snapshot['atom_ids'],snapshot['coords_angstrom']))
            for index, center in enumerate(metric['values'].get('geometry_assigned_centers', [])):
                if center['label'] not in ('R','S'):continue
                atom=center['atom_id']
                neighbors=sorted({a for bond in chemistry[metric['view']]['bonds']
                    if bond['bond_order']>0 and atom in bond['atom_ids']
                    for a in bond['atom_ids'] if a!=atom})
                if len(neighbors) not in (3,4):continue
                # Three heavy neighbors: reference at the center. Four explicit
                # neighbors: oriented substituent tetrahedron, in fixed order.
                ids=[atom,*neighbors] if len(neighbors)==3 else neighbors
                q=np.array([coordinates[a] for a in ids],dtype=float)
                volume=float(np.dot(q[1]-q[0],np.cross(q[2]-q[0],q[3]-q[0])))
                if not np.isfinite(volume) or abs(volume)<1e-10:continue
                ident=add(['geometry_assigned_centers',index,'derived_signed_volume'],
                    dict(atom_ids=ids,center_atom_id=atom,geometry_assigned_label=center['label'],
                         signed_volume_angstrom3=volume,absolute_volume_angstrom3=abs(volume),
                         derivation='(q1-q0) dot ((q2-q0) cross (q3-q0)); fixed listed atom order. Preserves the measured geometry, not a supplied reference stereochemistry.'))
                references[ident]['kind']='derived_geometry'
                references[ident]['coordinate_hash']=snapshot['coordinate_hash']
                references[ident]['units']={'volume':'angstrom^3'}
    return references
