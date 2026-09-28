"""Add steering facts without changing existing observations or source identity."""
from copy import deepcopy
import numpy as np
from molreader.packet import validate_packet
from molsteer.common import digest, file_hash
from . import state_capabilities, chemical_readiness, objective_context
from molsteer.contracts import validate_enriched


def enrich_packet(packet, contexts, objectives=None):
    validate_packet(packet)
    for c in contexts:
        if c.identity != packet['identity']:
            raise ValueError('Context/packet identity mismatch')
        for name, source in packet['provenance']['sources'].items():
            if name not in c.sources or c.sources[name]['sha256'] != source['sha256'] or file_hash(c.sources[name]['path']) != source['sha256']:
                raise ValueError(f'Source changed: {name}')
    if {c.view for c in contexts} != set(packet['representations']):
        raise ValueError('All measured representations must be supplied')
    result = deepcopy(packet)
    result['parent_packet_id'] = packet['packet_id']
    result['steering'] = {
        'generation': state_capabilities.compute(packet, contexts),
        'chemical_readiness': chemical_readiness.compute(packet, contexts),
        'objectives': objective_context.compute(objectives),
        'coordinate_snapshots': {c.view: dict(atom_ids=c.atom_ids.tolist(), coords_angstrom=c.coords.tolist(),
            frame=packet['representations'][c.view]['coordinate_frame'], coordinate_hash=digest(c.coords.tolist())) for c in contexts},
        'receptor_atoms': contexts[0].protein,
        'graph_signatures': {c.view: digest(dict(atom_ids=c.atom_ids.tolist(), atoms=c.atoms,
            formal_charges=c.charges, orders=c.orders.tolist())) for c in contexts},
        'categorical_uncertainty': {},
        'scope': 'Snapshot facts; live capability and objective declarations are not inferred',
    }
    for c in contexts:
        if c.view == 'sdf':
            continue
        summaries = {}
        for name, p in c.probs.items():
            if name == 'bonds':
                p = p[np.triu_indices(len(c.atom_ids), 1)]
            entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(-1)
            summaries[name] = dict(evaluated_count=int(entropy.size), mean_entropy_nats=float(entropy.mean()) if entropy.size else None,
                interpretation='Sampled one-hot state is not calibrated certainty' if c.view == 'state' else 'Category ambiguity, not defect probability')
        result['steering']['categorical_uncertainty'][c.view] = summaries
    result['packet_id'] = 'sp_' + digest(dict(parent=packet['packet_id'], steering=result['steering']))[:24]
    validate_enriched(result)
    return result
