"""Bind counterfactual outcomes without replacing the original diagnosis."""
from copy import deepcopy
from pathlib import Path
import json
from molsteer.common import digest,file_hash
from molsteer.contracts import validate_enriched


def attach(packet,comparison_path):
    validate_enriched(packet)
    path=Path(comparison_path);comparison=json.loads(path.read_text())
    if comparison.get('kind')!='ReaderComparisonContext':raise ValueError('Unsupported outcome evidence')
    expected='rc_'+digest({k:v for k,v in comparison.items() if k!='packet_id'})[:24]
    if comparison['packet_id']!=expected:raise ValueError('Comparison evidence digest mismatch')
    for key in ['target_id','ligand_id']:
        if comparison['subject'][key]!=packet['identity'][key]:raise ValueError('Comparison subject mismatch')
    out=deepcopy(packet);out['parent_packet_id']=packet['packet_id']
    out['steering']['outcome_context']=dict(value=dict(comparison_packet_id=comparison['packet_id'],
        reference_label=comparison['reference_label'],source_packet_ids=comparison['source_packet_ids'],
        subject=comparison['subject']),status='observed',source=dict(path=str(path.resolve()),sha256=file_hash(path)),
        reason='Matched unguided outcome, historical outcome and guided outcomes are distinct roles; the diagnosis observations remain unchanged')
    out['packet_id']='sp_'+digest(dict(parent=out['parent_packet_id'],steering=out['steering']))[:24]
    validate_enriched(out);return out
