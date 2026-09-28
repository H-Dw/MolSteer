from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from . import __version__
from .core import result, Unavailable, clean, sha256, write_json
from .metrics import METRICS

STATUSES = {"ok", "partial", "unavailable", "not_applicable", "error"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def identity_key(identity):
    return f"{identity['target_id']}/{identity['ligand_id']}/{identity['stage']}"


def persistence_signature(packet,metric,e):
    """Same slots alone cannot establish the same chemical constraint over time."""
    ids=e.get('atom_ids',[])
    if not ids:return None
    context=next((m['values'] for m in packet['observations'] if m['view']==metric['view'] and m['metric_id']=='chemistry_context'),None)
    if context is None:return None
    edges=[b for b in context['bonds'] if b['bond_order'] and set(b['atom_ids'])&set(ids)]
    involved=set(ids)|{a for b in edges for a in b['atom_ids']}
    atoms=[(a['atom_id'],a['element'],a['formal_charge']) for a in context['atoms'] if a['atom_id'] in involved]
    if any(a[1] is None or a[2] is None for a in atoms):return None
    return canonical_hash(dict(atom_roles=ids,atoms=atoms,bonds=[(b['atom_ids'],b['bond_order']) for b in edges],
        rule={k:e.get(k) for k in ('catalog','alert','kind','problem_type','check','residue_id','receptor_serial','message')},
        method=metric['method'],thresholds=metric['thresholds']))


def compute_metric(ctx, name):
    start=time.perf_counter()
    try:
        if ctx.view != 'sdf' and not ctx.options.get('_categorical_valid', True) and name not in ('tensor_integrity','shape','affinity','decode_consistency'):
            raise Unavailable('Active categorical input is nonfinite or not a probability distribution')
        if ctx.view != 'sdf' and (not ctx.options.get('_graph_symmetric', True) or ctx.options.get('_self_bonds', False)) and name not in ('tensor_integrity','shape','affinity','decode_consistency','atom_confidence','bond_confidence','charge_confidence','atom_inventory','formal_charge'):
            raise Unavailable('Bond graph is asymmetric or contains self bonds')
        module=importlib.import_module(f"molreader.metrics.{name}")
        data=module.compute(ctx)
    except Unavailable as exc:
        data=result(status="unavailable",notes=[str(exc)])
    except Exception as exc:
        data=result(status="error",notes=[f"{type(exc).__name__}: {exc}"])
    data=clean(data)
    data.update(metric_id=name,metric_version=__version__,view=ctx.view,identity=ctx.identity,
                context_hash=canonical_hash(dict(identity=ctx.identity,view=ctx.view,sources={k:v['sha256'] for k,v in ctx.sources.items()},config=ctx.config)),
                source_hashes={k:v['sha256'] for k,v in ctx.sources.items()},
                config_hash=canonical_hash(ctx.config),elapsed_seconds=time.perf_counter()-start)
    for i,e in enumerate(data['evidence']):
        seed=dict(identity=ctx.identity,view=ctx.view,metric=name,index=i,evidence=e)
        e['evidence_id']='ev_'+canonical_hash(seed)[:20]
    validate_metric(data,ctx.identity)
    return data


def validate_metric(metric,identity):
    required={'metric_id','metric_version','view','identity','context_hash','source_hashes','config_hash','status','values','units','evidence','notes','method','thresholds','elapsed_seconds'}
    if not required.issubset(metric):
        raise ValueError(f"Metric envelope missing {sorted(required-set(metric))}")
    if metric['identity']!=identity:
        raise ValueError('Metric belongs to a different target, ligand or stage')
    if metric['status'] not in STATUSES or metric['view'] not in ('state','prediction','sdf'):
        raise ValueError('Invalid metric status or view')
    if not isinstance(metric['values'],dict) or not isinstance(metric['evidence'],list):
        raise ValueError('Invalid metric values or evidence')
    for e in metric['evidence']:
        if not {'evidence_id','atom_ids','severity','message'}.issubset(e):raise ValueError('Malformed localized evidence')
        if e['severity'] not in ('info','warning','high'):raise ValueError('Unknown evidence severity')
    json.dumps(metric,allow_nan=False)


def build_packet(contexts,metrics=None,external_results=(),history_packet=None):
    if not contexts:raise ValueError('At least one view is required')
    identity=contexts[0].identity
    if any(c.identity!=identity for c in contexts):raise ValueError('Cannot merge different stages')
    if len({c.view for c in contexts})!=len(contexts):raise ValueError('Duplicate representation')
    selected=metrics or METRICS
    if any(n not in METRICS for n in selected):raise ValueError('Unknown metric requested')
    observations=[compute_metric(c,n) for c in contexts for n in selected]
    seen={(x['view'],x['metric_id']) for x in observations}
    provenance={c.view:{k:v['sha256'] for k,v in c.sources.items()} for c in contexts}
    for path in external_results:
        m=json.loads(Path(path).read_text(encoding='utf-8'))
        validate_metric(m,identity)
        if m['view'] not in provenance or m['source_hashes']!=provenance[m['view']]:raise ValueError('External metric input hashes differ from packet inputs')
        c=next(c for c in contexts if c.view==m['view'])
        expected=canonical_hash(dict(identity=c.identity,view=c.view,sources=provenance[c.view],config=c.config))
        if m['context_hash']!=expected or m['config_hash']!=canonical_hash(c.config):raise ValueError('External metric context/config mismatch')
        key=m['view'],m['metric_id']
        if key in seen:raise ValueError(f'Duplicate metric; do not silently overwrite {key}')
        seen.add(key);observations.append(m)
    versions={}
    for name in ('numpy','torch','rdkit','posebusters','prolif'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    packet=dict(schema_version='1.0.0',kind='StatePacket',packet_id='sp_'+canonical_hash(dict(identity=identity,provenance=provenance,config=contexts[0].config,metrics=[(m['view'],m['metric_id'],m['values'],m['evidence'],m['status']) for m in observations]))[:24],
                created_at=datetime.now(timezone.utc).isoformat(),identity=identity,
                representations={c.view:dict(meaning={'state':'Current integrated noisy state X_t','prediction':'Head endpoint estimate X_hat_1(X_t,t)','sdf':'Decoded SDF from head endpoint estimate'}[c.view],
                                            coordinate_unit='angstrom',coordinate_frame='receptor_world' if c.world_valid else 'model_scaled_unaligned',
                                            original_atom_ids=c.atom_ids.tolist(),sdf_mapping_verified=c.sdf_mapping_valid,transform=c.transform,
                                            sanitized=c.mol is not None,sanitize_error=c.sanitize_error) for c in contexts},
                provenance=dict(sources=contexts[0].sources,adapter=contexts[0].config,adapter_hash=canonical_hash(contexts[0].config),
                                generation_telemetry=contexts[0].metadata,telemetry_scope='Batch-level timing and CUDA memory are shared by the three sampled ligands',
                                software=dict(molreader=__version__,python=platform.python_version(),**versions)),
                observations=observations,coverage={s:sum(m['status']==s for m in observations) for s in sorted(STATUSES)},
                history=[],limitations=[
                    'Intermediate X_t is noisy by construction; observations do not prove a final molecule will fail.',
                    'Screening thresholds are versioned defaults, not stage-calibrated probabilities of failure.',
                    'Affinity endpoints are model predictions; missing measurements never count as passed.',
                    'Interface measurements use the supplied cropped pocket and its preparation state.'
                ])
    if history_packet:
        h=history_packet['identity']
        if any(h[k]!=identity[k] for k in ('target_id','ligand_id')) or h['stage_t']>=identity['stage_t']:raise ValueError('Invalid history identity or time')
        prior={(m['view'],m['metric_id']):m for m in history_packet['observations']}
        persistence=[]
        for m in observations:
            p=prior.get((m['view'],m['metric_id']))
            if p is None:continue
            old={persistence_signature(history_packet,p,e) for e in p['evidence']}-{None}
            same=[e['evidence_id'] for e in m['evidence'] if persistence_signature(packet,m,e) in old]
            if same:persistence.append(dict(view=m['view'],metric_id=m['metric_id'],current_evidence_ids=same,previous_count=len(p['evidence']),current_count=len(m['evidence'])))
        packet['history']=[dict(previous_packet_id=history_packet['packet_id'],previous_t=h['stage_t'],persistent_locations=persistence)]
    validate_packet(packet)
    return packet


def validate_packet(packet):
    if packet.get('kind')!='StatePacket' or packet.get('schema_version')!='1.0.0':raise ValueError('Unsupported StatePacket')
    evidence_ids=set();metric_ids=set()
    for m in packet['observations']:
        validate_metric(m,packet['identity'])
        key=m['view'],m['metric_id']
        if key in metric_ids:raise ValueError('Duplicate view/metric')
        metric_ids.add(key)
        allowed=set(packet['representations'][m['view']]['original_atom_ids'])
        for e in m['evidence']:
            if e['evidence_id'] in evidence_ids:raise ValueError('Duplicate evidence ID')
            evidence_ids.add(e['evidence_id'])
            if not set(e['atom_ids']).issubset(allowed):raise ValueError('Evidence references an inactive atom')
    if packet['coverage']!={s:sum(m['status']==s for m in packet['observations']) for s in sorted(STATUSES)}:raise ValueError('Coverage does not match observations')
    json.dumps(packet,allow_nan=False)
    return True
