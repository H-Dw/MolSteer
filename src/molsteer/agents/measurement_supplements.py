"""Read-only use of existing MolReader calculators, with separate evidence artifacts."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from langchain_core.tools import tool
from molsteer.common import digest


class MeasurementSupplements:
    def __init__(self, packet, raw, state, root, output, stage_paths=None, contexts=None):
        self.packet, self.raw, self.root = packet, raw, Path(root).resolve()
        self.output = Path(output).resolve()
        if self.output.is_relative_to(self.root / 'data'):
            raise ValueError('Evidence supplements must be outside data')
        self.stages, self.contexts = stage_paths or {}, contexts or {}
        self.records = state.setdefault('evidence_supplements', {})

    def measure(self, node_id, metric_id, view, atom_ids):
        from molreader.metrics import METRICS
        if metric_id not in METRICS:
            return dict(status='unavailable', blocking=False, missing=['Existing calculator: '+metric_id])
        if node_id == 'anchor':
            packet = self.packet
        else:
            node = next((n for n in self.raw.get('nodes', []) if n['node_id'] == node_id), None)
            if not node or not node.get('source') or node.get('status') != 'available':
                return dict(status='unavailable', blocking=False, missing=['Bound raw node packet'])
            path = (self.root / node['source']['path']).resolve()
            if not path.is_relative_to(self.root):
                return dict(status='unavailable', blocking=False, missing=['Repository-bound node source'])
            packet = json.loads(path.read_text(encoding='utf-8'))
            if digest(packet) != node['content_hash']:
                return dict(status='unavailable', blocking=False, missing=['Unchanged node source'])
        snapshot = packet['steering']['coordinate_snapshots'].get(view)
        if not snapshot or not set(atom_ids) <= set(snapshot['atom_ids']):
            return dict(status='unavailable', blocking=False, missing=['Bound view and atom mapping'])
        context = self.contexts.get((node_id, view))
        if context is None:
            relative = self.stages.get(node_id)
            if not relative:
                return dict(status='unavailable', blocking=False, missing=['Host-configured stage inputs for '+node_id])
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                return dict(status='unavailable', blocking=False, missing=['Repository-bound stage directory'])
            from molreader.io import load_stage
            try:
                context = load_stage(path, view=view)
            except Exception as exc:
                return dict(status='unavailable', blocking=False, missing=['Stage loading: '+type(exc).__name__])
        if (context.identity != packet['identity'] or list(context.atom_ids) != snapshot['atom_ids']
                or not np.array_equal(context.coords, np.asarray(snapshot['coords_angstrom']))):
            return dict(status='unavailable', blocking=False, missing=['Exact identity, coordinates and atom mapping binding'])
        binding = dict(packet_id=packet['packet_id'], packet_hash=digest(packet), node_id=node_id,
            coordinate_hash=snapshot['coordinate_hash'], configuration_hash=digest(context.config),
            source_hashes={k:v['sha256'] for k,v in context.sources.items()}, view=view, metric_id=metric_id,
            atom_ids=atom_ids, evidence_role='current_supplement' if node_id == 'anchor' else 'raw_reference_supplement')
        ident = 'ms_' + digest(binding)[:24]
        cached = ident in self.records
        if not cached:
            from molreader.packet import compute_metric
            measurement = compute_metric(deepcopy(context), metric_id)
            record = dict(kind='MeasurementSupplement', supplement_id=ident, binding=binding, measurement=measurement)
            record['content_hash'] = digest(record)
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / (ident+'.json')).write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            self.records[ident] = record
        return dict(status=self.records[ident]['measurement']['status'], supplement_id=ident, cached=cached,
            content_hash=self.records[ident]['content_hash'], binding=binding,
            note='Separate evidence supplement. The original packet and current DiagnosticReport are unchanged; future evidence is contextual, never current numerical binding.')

    def tools(self):
        @tool
        def measure_evidence_gap(metric_id: str, node_id: str = 'anchor', view: str = 'prediction', atom_ids: list[int] | None = None) -> dict:
            """Run an existing calculator on host-bound saved inputs; save a separate supplement or report missing inputs."""
            return self.measure(node_id, metric_id, view, atom_ids or [])

        @tool
        def read_measurement_supplement(supplement_id: str, path: list[str | int] | None = None) -> dict:
            """Read one saved supplement or a selected field, preserving source and coordinate bindings."""
            value = self.records.get(supplement_id)
            try:
                for key in path or []:
                    value = value[key]
            except (KeyError, IndexError, TypeError):
                value = None
            return dict(status='available' if value is not None else 'unavailable', value=deepcopy(value))
        return [measure_evidence_gap, read_measurement_supplement]
