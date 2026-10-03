"""Live view-faithful execution of version-2 expert controls."""
import torch
from molsteer.common import digest
from molsteer.agents.expert_contracts import validate_expert_spec
from .program import MolecularReward
from .expert_control import ExpertEvaluator
from .chemistry import decode_endpoint
from molsteer.molthinker.expressions import aggregate_objectives


class ExpertReward(MolecularReward):
    def __init__(self, program, baseline, receptor, vocabulary):
        if program.get('program_id')!='rp_'+digest({k:v for k,v in program.items() if k!='program_id'})[:24]:
            raise ValueError('Expert program digest mismatch')
        validate_expert_spec(program['expert_spec'],program['source_packet'],program['source_report'])
        design = program['expert_spec']['mathematical_design']
        if design['strategy']['mode'] != 'scalar_potential' or any(
                d['disposition'] == 'constraint' for d in program['expert_spec']['biology_plan']['directions']):
            raise ValueError('Legacy controller or proposal constraint needs scalar redesign')
        # The expert expression owns its physical scales. Historical MolecularReward
        # guard templates have no meaning in this scalar evaluator.
        self.vocab, self.receptor = vocabulary, receptor
        self.spec=program
        self.evaluator=ExpertEvaluator(program['expert_spec'],program['source_packet'])
        self.uses_state_view=any(o['view']=='state' for d in self.evaluator.directions for o in d['observables'])
        self.uses_state_graph='state' in self.evaluator.mmff or 'state' in self.evaluator.dynamic_views
        self.control_diagnostics={};self.component_gradients=None
        self.reference_policy=program['expert_spec']['mathematical_design'].get('design_audit', {}).get('graph_policy')

    def components(self,pred,state_coords=None,state_graph=None):
        if self.uses_state_view and state_coords is None:
            raise ValueError('Expert expressions require live state-world coordinates')
        coordinates={'prediction':pred['coords'],'state':state_coords}
        for view in {o['view'] for d in self.evaluator.directions for o in d['observables']}:
            ids=self.evaluator.packet['steering']['coordinate_snapshots'][view]['atom_ids']
            if coordinates[view].shape!=(len(ids),3):
                raise ValueError('Expert coordinate slot count changed; explicit mapping required')
        molecules={}; elements={}
        for view in set(self.evaluator.mmff) | self.evaluator.dynamic_views:
            if view=='state' and state_graph is None:
                raise ValueError('Live state categories are required for MMFF')
            current=pred if view=='prediction' else dict(state_graph,coords=state_coords)
            elements[view] = [self.vocab['atomic_tokens'][i] for i in current['atomics'].detach().argmax(-1).cpu().tolist()]
            try:
                molecules[view]=decode_endpoint(current,self.vocab)
            except ValueError:
                molecules[view]=None
        return self.evaluator.components(coordinates,molecules=molecules,elements=elements)

    def evaluate(self,pred,state_coords=None,state_graph=None):
        values=self.components(pred,state_coords,state_graph)
        reward=-aggregate_objectives(self.evaluator.objectives(values),self.evaluator.strategy)
        return reward,{'components':{k:float(v.detach()) for k,v in values.items()},
                       'control_mode':self.evaluator.strategy['mode'],
                       'scalar_is_reporting_only':False,
                       'chemical_applicability': 'Current-chemistry mechanism evaluation; whole-graph identity is not an activation condition',
                       'reference_bindings':self.evaluator.reference_diagnostics,
                       'unavailable_directions':self.evaluator.unavailable,
                       'priority_weights':self.evaluator.strategy.get('priority_weights', {})}

    def state_graph(self,adapter):
        if not self.uses_state_graph:return None
        return {k:adapter.curr[k][adapter.index] for k in ('atomics','charges','bonds')}
