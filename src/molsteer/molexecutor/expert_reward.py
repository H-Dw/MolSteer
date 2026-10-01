"""Live view-faithful execution of version-2 expert controls."""
import torch
from molsteer.common import digest
from molsteer.agents.expert_contracts import validate_expert_spec
from .program import MolecularReward
from .expert_control import ExpertEvaluator, control_direction, check_displacement
from .chemistry import decode_endpoint
from molsteer.molthinker.expressions import aggregate_objectives


class ExpertReward(MolecularReward):
    def __init__(self, program, baseline, receptor, vocabulary):
        if program.get('program_id')!='rp_'+digest({k:v for k,v in program.items() if k!='program_id'})[:24]:
            raise ValueError('Expert program digest mismatch')
        validate_expert_spec(program['expert_spec'],program['source_packet'],program['source_report'])
        super().__init__({**program,'mode':'selection'},baseline,receptor,vocabulary)
        self.spec=program
        self.evaluator=ExpertEvaluator(program['expert_spec'],program['source_packet'])
        self.uses_state_view=any(o['view']=='state' for d in self.evaluator.directions for o in d['observables'])
        self.uses_state_graph='state' in self.evaluator.mmff
        self.control_diagnostics={};self.component_gradients=None
        self.reference_policy=program['expert_spec']['mathematical_design'].get('design_audit', {}).get('graph_policy')

    def components(self,pred,state_coords=None,state_graph=None):
        if getattr(self, 'reference_policy', None) == 'suspend_on_graph_change' and self.graph(pred) != self.graph(self.p0):
            raise ValueError('reward_reference_graph_changed_guidance_suspended')
        if self.uses_state_view and state_coords is None:
            raise ValueError('Expert expressions require live state-world coordinates')
        coordinates={'prediction':pred['coords'],'state':state_coords}
        for view in {o['view'] for d in self.evaluator.directions for o in d['observables']}:
            ids=self.evaluator.packet['steering']['coordinate_snapshots'][view]['atom_ids']
            if coordinates[view].shape!=(len(ids),3):
                raise ValueError('Expert coordinate slot count changed; explicit mapping required')
        molecules={}
        for view in self.evaluator.mmff:
            if view=='state' and state_graph is None:
                raise ValueError('Live state categories are required for MMFF')
            current=pred if view=='prediction' else dict(state_graph,coords=state_coords)
            molecules[view]=decode_endpoint(current,self.vocab)
        return self.evaluator.components(coordinates,molecules=molecules)

    def evaluate(self,pred,state_coords=None,state_graph=None):
        values=self.components(pred,state_coords,state_graph)
        reward=-aggregate_objectives(self.evaluator.objectives(values),self.evaluator.strategy)
        return reward,{'components':{k:float(v.detach()) for k,v in values.items()},
                       'control_mode':self.evaluator.strategy['mode'],
                       'scalar_is_reporting_only':self.evaluator.strategy['mode']=='common_descent',
                       'chemical_applicability': self.reference_policy or 'Coordinate expressions evaluated; semantic review belongs to MolMonitor'}

    def control_gradient(self,adapter,endpoint,x,mask):
        current=self.state_graph(adapter)
        values=self.components(endpoint,adapter.world_state_coordinates(x) if self.uses_state_view else None,current)
        direction,self.component_gradients,self.control_diagnostics=control_direction(
            self.evaluator.objectives(values),x,mask,self.evaluator.strategy)
        self.potential_gradient=-direction
        if float(direction.norm())==0:
            raise ValueError('Expert control has no feasible nonzero direction: '+self.control_diagnostics['status'])
        value=-aggregate_objectives(self.evaluator.objectives(values),self.evaluator.strategy)
        return direction,value,{'components':{k:float(v.detach()) for k,v in values.items()},
                                'conflict':self.control_diagnostics}

    def proposal_failures(self,adapter,candidate,base,state_coords,base_state,delta):
        failures=[]
        current=self.state_graph(adapter)
        ok,derivatives=check_displacement(delta,self.component_gradients,self.evaluator.strategy,scalar_gradient=self.potential_gradient)
        self.control_diagnostics['post_injection_directional_derivatives']=derivatives
        if not ok: failures.append('post_injection_direction_not_feasible')
        coords=adapter.world_state_coordinates(state_coords) if self.uses_state_view else None
        basecoords=adapter.world_state_coordinates(base_state) if self.uses_state_view else None
        values=self.components(candidate,coords,current);baseline=self.components(base,basecoords,current)
        failures.extend('constraint:'+k for k in self.evaluator.constraints(values))
        if self.evaluator.strategy['mode']=='common_descent':
            failures.extend('objective_regression:'+k for k in self.evaluator.objectives(values)
                            if float(values[k])>float(baseline[k])+1e-7)
        return failures

    def state_graph(self,adapter):
        if not self.uses_state_graph:return None
        return {k:adapter.curr[k][adapter.index] for k in ('atomics','charges','bonds')}
