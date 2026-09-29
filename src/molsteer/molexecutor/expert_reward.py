"""Live view-faithful execution of version-2 expert controls."""
import torch
from molsteer.common import digest
from molsteer.agents.expert_contracts import validate_expert_spec
from .program import MolecularReward
from .expert_control import ExpertEvaluator, control_direction, check_displacement
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
        self.control_diagnostics={};self.component_gradients=None
        self.check_graph(baseline,'prediction')

    def check_graph(self,pred,view):
        packet=self.evaluator.packet
        atomic=pred['atomics'].detach().argmax(-1).cpu().tolist()
        charges=pred['charges'].detach().argmax(-1).cpu().tolist()
        bond=pred['bonds'].detach().argmax(-1).cpu().tolist()
        signature=digest(dict(atom_ids=packet['representations'][view]['original_atom_ids'],
                              atoms=[self.vocab['atomic_tokens'][i] for i in atomic],
                              formal_charges=[self.vocab['charge_tokens'][i] for i in charges],
                              orders=[[float(self.vocab['bond_orders'][i]) for i in row] for row in bond]))
        if signature!=packet['steering']['graph_signatures'][view]:
            raise ValueError('Chemical graph changed; fresh expert state binding is required')

    def components(self,pred,state_coords=None):
        self.check_graph(pred,'prediction')
        if self.uses_state_view and state_coords is None:
            raise ValueError('Expert expressions require live state-world coordinates')
        return self.evaluator.components({'prediction':pred['coords'],'state':state_coords})

    def evaluate(self,pred,state_coords=None):
        values=self.components(pred,state_coords)
        reward=-aggregate_objectives(self.evaluator.objectives(values),self.evaluator.strategy)
        return reward,{'components':{k:float(v.detach()) for k,v in values.items()},
                       'control_mode':self.evaluator.strategy['mode'],
                       'scalar_is_reporting_only':self.evaluator.strategy['mode']=='common_descent'}

    def control_gradient(self,adapter,endpoint,x,mask):
        if self.uses_state_view:
            current={k:v[adapter.index] for k,v in adapter.curr.items() if k in ('atomics','charges','bonds')}
            self.check_graph(current,'state')
        values=self.components(endpoint,adapter.world_state_coordinates(x) if self.uses_state_view else None)
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
        if self.uses_state_view:
            current={k:v[adapter.index] for k,v in adapter.curr.items() if k in ('atomics','charges','bonds')}
            self.check_graph(current,'state')
        ok,derivatives=check_displacement(delta,self.component_gradients,self.evaluator.strategy,scalar_gradient=self.potential_gradient)
        self.control_diagnostics['post_injection_directional_derivatives']=derivatives
        if not ok: failures.append('post_injection_direction_not_feasible')
        coords=adapter.world_state_coordinates(state_coords) if self.uses_state_view else None
        basecoords=adapter.world_state_coordinates(base_state) if self.uses_state_view else None
        values=self.components(candidate,coords);baseline=self.components(base,basecoords)
        failures.extend('constraint:'+k for k in self.evaluator.constraints(values))
        if self.evaluator.strategy['mode']=='common_descent':
            failures.extend('objective_regression:'+k for k in self.evaluator.objectives(values)
                            if float(values[k])>float(baseline[k])+1e-7)
        return failures
