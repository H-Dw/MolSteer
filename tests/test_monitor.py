import copy
import unittest
import torch
from molsteer.molmonitor.reference import ReferenceTrajectory
from molsteer.molmonitor.controller import AdaptiveController,MonitorPolicy
from molsteer.molmonitor.runtime import candidate_failures,guidance_capacity,quality_sentinel
from molsteer.molthinker.feedback import apply_revision_response


def frame(t,v,**kwargs):
    result=dict(time=t,valid=True,finite=True,graph_id='g',features={'distance:0:1':dict(value=v,scale=1.,unit='angstrom',family='distance',atom_ids=[0,1])},geometry_rms_z=.1,geometry_max_abs_z=.2)
    result.update(kwargs);return result


class MonitorTests(unittest.TestCase):
    def test_zero_gradient_is_not_exhausted_budget(self):
        movable,exhausted=guidance_capacity(torch.zeros(2),torch.ones(2),torch.ones(2,dtype=torch.bool))
        self.assertFalse(movable.any());self.assertFalse(exhausted)
        _,exhausted=guidance_capacity(torch.ones(2),torch.zeros(2),torch.ones(2,dtype=torch.bool))
        self.assertTrue(exhausted)

    def test_new_graph_uses_own_strain_and_missing_checks_require_review(self):
        current=dict(time=.85,graph_id='new',strain=dict(value=42.,converged=True),affinity={'pkd':6.},atoms=[{'element':'C'}]*20)
        control=dict(graph_id='native',strain=dict(value=20.,converged=True),affinity={'pkd':5.5})
        observed=quality_sentinel(current,control,MonitorPolicy(),'pkd')
        self.assertFalse(observed['comparable']);self.assertFalse(observed['reward_quality_conflict'])
        self.assertTrue(observed['changed_graph_quality_review'])
        ctl=AdaptiveController(MonitorPolicy())
        for step in range(3):decision=ctl.route(step,.85,[],None,sentinel=observed)
        self.assertEqual(decision['destination'],'MolThinker')
        current['strain']['value']=15.
        self.assertFalse(quality_sentinel(current,control,MonitorPolicy(),'pkd')['changed_graph_quality_review'])
        del current['strain']
        self.assertTrue(quality_sentinel(current,control,MonitorPolicy(),'pkd')['changed_graph_quality_review'])

    def test_native_trajectory_is_not_a_temporal_outlier(self):
        ref=ReferenceTrajectory(dict(frames=[frame(.5+i*.01,1+i*.01) for i in range(6)]))
        for a,b in zip(ref.frames,ref.frames[1:]):
            self.assertEqual(ref.temporal_evidence(a,b)['events'],[])
        self.assertTrue(ref.dense)

    def test_zero_mad_uses_physical_floor_and_localizes(self):
        ref=ReferenceTrajectory(dict(frames=[frame(.5,1),frame(.51,1),frame(.52,1)]))
        evidence=ref.temporal_evidence(frame(.5,1),frame(.51,2))
        self.assertEqual(evidence['events'][0]['atom_ids'],[0,1])
        self.assertGreater(evidence['max_ratio'],1)
        self.assertGreater(evidence['events'][0]['limit'],0)

    def test_fast_but_nonworsening_change_is_not_broken(self):
        event=dict(max_ratio=100)
        initial=frame(.5,1);base=frame(.51,1);candidate=frame(.51,2)
        self.assertEqual(candidate_failures(candidate,base,initial,event,MonitorPolicy()),[])
        candidate['geometry_rms_z']=.2
        self.assertIn('temporal_spike_with_geometry_worsening',candidate_failures(candidate,base,initial,event,MonitorPolicy()))
        candidate['graph_id']='new_legal_graph'
        self.assertEqual(candidate_failures(candidate,base,initial,event,MonitorPolicy()),[])

    def test_sparse_reference_is_not_fine_calibration(self):
        ref=ReferenceTrajectory(dict(frames=[frame(.5,1),frame(.75,2),frame(1.,3)]))
        self.assertFalse(ref.dense)
        with self.assertRaises(ValueError):ref.frame(.51)
        self.assertIn('Two coarse slopes',ref.coarse_summary()['warning'])

    def test_largest_feasible_effective_step_not_largest_nominal_eta(self):
        ctl=AdaptiveController(MonitorPolicy())
        trials=[dict(eta=1,effective_l2=.1,gain=.01,failures=[]),dict(eta=2,effective_l2=.2,gain=.02,failures=[]),
            dict(eta=10,effective_l2=.2,gain=.02,failures=[]),dict(eta=100,effective_l2=.3,gain=10,failures=['severe_clash'])]
        self.assertEqual(ctl.choose(trials)['eta'],2)

    def test_persistent_conflict_routes_to_thinker_but_budget_does_not(self):
        ctl=AdaptiveController(MonitorPolicy(persistence=3))
        bad=[dict(eta=1,effective_l2=.1,gain=1,failures=['new_or_worsened_severe_geometry'])]
        for step in range(3):decision=ctl.route(step,.8,bad,None)
        self.assertEqual(decision['destination'],'MolThinker')
        ctl=AdaptiveController(MonitorPolicy())
        for step in range(10):decision=ctl.route(step,.8,[],None,budget_exhausted=True)
        self.assertEqual(decision['action'],'stop_guidance');self.assertEqual(decision['destination'],'MolExecutor')

    def test_controller_restart_preserves_next_search_and_routing(self):
        ctl=AdaptiveController(MonitorPolicy());ctl.route(5,.7,[],None)
        copyctl=AdaptiveController(MonitorPolicy(),copy.deepcopy(ctl.state))
        self.assertEqual(ctl.strengths(175.),copyctl.strengths(175.))
        self.assertEqual(ctl.route(6,.8,[],None),copyctl.route(6,.8,[],None))

    def test_revision_binding_preserves_path_and_rejects_changed_guard(self):
        pending=dict(request_id='r',packet_id='p',identity='i',guard_contract={'severe_overlap_angstrom':.4})
        cp=dict(guidance_state=dict(program_id='old',pending_request=pending,path_used=[.4],monitor=dict(controller={'streak':3})))
        response=dict(request_id='r',parent_program_id='old',program=dict(program_id='new',packet_id='p',identity='i',severe_overlap_angstrom=.4),rationale='evidence',validation_plan='paired test')
        revised,program=apply_revision_response(cp,response)
        self.assertEqual(revised['guidance_state']['path_used'],[.4]);self.assertEqual(cp['guidance_state']['program_id'],'old')
        stopped=copy.deepcopy(response);stopped.update(resolution='stop_guidance')
        stopped['program']['program_id']='old'
        stopped_cp,_=apply_revision_response(cp,stopped)
        self.assertTrue(stopped_cp['guidance_state']['monitor']['stopped'])
        self.assertEqual(stopped_cp['guidance_state']['path_used'],[.4])
        response['program']['severe_overlap_angstrom']=10
        with self.assertRaises(ValueError):apply_revision_response(cp,response)


if __name__=='__main__':unittest.main()
