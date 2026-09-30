"""Declarative reward evaluator; no dynamic Python or expression evaluation."""
import math
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from molreader.io import decode


def angle(x, ids):
    a, b, c = ids
    u, v = x[a]-x[b], x[c]-x[b]
    if min(float(u.detach().norm()), float(v.detach().norm())) < 1e-7:
        raise ValueError('Degenerate angle')
    cosine = (u*v).sum() / (u.norm()*v.norm()).clamp(min=1e-12)
    return torch.acos(cosine.clamp(-1+1e-7, 1-1e-7))*180/math.pi


def interval(value, lower, upper, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid normalization')
    residual = torch.relu((lower-value)/scale).square()
    if upper is not None:
        residual = residual + torch.relu((value-upper)/scale).square()
    return 0.5*residual


def balance(values, weights, tau, rho):
    if tau <= 0 or rho < 0 or not values:
        raise ValueError('Invalid balance configuration')
    z = torch.stack(values)*torch.as_tensor(weights, device=values[0].device, dtype=values[0].dtype)
    return tau*(torch.logsumexp(z/tau, 0)-math.log(len(values)))+rho*z.sum()


class MolecularReward:
    """Graph-conditioned, coordinate-differentiable molecular objectives.

    Discrete graph selection is deliberately outside autograd. Frozen model
    preference contributes to candidate acceptance, not a fictitious derivative.
    Missing graph references raise rather than shrinking the objective set.
    """
    def __init__(self, program, baseline, receptor, vocabulary):
        if program.get('mode') not in ('selection','creativity'):
            raise ValueError('Unsupported composition')
        positive=['tau','bond_tolerance_fraction','angle_tolerance_degrees','clash_scale_angstrom',
            'centroid_scale_angstrom','movement_scale_angstrom','max_endpoint_displacement_angstrom','graph_epsilon']
        if any(not math.isfinite(program[k]) or program[k]<=0 for k in positive):
            raise ValueError('Nonpositive/nonfinite reward scale')
        if len(program['weights'])!=len(program['active_objectives']) or any(not math.isfinite(w) or w<0 for w in program['weights']):
            raise ValueError('Invalid objective weights')
        if not program['region_atom_ids'] or not set(program['region_atom_ids'])<=set(range(len(baseline['coords']))):
            raise ValueError('Unknown or empty atom region')
        self.spec = program
        self.x0 = baseline['coords'].detach().clone()
        self.p0 = {k: baseline[k].detach().clone() for k in ['atomics','charges','bonds']}
        self.receptor = receptor
        self.vocab = vocabulary
        self.cache = {}
        self.region = set(program['region_atom_ids'])
        self.reference_graph = self.graph(baseline)

    def graph(self, pred):
        return tuple(tuple(pred[k].detach().argmax(-1).reshape(-1).cpu().tolist())
                     for k in ['atomics','charges','bonds'])

    def chemistry(self, pred):
        if not torch.isfinite(pred['coords']).all():raise ValueError('Nonfinite endpoint coordinates')
        classes=pred['bonds'].detach().argmax(-1)
        if not torch.equal(classes,classes.T) or classes.diagonal().any():
            raise ValueError('Asymmetric or self-bond graph')
        signature = self.graph(pred)
        if signature in self.cache:
            return self.cache[signature]
        n = len(pred['coords'])
        atomic, charge, bond = signature
        atoms = [self.vocab['atomic_tokens'][i] for i in atomic]
        charges = [self.vocab['charge_tokens'][i] for i in charge]
        if any(a in ('PAD',None) for a in atoms) or any(c in ('PAD',None) for c in charges) or len(atoms) != n:
            raise ValueError('Unresolved categorical identity')
        orders = torch.tensor([self.vocab['bond_orders'][i] for i in bond]).reshape(n,n).numpy()
        raw, mol, error = decode(pred['coords'].detach().cpu().numpy(), atoms, charges, orders)
        if mol is None:
            raise ValueError('Chemical validity: '+str(error))
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError('Disconnected chemical graph')
        props = AllChem.MMFFGetMoleculeProperties(Chem.AddHs(mol, addCoords=True))
        geometries = []
        expected_geometries = sum({b.GetBeginAtomIdx(),b.GetEndAtomIdx()}<=self.region for b in mol.GetBonds())
        for center in self.region:
            degree=sum(a.GetIdx() in self.region for a in mol.GetAtomWithIdx(center).GetNeighbors())
            expected_geometries += degree*(degree-1)//2
        if props is not None:
            mh = Chem.AddHs(mol, addCoords=True)
            for b in mol.GetBonds():
                i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if {i,j} <= self.region:
                    p = props.GetMMFFBondStretchParams(mh,i,j)
                    if p:
                        geometries.append(('bond',[i,j],p[2],p[2]*self.spec['bond_tolerance_fraction']))
            for center in self.region:
                neigh = sorted(a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors() if a.GetIdx() in self.region)
                for ii,i in enumerate(neigh):
                    for j in neigh[ii+1:]:
                        p = props.GetMMFFAngleBendParams(mh,i,center,j)
                        if p:
                            geometries.append(('angle',[i,center,j],p[2],self.spec['angle_tolerance_degrees']))
        distances = Chem.GetDistanceMatrix(mol)
        pairs = [(i,j) for i in range(n) for j in range(i+1,n) if distances[i,j] > 2]
        periodic = Chem.GetPeriodicTable()
        radii = [periodic.GetRvdw(periodic.GetAtomicNumber(s)) for s in atoms]
        result = dict(geometries=geometries, pairs=pairs, radii=radii, smiles=Chem.MolToSmiles(mol), signature=signature,
                      missing_geometry_count=expected_geometries-len(geometries))
        self.cache[signature] = result
        return result

    def overlaps(self, x, chemistry):
        r = x.new_tensor(chemistry['radii'])
        pxyz = x.new_tensor([a['coords'] for a in self.receptor])
        pr = x.new_tensor([a['vdw_radius'] for a in self.receptor])
        if len(pxyz) == 0:
            raise ValueError('Missing receptor; cannot evaluate common clash objective')
        # Positive signed penetration in Angstrom; matrices preserve pair identity.
        protein = self.spec['protein_vdw_ratio']*(r[:,None]+pr[None,:])-torch.cdist(x,pxyz)
        intramol = x.new_full((len(x),len(x)), -1e6)
        if chemistry['pairs']:
            ids = torch.tensor(chemistry['pairs'],device=x.device)
            i,j = ids.T
            ov = self.spec['intra_vdw_ratio']*(r[i]+r[j])-(x[i]-x[j]).norm(dim=-1)
            intramol = intramol.index_put((i,j),ov)
        return protein, intramol

    def graph_cost(self, pred):
        costs=[]
        eps=self.spec['graph_epsilon']
        for k in ['atomics','charges']:
            for i in sorted(self.region):
                p=self.p0[k][i]
                index=int(pred[k][i].detach().argmax())
                costs.append(torch.log((p.max()+eps)/(p[index]+eps)))
        for i in sorted(self.region):
            for j in sorted(self.region):
                if i<j:
                    p=self.p0['bonds'][i,j]
                    index=int(pred['bonds'][i,j].detach().argmax())
                    costs.append(torch.log((p.max()+eps)/(p[index]+eps)))
        return torch.stack(costs).mean() if costs else pred['coords'].sum()*0

    def evaluate(self, pred):
        x=pred['coords']
        chem=self.chemistry(pred)
        protein,intra=self.overlaps(x,chem)
        if self.spec['mode']=='selection':
            components={}
            for term in self.spec['terms']:
                ids=term['atom_ids']
                if term['family']=='flat_bottom_angle':value=angle(x,ids)
                elif term['family']=='minimum_distance':value=(x[ids[0]]-x.new_tensor(term['reference_coords'])).norm()
                elif term['family']=='flat_bottom_distance':value=(x[ids[0]]-x[ids[1]]).norm()
                else:raise ValueError('Unsupported reward primitive: '+term['family'])
                components[term['term_id']]=term['weight']*interval(value,term['lower'],term['upper'],term['scale'])
            reward=-sum(components.values(),x.sum()*0)
            cg=x.sum()*0
        else:
            if not chem['geometries']:
                raise ValueError('Missing graph-conditioned geometry references')
            residual=[]
            for kind,ids,ref,tol in chem['geometries']:
                value=angle(x,ids) if kind=='angle' else (x[ids[0]]-x[ids[1]]).norm()
                residual.append(torch.relu((value-ref).abs()-tol)/tol)
            geometry=torch.sqrt(1+torch.stack(residual).square().sum())-1
            clash=torch.maximum(protein.max(),intra.max()).clamp(min=0).square()/self.spec['clash_scale_angstrom']**2
            pocket=(x.mean(0)-self.x0.mean(0)).square().sum()/self.spec['centroid_scale_angstrom']**2
            movement=(x-self.x0).square().sum(-1).mean()/self.spec['movement_scale_angstrom']**2
            components=dict(geometry=geometry,clash=clash,pocket=pocket,movement=movement)
            if list(components)!=self.spec['active_objectives']:
                raise ValueError('Common objective set changed')
            cg=self.graph_cost(pred)
            reward=-balance(list(components.values()),self.spec['weights'],self.spec['tau'],self.spec['rho'])-self.spec['lambda_graph']*cg
        return reward,dict(components={k:float(v.detach()) for k,v in components.items()},
            graph_cost=float(cg.detach()), graph_changed=chem['signature']!=self.reference_graph,
            smiles=chem['smiles'], max_protein_overlap=float(protein.max().detach()),
            max_intra_overlap=float(intra.max().detach()), graph_reference_count=len(chem['geometries']))

    def feasible(self, candidate, reference):
        """Noncompensable constraints on the proposed guidance endpoint."""
        reasons=[]
        try:
            cc=self.chemistry(candidate)
            bc=self.chemistry(reference)
            cp,ci=self.overlaps(candidate['coords'],cc)
            bp,bi=self.overlaps(reference['coords'],bc)
            tol=self.spec['severe_overlap_angstrom']
            # A new severe pair or worsening an existing severe overlap is rejected.
            if ((cp>tol)&(cp>bp+1e-5)).any() or ((ci>tol)&(ci>bi+1e-5)).any():
                reasons.append('new_or_worsened_severe_clash')
            if (candidate['coords']-self.x0).norm(dim=-1).max()>self.spec['max_endpoint_displacement_angstrom']:
                reasons.append('endpoint_trust_region')
        except ValueError as exc:
            reasons.append(str(exc))
        return reasons


class AgentMixedReward(MolecularReward):
    """Apply checked Agent terms to their declared state or prediction view."""

    def __init__(self, program, baseline, receptor, vocabulary):
        if program.get('mode') not in ('agent_selection','agent_design') or program.get('evaluator')!='agent_mixed':
            raise ValueError('Expected a compiled agent coordinate program')
        super().__init__({**program,'mode':'selection'},baseline,receptor,vocabulary)
        self.spec=program
        self.uses_state_view=any(term['view']=='state' for term in program['terms'])
        if not program['terms'] or any(term['view'] not in ('state','prediction') for term in program['terms']):
            raise ValueError('Agent program requires supported, view-bound terms')

    def evaluate(self, pred, state_coords=None):
        chemistry=self.chemistry(pred)
        protein,intra=self.overlaps(pred['coords'],chemistry)
        if self.uses_state_view and state_coords is None:
            raise ValueError('Live state-world coordinates are required')
        if state_coords is not None and state_coords.shape!=pred['coords'].shape:
            raise ValueError('State/prediction coordinate shape mismatch')
        components={}
        for term in self.spec['terms']:
            coords=pred['coords'] if term['view']=='prediction' else state_coords
            ids=term['atom_ids']
            if term['family']=='flat_bottom_angle':
                value=angle(coords,ids)
            elif term['family']=='flat_bottom_distance':
                value=(coords[ids[0]]-coords[ids[1]]).norm()
            elif term['family']=='minimum_distance':
                value=(coords[ids[0]]-coords.new_tensor(term['reference_coords'])).norm()
            else:
                raise ValueError('Unsupported agent reward primitive: '+term['family'])
            components[term['term_id']]=term['weight']*interval(
                value,term['lower'],term['upper'],term['scale'])
        if 'design' in self.spec:
            from molsteer.molthinker.composition import objective_value
            reward=-objective_value(self.spec['design']['objective_tree'],components)
        else:
            reward=-sum(components.values(),pred['coords'].sum()*0)
        return reward,dict(components={k:float(v.detach()) for k,v in components.items()},
            graph_changed=chemistry['signature']!=self.reference_graph,
            smiles=chemistry['smiles'],max_protein_overlap=float(protein.max().detach()),
            max_intra_overlap=float(intra.max().detach()))


def evaluate_with_state(reward, adapter, endpoint, state_coordinates):
    if getattr(reward,'uses_state_view',False):
        kwargs={'state_graph':reward.state_graph(adapter)} if getattr(reward,'uses_state_graph',False) else {}
        return reward.evaluate(endpoint,
            state_coords=adapter.world_state_coordinates(state_coordinates),**kwargs)
    return reward.evaluate(endpoint)


def _make_reward(program,baseline,receptor,vocabulary,control=None):
    if program.get('evaluator')=='agent_expert':
        from .expert_reward import ExpertReward
        return ExpertReward(program,baseline,receptor,vocabulary)
    if program.get('evaluator')=='agent_mixed':
        return AgentMixedReward(program,baseline,receptor,vocabulary)
    if program.get('evaluator')=='augmented_lagrangian':
        from .augmented_lagrangian_reward import AugmentedLagrangianReward
        if control is None:raise ValueError('Augmented-Lagrangian guidance requires a matched control trajectory')
        return AugmentedLagrangianReward(program,baseline,receptor,vocabulary,control)
    if program.get('evaluator')=='local_first':
        from .local_first_reward import LocalFirstReward
        if control is None:raise ValueError('Local-first guidance requires a matched control trajectory')
        return LocalFirstReward(program,baseline,receptor,vocabulary,control)
    if program.get('evaluator')=='outcome_aware':
        from .outcome_reward import OutcomeAwareReward
        if control is None:raise ValueError('Outcome guidance requires a matched control trajectory')
        return OutcomeAwareReward(program,baseline,receptor,vocabulary,control)
    if program.get('evaluator')=='affinity_structure':
        from .affinity import AffinityStructureReward
        if control is None:raise ValueError('Affinity guidance requires a matched control trajectory')
        return AffinityStructureReward(program,baseline,receptor,vocabulary,control)
    return MolecularReward(program,baseline,receptor,vocabulary)


def make_reward(program,baseline,receptor,vocabulary,control=None):
    from .weighted_reward import WeightedReward
    reward=_make_reward(program,baseline,receptor,vocabulary,control)
    return WeightedReward(reward,program['reward_weight']) if 'reward_weight' in program else reward
