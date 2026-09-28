"""Differentiable coordinate energies; categorical decisions remain frozen."""
import torch


def observable(term, coords, slot_index):
    q = [coords[slot_index[a]] for a in term['atom_ids']]
    if term['family'] == 'flat_bottom_distance':
        d = torch.linalg.vector_norm(q[0]-q[1])
        if d.detach().item() < 1e-10:
            raise ValueError('Coincident pair has undefined radial direction')
        return d
    if term['family'] == 'minimum_distance':
        ref = coords.new_tensor(term['reference_coords'])
        d = torch.linalg.vector_norm(q[0]-ref)
        if d.detach().item() < 1e-10:
            raise ValueError('Coincident receptor pair has undefined radial direction')
        return d
    if term['family'] == 'flat_bottom_angle':
        u, v = q[0]-q[1], q[2]-q[1]
        den = torch.linalg.vector_norm(u)*torch.linalg.vector_norm(v)
        if den.detach().item() < 1e-10:
            raise ValueError('Degenerate angle')
        cos = torch.dot(u,v)/den
        if abs(cos.detach().item()) > 1-1e-10:
            raise ValueError('Collinear angle needs another observable')
        return torch.rad2deg(torch.acos(cos))
    raise ValueError('Reward family has no executable backend')


def term_energy(term, coords, slot_index):
    v = observable(term, coords, slot_index)
    e = torch.relu((term['lower']-v)/term['scale'])**2
    if term['upper'] is not None:
        e = e + torch.relu((v-term['upper'])/term['scale'])**2
    return .5*term['weight']*e


def energy(terms, coords, atom_ids):
    index = {a:i for i,a in enumerate(atom_ids)}
    return sum((term_energy(t, coords, index) for t in terms), coords.sum()*0.)
