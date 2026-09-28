"""Local geometry deviations against the current graph's MMFF94s parameters."""
from itertools import combinations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ..core import result,evidence,need_mol,need_coords,Unavailable


def compute(ctx):
    x=need_coords(ctx)
    mol=Chem.AddHs(Chem.Mol(need_mol(ctx)),addCoords=True)
    if not AllChem.MMFFHasAllMoleculeParams(mol):raise Unavailable('MMFF94s local parameters unavailable')
    props=AllChem.MMFFGetMoleculeProperties(mol,mmffVariant='MMFF94s')
    thresholds=dict(bond_relative_deviation=ctx.config['thresholds'].get('mmff_local_bond_relative_deviation',0.10),
                    angle_absolute_deviation_degrees=ctx.config['thresholds'].get('mmff_local_angle_deviation_degrees',30.),
                    calibration='Uncalibrated prioritization screens; force-field references are not experimental limits')
    bonds=[];angles=[];flags=[];missing=[]
    for i,j in combinations(range(len(x)),2):
        if not ctx.orders[i,j]:continue
        p=props.GetMMFFBondStretchParams(mol,i,j)
        if p is None:missing.append(ctx.ids([i,j]));continue
        actual=float(np.linalg.norm(x[i]-x[j]));relative=(actual-p[2])/p[2]
        row=dict(atom_ids=ctx.ids([i,j]),kind='bond_length',bond_order=float(ctx.orders[i,j]),distance_angstrom=actual,
                 reference_angstrom=float(p[2]),relative_deviation=float(relative),mmff_parameter_type=int(p[0]),
                 force_constant=float(p[1]),mmff_atom_types=[props.GetMMFFAtomType(i),props.GetMMFFAtomType(j)])
        bonds.append(row)
        if abs(relative)>thresholds['bond_relative_deviation']:
            flags.append(evidence(row['atom_ids'],message='Local bond length departs from MMFF reference',**{k:v for k,v in row.items() if k!='atom_ids'}))
    for j in range(len(x)):
        for i,k in combinations(np.flatnonzero(ctx.orders[j]>0),2):
            i,k=int(i),int(k);p=props.GetMMFFAngleBendParams(mol,i,j,k)
            if p is None:missing.append(ctx.ids([i,j,k]));continue
            u,v=x[i]-x[j],x[k]-x[j];den=np.linalg.norm(u)*np.linalg.norm(v)
            if den<=1e-12:missing.append(ctx.ids([i,j,k]));continue
            actual=float(np.degrees(np.arccos(np.clip(np.dot(u,v)/den,-1,1))))
            row=dict(atom_ids=ctx.ids([i,j,k]),kind='bond_angle',center_atom_id=int(ctx.atom_ids[j]),angle_degrees=actual,
                     reference_degrees=float(p[2]),deviation_degrees=float(actual-p[2]),mmff_parameter_type=int(p[0]),
                     force_constant=float(p[1]),mmff_atom_types=[props.GetMMFFAtomType(q) for q in (i,j,k)])
            angles.append(row)
            if abs(row['deviation_degrees'])>thresholds['angle_absolute_deviation_degrees']:
                flags.append(evidence(row['atom_ids'],message='Local angle departs from MMFF reference',**{k:v for k,v in row.items() if k!='atom_ids'}))
    return result(dict(bonds=bonds,angles=angles,missing_parameter_or_degenerate_locations=missing,outlier_count=len(flags)),
        evidence=flags,thresholds=thresholds,units={'distance':'angstrom','angle':'degree'},status='partial' if missing else 'ok',
        method='MMFF94s local reference geometry on a copy; original atom coordinates unchanged',
        notes=['Parameter availability is not parameter validation, especially for uncommon charged environments. Empirical parameter provenance is not resolved by this API.',
               'This is a parameter-dependent screening observation, not proof of illegal chemistry or chemical instability.'])


if __name__=='__main__':
    from ..cli import metric_main
    metric_main('mmff_local_geometry')
