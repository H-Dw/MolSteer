import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ..core import Unavailable, result, evidence, need_coords, need_mol, need_protein, distance_matrix, summarize


def confidence(ctx, key):
    if key not in ctx.probs:
        return result(status="not_applicable", notes=["SDF view has no categorical distribution; head distribution is recorded in prediction view."])
    p = ctx.probs[key]
    if key == "bonds":
        i, j = np.triu_indices(len(ctx.atom_ids), 1)
        p = p[i, j]
        locations = [ctx.ids([a, b]) for a, b in zip(i, j)]
    else:
        locations = [[int(a)] for a in ctx.atom_ids]
    if not len(p):
        return result(status="not_applicable", notes=["No active category rows"])
    tol = ctx.config["probability_tolerance"]
    if not np.isfinite(p).all() or np.any(p < -tol) or not np.allclose(p.sum(-1), 1, atol=tol, rtol=0):
        raise Unavailable("Category rows are not valid probability distributions")
    p = np.clip(p, 0, 1)
    h = -(p * np.log(np.maximum(p, 1e-300))).sum(-1) / np.log(p.shape[-1])
    sorted_p = np.sort(p, axis=-1)
    maximum = sorted_p[:, -1]
    threshold = ctx.config["thresholds"]["low_confidence"]
    entries = [evidence(ids, message="Diffuse model category distribution", confidence=float(maximum[k]),
                        normalized_entropy=float(h[k]), margin=float(sorted_p[k,-1]-sorted_p[k,-2]))
               for k, ids in enumerate(locations) if maximum[k] < threshold]
    values = dict(normalized_entropy=summarize(h), max_probability=summarize(maximum), low_confidence_count=len(entries),
                  row_count=len(p), one_hot_fraction=float((maximum > 1-1e-6).mean()))
    if key == "bonds":
        # All-pair averages are dominated by absent edges; retain predicted-edge statistics.
        edge = p.argmax(-1) != 0
        values["predicted_edge_entropy"] = summarize(h[edge])
        values["predicted_edge_count"] = int(edge.sum())
    return result(values, evidence=entries, units={"entropy": "normalized [0,1]", "probability": "[0,1]"},
                  method="Shannon entropy / log(K), maximum probability and top-two margin",
                  thresholds={"maximum_probability_below": threshold},
                  notes=["A sampled one-hot state is not calibrated certainty. Model probabilities are not defect probabilities."])


def receptor_distances(ctx):
    protein = need_protein(ctx)
    xyz = np.array([a["coords"] for a in protein])
    return protein, distance_matrix(ctx.coords, xyz)


def receptor_location(a):
    return dict(receptor_serial=a["serial"], residue_id=a["residue_id"], receptor_atom=a["atom_name"])


def mmff(ctx):
    if "mmff" in ctx.cache:
        return ctx.cache["mmff"]
    mol = Chem.AddHs(Chem.Mol(need_mol(ctx)), addCoords=True)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise Unavailable("MMFF strain requires a single connected molecule")
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise Unavailable("MMFF94s parameters missing for one or more atoms")
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    if ff is None:
        raise Unavailable("MMFF force field could not be constructed")
    heavy_n = ctx.mol.GetNumAtoms()
    for i in range(heavy_n):
        ff.AddFixedPoint(i)
    h_status = ff.Minimize(maxIts=300)
    initial = float(ff.CalcEnergy())
    relaxed = Chem.Mol(mol)
    ff2 = AllChem.MMFFGetMoleculeForceField(relaxed, AllChem.MMFFGetMoleculeProperties(relaxed, mmffVariant="MMFF94s"))
    iterations = int(ctx.config["thresholds"]["mmff_max_iterations"])
    convergence = ff2.Minimize(maxIts=iterations)
    final = float(ff2.CalcEnergy())
    displacement = np.linalg.norm(relaxed.GetConformer().GetPositions()[:heavy_n] - ctx.coords, axis=-1)
    if not np.isfinite([initial, final]).all():
        raise Unavailable("Nonfinite MMFF energy")
    out = dict(energy_kcal_mol=initial, relaxed_energy_kcal_mol=final, strain_proxy_kcal_mol=initial-final,
               hydrogen_relaxation_status=int(h_status), minimization_status=int(convergence),
               max_iterations=iterations, heavy_atom_displacement_angstrom=displacement.tolist())
    ctx.cache["mmff"] = out
    return out
