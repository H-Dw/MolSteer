import numpy as np
from ..core import result, evidence, Unavailable
from ..io import active_arrays


def compute(ctx):
    if ctx.view != 'prediction':
        return result(status='not_applicable', notes=['Cross-source decoding comparison is recorded once in prediction view.'])
    if ctx.sdf_mol is None:
        raise Unavailable("No decoded SDF molecule")
    if not ctx.sdf_mapping_valid:
        return result({"mapping_verified": False}, evidence=[evidence(severity="high", message="SDF atom order/element/coordinate correspondence is unverified")],
                      method="Direct slot-to-SDF correspondence; no speculative graph mapping")
    _, a = active_arrays(ctx.bundles["prediction"], ctx.config)
    raw_orders = np.array(ctx.config["bond_orders"])[a["bonds"].argmax(-1)]
    raw_charges = [ctx.config["charge_tokens"][i] for i in a["charges"].argmax(-1)]
    sdf_orders = np.zeros_like(raw_orders)
    for b in ctx.sdf_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        sdf_orders[i,j] = sdf_orders[j,i] = b.GetBondTypeAsDouble()
    differences = []
    for i, atom in enumerate(ctx.sdf_mol.GetAtoms()):
        if raw_charges[i] != atom.GetFormalCharge():
            differences.append(evidence(ctx.ids([i]), message="Decoded SDF formal charge differs from head argmax", head_charge=raw_charges[i], sdf_charge=atom.GetFormalCharge()))
    for i, j in zip(*np.triu_indices(len(raw_charges), 1)):
        if raw_orders[i,j] != sdf_orders[i,j]:
            differences.append(evidence(ctx.ids([i,j]), message="Decoded SDF bond order differs from head argmax", head_order=float(raw_orders[i,j]), sdf_order=float(sdf_orders[i,j])))
    return result({"mapping_verified": True, "difference_count": len(differences)}, evidence=differences,
                  method="Index, element and coordinate verification followed by graph comparison",
                  notes=["Differences may originate from decoding/valence repair or representation normalization; this check does not assign causality."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("decode_consistency")
