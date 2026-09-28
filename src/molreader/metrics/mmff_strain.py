import numpy as np
from ._shared import mmff
from ..core import result,evidence

def compute(ctx):
    data=mmff(ctx);flags=[]
    threshold=ctx.config["thresholds"]["strain_kcal_mol"]
    converged=data["minimization_status"]==0 and data["hydrogen_relaxation_status"]==0
    if converged and data["strain_proxy_kcal_mol"]>threshold:
        displacement=np.asarray(data["heavy_atom_displacement_angstrom"])
        ids=ctx.ids(np.argsort(displacement)[-min(5,len(displacement)):][::-1])
        flags.append(evidence(ids,message="Large MMFF local relaxation energy drop",strain_proxy_kcal_mol=data["strain_proxy_kcal_mol"],
                              localization="Atoms with largest relaxation displacement; not an energy decomposition"))
    return result(data,evidence=flags,status="ok" if converged else "partial",units={"energy":"kcal/mol","displacement":"angstrom"},
                  method="E(original heavy-atom pose with relaxed H) minus E(local unconstrained MMFF94s minimum)",
                  thresholds={"screening_strain_kcal_mol":threshold},
                  notes=["Local relaxation proxy, not global-minimum strain or binding free energy. Nonconverged estimates cannot trigger a strain finding.",
                         "Minimization uses a molecule copy and never changes the submitted pose."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("mmff_strain")
