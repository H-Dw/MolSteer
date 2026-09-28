import numpy as np
from ..core import result,need_coords,distance_matrix

def compute(ctx):
    x=need_coords(ctx); centered=x-x.mean(0)
    eig=np.linalg.eigvalsh(centered.T@centered/len(x))
    total=float(eig.sum())
    asphericity=float(1.5*(eig*eig).sum()/total**2-0.5) if total>1e-15 else 0.0
    return result({"geometric_centroid_angstrom":x.mean(0).tolist(),"radius_of_gyration_angstrom":total**0.5,
                   "diameter_angstrom":float(distance_matrix(x).max()),"gyration_eigenvalues_angstrom2":eig.tolist(),"relative_shape_anisotropy":asphericity},
                  method="Unweighted active-atom gyration tensor; no arbitrary size-risk threshold",
                  units={"length":"angstrom","eigenvalues":"angstrom^2","anisotropy":"dimensionless"})

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("shape")
