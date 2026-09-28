from itertools import combinations
import numpy as np
from rdkit.Chem import rdDistGeom
from ..core import result, evidence, need_coords

def compute(ctx):
    x=need_coords(ctx)
    bounds=rdDistGeom.GetMoleculeBoundsMatrix(ctx.mol) if ctx.mol is not None else None
    tol=ctx.config["thresholds"]["angle_endpoint_relative_tolerance"]
    rows, flags=[],[]
    for j in range(len(x)):
        for i,k in combinations(np.flatnonzero(ctx.orders[j]>0),2):
            u,v=x[i]-x[j],x[k]-x[j]
            denom=np.linalg.norm(u)*np.linalg.norm(v)
            angle=float(np.degrees(np.arccos(np.clip(np.dot(u,v)/denom,-1,1)))) if denom>1e-12 else None
            d=float(np.linalg.norm(x[i]-x[k]))
            low,high=(float(bounds[max(i,k),min(i,k)])*(1-tol),float(bounds[min(i,k),max(i,k)])*(1+tol)) if bounds is not None else (None,None)
            row=dict(atom_ids=ctx.ids([i,j,k]),angle_degrees=angle,endpoint_distance_angstrom=d,endpoint_lower_angstrom=low,endpoint_upper_angstrom=high)
            rows.append(row)
            if angle is None or (low is not None and (d<low or d>high)):
                flags.append(evidence(ctx.ids([i,j,k]),message="Degenerate angle or 1-3 endpoint distance outside topology bounds",**{k:v for k,v in row.items() if k!="atom_ids"}))
    return result({"angle_count":len(rows),"evaluated_angle_count":len(rows) if bounds is not None else 0,"outlier_count":len(flags) if bounds is not None else None,"angles":rows},evidence=flags,
                  units={"angle":"degree","endpoint_distance":"angstrom"},method="Angles plus RDKit 1-3 distance-geometry screen (ring aware)",
                  status="ok" if bounds is not None else "partial", thresholds={"relative_expansion":tol},
                  notes=["Endpoint violations couple bond length and angle; they do not isolate angular strain."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("bond_angles")
