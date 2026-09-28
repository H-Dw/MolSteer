import numpy as np
from ..core import result, need_coords

def compute(ctx):
    x=need_coords(ctx)
    rows=[]
    for j,k in zip(*np.where(np.triu(ctx.orders>0,1))):
        for i in np.flatnonzero(ctx.orders[j]>0):
            for l in np.flatnonzero(ctx.orders[k]>0):
                if len({int(i),int(j),int(k),int(l)})<4:
                    continue
                b0,b1,b2=-(x[j]-x[i]),x[k]-x[j],x[l]-x[k]
                norm=np.linalg.norm(b1)
                angle=None
                if norm>1e-12:
                    b1=b1/norm
                    v,w=b0-np.dot(b0,b1)*b1,b2-np.dot(b2,b1)*b1
                    if np.linalg.norm(v)*np.linalg.norm(w)>1e-12:
                        angle=float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))
                rows.append(dict(atom_ids=ctx.ids([i,j,k,l]),dihedral_degrees=angle))
    return result({"torsion_count":len(rows),"torsions":rows},units={"torsion":"degree"},method="Signed four-atom dihedral",
                  notes=["A torsion angle has no universal valid interval; no unsupported torsion-outlier labels are assigned."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("torsions")
