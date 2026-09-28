"""First-order autograd bridge to MMFF94s with optimized hydrogens.

Hydrogens are minimized with heavy atoms fixed. The envelope derivative uses
the heavy-atom energy gradient at that hydrogen stationary point. A graph's
local reference minimum is frozen, not differentiated or silently updated.
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from .chemistry import signature


class MMFFStrain:
    def __init__(self):
        self.references={}
        self.cache={}

    def evaluate(self,mol,coordinates):
        xyz=np.asarray(coordinates,dtype=np.float64)
        key=signature(mol);ck=(key,xyz.tobytes())
        if ck in self.cache:return self.cache[ck]
        m=Chem.Mol(mol);n=m.GetNumAtoms()
        for i,row in enumerate(xyz):m.GetConformer().SetAtomPosition(i,row)
        h=Chem.AddHs(m,addCoords=True)
        if not AllChem.MMFFHasAllMoleculeParams(h):raise ValueError('MMFF parameters unavailable')
        props=AllChem.MMFFGetMoleculeProperties(h,mmffVariant='MMFF94s')
        fixed=AllChem.MMFFGetMoleculeForceField(h,props)
        for i in range(n):fixed.AddFixedPoint(i)
        status=fixed.Minimize(maxIts=1000,forceTol=1e-5,energyTol=1e-8)
        if status:raise ValueError('Hydrogen minimization did not converge')
        ff=AllChem.MMFFGetMoleculeForceField(h,props)
        energy=float(ff.CalcEnergy());gradient=np.array(ff.CalcGrad()).reshape(-1,3)
        if key not in self.references:
            relaxed=Chem.Mol(h);opt=AllChem.MMFFGetMoleculeForceField(relaxed,AllChem.MMFFGetMoleculeProperties(relaxed,mmffVariant='MMFF94s'))
            status=opt.Minimize(maxIts=2000,forceTol=1e-5,energyTol=1e-8)
            if status:raise ValueError('Graph reference minimization did not converge')
            self.references[key]=float(opt.CalcEnergy())
        strain=energy-self.references[key]
        if strain < -0.1:raise ValueError('Observed energy below frozen local reference; rebaseline explicitly')
        result=dict(strain=strain,energy=energy,reference=self.references[key],gradient=gradient[:n].copy(),
            hydrogen_gradient_max=float(np.linalg.norm(gradient[n:],axis=-1).max()) if len(gradient)>n else 0.,prepared=h)
        if len(self.cache)>=128:self.cache.clear()
        self.cache[ck]=result
        return result

    def tensor(self,x,mol):
        return _Strain.apply(x,mol,self)


class _Strain(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,mol,oracle):
        result=oracle.evaluate(mol,x.detach().cpu().numpy())
        ctx.save_for_backward(x.new_tensor(result['gradient']))
        return x.new_tensor(result['strain'])

    @staticmethod
    def backward(ctx,upstream):
        gradient,=ctx.saved_tensors
        return upstream*gradient,None,None
