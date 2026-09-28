from rdkit.Chem import rdMolDescriptors
from ..core import result, need_mol

def compute(ctx):
    mol = need_mol(ctx)
    value = rdMolDescriptors.CalcFractionCSP3(mol)
    return result({"fraction_csp3": value}, units={"fraction_csp3": "[0,1]"}, method="RDKit fraction of carbon atoms that are SP3",
                  notes=["Descriptive property only. No target-specific acceptance range was supplied; no clinical or experimental property is inferred."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("fraction_csp3")
