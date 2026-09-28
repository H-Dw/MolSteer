from rdkit.Chem import Descriptors
from ..core import result, need_mol

def compute(ctx):
    mol = need_mol(ctx)
    value = Descriptors.MolWt(mol)
    return result({"molecular_weight": value}, units={"molecular_weight": "g/mol"}, method="RDKit average molecular weight with implicit hydrogens",
                  notes=["Descriptive property only. No target-specific acceptance range was supplied; no clinical or experimental property is inferred."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("molecular_weight")
