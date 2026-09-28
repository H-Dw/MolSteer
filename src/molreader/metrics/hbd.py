from rdkit.Chem import rdMolDescriptors
from ..core import result, need_mol

def compute(ctx):
    mol = need_mol(ctx)
    value = rdMolDescriptors.CalcNumHBD(mol)
    return result({"hbd": value}, units={"hbd": "count"}, method="RDKit hydrogen-bond donor count",
                  notes=["Descriptive property only. No target-specific acceptance range was supplied; no clinical or experimental property is inferred."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("hbd")
