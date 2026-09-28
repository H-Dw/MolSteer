from rdkit.Chem import Crippen
from ..core import result, need_mol

def compute(ctx):
    mol = need_mol(ctx)
    value = Crippen.MolLogP(mol)
    return result({"logp": value}, units={"logp": "dimensionless"}, method="Wildman-Crippen fragment logP estimate",
                  notes=["Descriptive property only. No target-specific acceptance range was supplied; no clinical or experimental property is inferred."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("logp")
