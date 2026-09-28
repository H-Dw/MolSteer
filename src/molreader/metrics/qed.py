from rdkit.Chem import QED
from ..core import result, need_mol

def compute(ctx):
    mol = need_mol(ctx)
    value = QED.qed(mol)
    return result({"qed": value}, units={"qed": "[0,1]"}, method="RDKit quantitative estimate of drug-likeness",
                  notes=["Descriptive property only. No target-specific acceptance range was supplied; no clinical or experimental property is inferred."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("qed")
