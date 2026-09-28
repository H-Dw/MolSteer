from ._shared import mmff
from ..core import result

def compute(ctx):
    data=mmff(ctx)
    return result({k:v for k,v in data.items() if k in ("energy_kcal_mol","hydrogen_relaxation_status")},
                  status="ok" if data["hydrogen_relaxation_status"]==0 else "partial",
                  units={"energy":"kcal/mol"},method="MMFF94s intramolecular energy; added hydrogens relaxed with original heavy atoms fixed",
                  notes=["Absolute MMFF energies across different molecules are not directly rank-comparable and are not binding free energies."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("mmff_energy")
