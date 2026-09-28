from collections import Counter
from ..core import result, evidence

def compute(ctx):
    unknown = [int(i) for i, a in zip(ctx.atom_ids, ctx.atoms) if a is None]
    return result({"active_atoms": len(ctx.atoms), "elements": dict(Counter(a or "UNKNOWN" for a in ctx.atoms)),
                   "known_heavy_atoms": sum(a not in ("H", None) for a in ctx.atoms)},
                  evidence=[evidence(unknown, message="Active special tokens do not specify chemical elements")] if unknown else [],
                  method="Adapter vocabulary decoding over active mask", units={"counts": "atoms"})

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("atom_inventory")
