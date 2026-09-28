from ._shared import confidence

def compute(ctx):
    return confidence(ctx, "bonds")

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("bond_confidence")
