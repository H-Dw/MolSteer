import numpy as np
from ..core import result, evidence, Unavailable

def compute(ctx):
    if not np.array_equal(ctx.orders, ctx.orders.T):
        raise Unavailable("Asymmetric adjacency")
    n = len(ctx.atoms)
    remaining, components = set(range(n)), []
    while remaining:
        stack, component = [min(remaining)], []
        while stack:
            i = stack.pop()
            if i not in remaining:
                continue
            remaining.remove(i)
            component.append(i)
            stack.extend(int(j) for j in np.flatnonzero(ctx.orders[i]) if j in remaining)
        components.append(ctx.ids(sorted(component)))
    return result({"component_count": len(components), "components": components,
                   "largest_component_fraction": max(map(len, components), default=0)/n if n else None},
                  evidence=[evidence(c, message="Disconnected component", component_size=len(c)) for c in components] if len(components)>1 else [],
                  method="Connected components of active argmax bond graph", units={"component_count": "count"})

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("connectivity")
