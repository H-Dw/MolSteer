import numpy as np
from ..core import result, evidence
from ..io import active_arrays


def compute(ctx):
    findings, summaries = [], {}
    selected = ['state'] if ctx.view == 'state' else ['prediction', 'world'] if ctx.view == 'prediction' else []
    for source in selected:
        bundle = ctx.bundles[source]
        ids, arrays = active_arrays(bundle, ctx.config)
        summaries[source] = {}
        for key, a in arrays.items():
            nonfinite = int((~np.isfinite(a)).sum())
            item = {"shape_active": list(a.shape), "nonfinite_count": nonfinite}
            if nonfinite:
                bad = np.unique(np.argwhere(~np.isfinite(a))[:, 0])
                findings.append(evidence(ids[bad], severity="high", message=f"{source}.{key} contains nonfinite values", count=nonfinite))
            if key != "coords" and ctx.config["categorical_encoding"] == "probabilities":
                tol = ctx.config["probability_tolerance"]
                item["invalid_probability_row_count"] = int(((~np.isfinite(a).all(-1)) | (a.min(-1) < -tol) | (np.abs(a.sum(-1)-1) > tol)).sum())
                if item["invalid_probability_row_count"]:
                    findings.append(evidence(message=f"{source}.{key} invalid probability rows", severity="high", count=item["invalid_probability_row_count"]))
            summaries[source][key] = item
        b = arrays["bonds"]
        asymmetry = float(np.max(np.abs(b - b.transpose(1,0,2)))) if b.size else 0
        labels = b.argmax(-1)
        if not np.array_equal(labels, labels.T):
            pairs = np.argwhere(np.triu(labels != labels.T, 1))
            for i, j in pairs:
                findings.append(evidence(ids[[i,j]], severity="high", message=f"{source} asymmetric bond classes"))
        if np.any(np.diag(labels) != 0):
            findings.append(evidence(ids[np.diag(labels) != 0], severity="high", message=f"{source} active self bonds"))
        summaries[source]["bond_probability_asymmetry_max"] = asymmetry
    if not ctx.world_valid:
        findings.append(evidence(severity="high", message="Head/world coordinate or category correspondence failed"))
    return result(summaries, evidence=findings, method="Active-mask finite, shape, probability, graph-symmetry and world-frame checks",
                  notes=["Masked padding is excluded. Full shape validation happens at the adapter boundary."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("tensor_integrity")
