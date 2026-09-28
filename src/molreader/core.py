from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from rdkit import Chem


class Unavailable(Exception):
    """A prerequisite is missing; this is not a passing measurement."""


def clean(value: Any):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def result(values=None, *, units=None, evidence=None, status="ok", notes=None, method="", thresholds=None):
    return dict(status=status, values=values or {}, units=units or {}, evidence=evidence or [],
                notes=notes or [], method=method, thresholds=thresholds or {})


def evidence(atoms=(), *, severity="warning", message="", **values):
    return dict(atom_ids=[int(a) for a in atoms], severity=severity, message=message, **values)


def need_mol(ctx):
    if ctx.mol is None:
        raise Unavailable("Sanitized graph unavailable: " + str(ctx.sanitize_error))
    return ctx.mol


def need_coords(ctx):
    if not len(ctx.coords) or not np.isfinite(ctx.coords).all():
        raise Unavailable("Finite coordinates for active atoms are required")
    return ctx.coords


def need_protein(ctx):
    need_coords(ctx)
    if not ctx.world_valid:
        raise Unavailable("Coordinate transformation to receptor frame was not verified")
    if not ctx.protein:
        raise Unavailable("No usable receptor heavy atoms")
    return ctx.protein


def distance_matrix(a, b=None):
    b = a if b is None else b
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def plane(coords):
    center = coords.mean(0)
    _, _, vh = np.linalg.svd(coords - center)
    normal = vh[-1]
    distances = np.abs((coords - center) @ normal)
    return center, normal, distances


def features(mol):
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    out = {}
    for f in factory.GetFeaturesForMol(mol):
        out.setdefault(f.GetFamily(), []).append(tuple(f.GetAtomIds()))
    return out


def summarize(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"n": len(x), "min": float(x.min()) if x.size else None,
            "mean": float(x.mean()) if x.size else None, "max": float(x.max()) if x.size else None}
