from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .core import sha256, Unavailable

DEFAULT_CONFIG = Path(__file__).parent / "configs/flowr_root_v22.json"


def load_config(path=None):
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if config["categorical_encoding"] not in ("probabilities", "logits"):
        raise ValueError("Declare probabilities or logits explicitly")
    if not np.isfinite(config["coord_scale"]) or config["coord_scale"] <= 0:
        raise ValueError("coord_scale must be finite and positive")
    return config


def load_tensor(path):
    # No arbitrary pickle objects, GPU allocation, model import or checkpoint execution.
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dictionary: {path}")
    return obj


def array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def active_arrays(bundle, config):
    mask = array(bundle["mask"])
    if mask.ndim != 2 or mask.shape[0] != 1 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected a binary mask [1,N] for exactly one saved ligand")
    ids = np.flatnonzero(mask[0])
    n = mask.shape[1]
    shapes = {"coords": (1, n, 3), "atomics": (1, n, len(config["atomic_tokens"])),
              "charges": (1, n, len(config["charge_tokens"])),
              "bonds": (1, n, n, len(config["bond_orders"]))}
    arrays = {}
    for k, shape in shapes.items():
        a = array(bundle[k])
        if a.shape != shape:
            raise ValueError(f"{k}: expected {shape}, got {a.shape}; use the matching vocabulary adapter")
        a = a[0][ids]
        if k == "bonds":
            a = a[:, ids]
        arrays[k] = a.astype(float)
    return ids, arrays


def probabilities(a, encoding):
    if encoding == "logits":
        e = np.exp(a - np.max(a, axis=-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    return a.copy()


def parse_pdb(path):
    """First model; deterministic alternate selection; preserve residue/serial IDs."""
    if not path or not Path(path).exists():
        return [], []
    choices, notes = {}, []
    periodic = Chem.GetPeriodicTable()
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        try:
            element = line[76:78].strip().title()
            if not element:
                # Protein atom names CA/CD are carbon, never calcium/cadmium.
                element = line[12:16].strip().lstrip("0123456789")[0].title()
                notes.append("Missing PDB element inferred from first atom-name letter")
            z = periodic.GetAtomicNumber(element)
            if z <= 1:
                continue
            xyz = [float(line[a:b]) for a, b in [(30, 38), (38, 46), (46, 54)]]
            if not np.isfinite(xyz).all():
                raise ValueError("nonfinite receptor coordinate")
            res = f"{line[21].strip() or '_'}:{line[22:26].strip()}{line[26].strip()}:{line[17:20].strip()}"
            key = (res, line[12:16].strip())
            occupancy = float(line[54:60].strip() or 0)
            rank = (occupancy, line[16] == " ", line[16] == "A")
            record = dict(serial=int(line[6:11]), residue_id=res, atom_name=line[12:16].strip(),
                          residue_name=line[17:20].strip(), element=element, atomic_number=z,
                          coords=xyz, record_type=line[:6].strip(), altloc=line[16].strip())
            if key not in choices or rank > choices[key][0]:
                choices[key] = rank, record
        except (ValueError, IndexError, RuntimeError) as exc:
            raise ValueError(f"Invalid receptor record: {line[:30]}: {exc}") from exc
    return [r for _, r in choices.values()], sorted(set(notes))


def decode(coords, atoms, charges, orders):
    if any(a is None for a in atoms) or any(c is None for c in charges):
        return None, None, "Active PAD/unknown atom or charge token"
    if not np.isfinite(coords).all():
        return None, None, "Nonfinite coordinates"
    try:
        rw = Chem.RWMol()
        for symbol, charge in zip(atoms, charges):
            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(int(charge))
            rw.AddAtom(atom)
        types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
        for i, j in zip(*np.triu_indices(len(atoms), 1)):
            order = orders[i, j]
            if order:
                rw.AddBond(int(i), int(j), types[order])
        raw = rw.GetMol()
        conf = Chem.Conformer(len(atoms))
        conf.Set3D(True)
        for i, xyz in enumerate(coords):
            conf.SetAtomPosition(i, xyz.tolist())
        raw.AddConformer(conf)
        raw.UpdatePropertyCache(strict=False)
        sanitized = Chem.Mol(raw)
        code = Chem.SanitizeMol(sanitized, catchErrors=True)
        return raw, sanitized if int(code) == 0 else None, None if int(code) == 0 else str(code)
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


@dataclass
class Context:
    stage_dir: Path
    view: str
    config: dict
    bundles: dict
    metadata: dict
    atom_ids: np.ndarray
    coords: np.ndarray
    probs: dict
    atoms: list
    charges: list
    orders: np.ndarray
    raw_mol: object
    mol: object
    sanitize_error: str | None
    receptor_path: Path | None
    protein: list
    receptor_notes: list
    world_valid: bool
    transform: dict
    sdf_mol: object
    sdf_mapping_valid: bool
    sources: dict
    previous: object = None
    options: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

    @property
    def identity(self):
        return dict(target_id=self.stage_dir.parent.parent.name, ligand_id=self.stage_dir.parent.name,
                    stage=self.stage_dir.name, stage_t=float(self.metadata["stage_t"]))

    def ids(self, indices):
        return [int(self.atom_ids[i]) for i in indices]


def load_stage(stage_dir, view="prediction", receptor=None, config=None, previous=None, **options):
    stage_dir = Path(stage_dir).resolve()
    config = load_config(config) if not isinstance(config, dict) else config
    names = {"state": "state.pt", "prediction": "structure_affinity_prediction.pt", "world": "world_prediction.pt"}
    bundles = {k: load_tensor(stage_dir / v) for k, v in names.items()}
    parsed = {k: active_arrays(v, config) for k, v in bundles.items()}
    ids = parsed["prediction"][0]
    if any(not np.array_equal(ids, x[0]) for x in parsed.values()):
        raise ValueError("State, head and world masks do not identify the same atoms")
    meta = json.loads((stage_dir / "predictions.json").read_text())
    t = float(meta["stage_t"])
    if not 0 <= t <= 1 or not np.isfinite(t):
        raise ValueError("stage_t must be within [0,1]")
    root = stage_dir.parents[2]
    if receptor is None and (root / "experiment_manifest.json").exists():
        manifest = json.loads((root / "experiment_manifest.json").read_text())
        rec = manifest.get("dataset", {}).get("inputs", {}).get(stage_dir.parent.parent.name, {}).get("receptor")
        if rec:
            receptor = root / rec
    receptor = Path(receptor).resolve() if receptor else None
    protein, receptor_notes = parse_pdb(receptor)
    local = parsed["prediction"][1]["coords"]
    world = parsed["world"][1]["coords"]
    scale = config["coord_scale"]
    valid = bool(len(ids) and np.isfinite(local).all() and np.isfinite(world).all())
    shift = (world - local * scale).mean(0) if valid else np.zeros(3)
    residual = float(np.max(np.abs(world - local * scale - shift))) if valid else None
    valid = valid and residual <= config["coordinate_tolerance_angstrom"]
    categorical_equal = all(np.allclose(parsed["prediction"][1][k], parsed["world"][1][k], atol=1e-7, rtol=0, equal_nan=False) for k in ("atomics", "bonds", "charges"))
    valid = valid and categorical_equal
    transform = dict(scale=scale, translation_angstrom=shift.tolist(), max_residual_angstrom=residual,
                     categorical_fields_equal=categorical_equal, verified=valid,
                     method="Declared checkpoint scale + translation verified across paired head/world active atoms")
    sdf_path = stage_dir / "ligand.sdf"
    sdf = Chem.MolFromMolFile(str(sdf_path), sanitize=False, removeHs=False) if sdf_path.exists() and sdf_path.stat().st_size else None
    pred_atoms = [config["atomic_tokens"][i] for i in parsed["prediction"][1]["atomics"].argmax(-1)]
    mapped = bool(sdf is not None and sdf.GetNumAtoms() == len(ids) and sdf.GetNumConformers() and
                  [a.GetSymbol() for a in sdf.GetAtoms()] == pred_atoms and
                  np.allclose(sdf.GetConformer().GetPositions(), world, atol=config["coordinate_tolerance_angstrom"], rtol=0))
    if view not in ("state", "prediction", "sdf"):
        raise ValueError("view must be state, prediction, or sdf")
    a = parsed["state" if view == "state" else "prediction"][1]
    probs = {k: probabilities(a[k], config["categorical_encoding"]) for k in ("atomics", "charges", "bonds")}
    atoms = [config["atomic_tokens"][i] for i in probs["atomics"].argmax(-1)]
    charges = [config["charge_tokens"][i] for i in probs["charges"].argmax(-1)]
    orders = np.array(config["bond_orders"])[probs["bonds"].argmax(-1)]
    coords = a["coords"] * scale + shift if valid else a["coords"] * scale
    # Do not manufacture a valid graph from malformed distributions or asymmetric bonds.
    tol = config["probability_tolerance"]
    distributions_valid = all(np.isfinite(p).all() and (p >= -tol).all() and
                              np.allclose(p.sum(-1), 1, atol=tol, rtol=0) for p in probs.values())
    symmetric = np.array_equal(orders, orders.T)
    self_bonds = np.any(np.diag(orders) != 0)
    if distributions_valid and symmetric and not self_bonds:
        raw, mol, err = decode(coords, atoms, charges, orders)
    else:
        raw, mol, err = None, None, "Malformed probabilities, asymmetric bond classes or self bonds"
    if view == "sdf":
        if not mapped:
            raise Unavailable("SDF atom mapping to tensor slots cannot be verified")
        raw = Chem.Mol(sdf)
        mol = Chem.Mol(sdf)
        code = Chem.SanitizeMol(mol, catchErrors=True)
        err = None if int(code) == 0 else str(code)
        if err:
            mol = None
        coords = sdf.GetConformer().GetPositions()
        atoms = [x.GetSymbol() for x in sdf.GetAtoms()]
        charges = [x.GetFormalCharge() for x in sdf.GetAtoms()]
        orders = np.zeros((len(ids), len(ids)))
        for bond in sdf.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            orders[i, j] = orders[j, i] = bond.GetBondTypeAsDouble()
        probs = {}  # SDF has no category probabilities of its own.
    files = [stage_dir / name for name in names.values()] + [stage_dir / "predictions.json", sdf_path]
    files += [stage_dir / name for name in ('runtime.pt','runtime.json') if (stage_dir / name).exists()]
    files += [root / name for name in ('experiment_manifest.json', 'runner_metadata.json', 'stage_runner.py') if (root / name).exists()]
    if receptor and receptor.exists():
        files.append(receptor)
    sources = {p.name: dict(path=str(p), sha256=sha256(p), bytes=p.stat().st_size) for p in files if p.exists()}
    options['_categorical_valid'] = distributions_valid
    options['_graph_symmetric'] = symmetric
    options['_self_bonds'] = bool(self_bonds)
    return Context(stage_dir, view, config, bundles, meta, ids, coords, probs, atoms, charges, orders,
                   raw, mol, err, receptor, protein, receptor_notes, valid, transform, sdf, mapped, sources,
                   previous=previous, options=options)
