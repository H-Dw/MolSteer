"""Read one molecule at an exact saved time, on CPU and without loading FLOWR."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from molsteer.preprocessing.checkpoints import TIMES, atomic_save, cpu_copy, load_bundle, sha256
from molsteer.preprocessing.storage import REFERENCE_FORMAT, pack, tensor_digest


def time_key(t):
    value = float(t)
    if not math.isfinite(value):
        raise ValueError("Time must be finite")
    key = f"{value:.2f}"
    if key not in TIMES or abs(value - float(key)) > 1e-8:
        raise ValueError(f"Exact saved time required; choose from {', '.join(TIMES)}")
    return key


def resolve_file(file=None, root=None, target_index=None, molecule_index=None):
    if file is not None:
        if any(x is not None for x in (root, target_index, molecule_index)):
            raise ValueError("Use --file OR --root with --target-index and --molecule-index")
        return Path(file).resolve()
    if root is None or target_index is None or molecule_index is None:
        raise ValueError("--root requires --target-index and --molecule-index (both zero-based)")
    if not 0 <= target_index < 100 or not 0 <= molecule_index < 3:
        raise ValueError("Target index must be 0..99; molecule index must be 0..2")
    matches = sorted(p for p in Path(root).glob(f"{target_index:03d}_*") if p.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected one target directory, found {len(matches)}")
    return (matches[0] / f"molecule_{molecule_index:03d}.pt").resolve()


def read_at(path, t):
    """Return (full_batch_bundle, exact_checkpoint). Treat mapped tensors as read-only.

    The complete batch, including RNG, must be retained for exact continuation.
    This function never restores RNG or imports the model or CUDA runtime.
    """
    key = time_key(t)
    bundle = load_bundle(path)  # Checks v2 metadata, every tensor and molecule membership.
    if key not in bundle["checkpoints"]:
        raise ValueError(f"t={key} not present; available: {list(bundle['checkpoints'])}")
    index = bundle["batch_index"]
    if type(index) is not int or not 0 <= index < bundle["shared"]["batch_size"]:
        raise ValueError("Invalid saved batch index")
    return bundle, bundle["checkpoints"][key]


def selected_state(bundle, state):
    """Detached copies for analysis, preserving padding/dtypes; not a resume artifact."""
    index, size = bundle["batch_index"], bundle["shared"]["batch_size"]

    def select(value):
        if torch.is_tensor(value):
            return value[index].clone() if value.ndim and value.shape[0] == size else value.clone()
        if isinstance(value, dict):
            return {k: select(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(select(v) for v in value)
        return value

    curr = select(state["curr"])
    mask = curr["mask"].bool()
    com = bundle["shared"]["com"][index].clone()
    scale = bundle["shared"]["model_signature"]["coord_scale"]
    coords_world = (curr["coords"] * scale + com) * mask.unsqueeze(-1)
    return {"format": "molsteer.flowr.selected_state.analysis.v1", "resume_supported": False,
            "target": bundle["target"], "molecule_index": bundle["molecule_index"],
            "batch_index": index, "step_index": state["step_index"], "capture": dict(state["capture"]),
            "curr": curr, "cond": select(state["cond"]), "times": select(state["times"]),
            "coords_world": coords_world, "active_atom_indices": mask.nonzero().flatten(),
            "active_coords_world": coords_world[mask].clone(), "pocket_com": com,
            "coord_scale": cpu_copy(scale)}


def tensor_info(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "bytes": value.numel() * value.element_size(), "sha256": tensor_digest(value)}
    if isinstance(value, dict):
        return {k: tensor_info(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_info(v) for v in value]
    return value


def summarize(path, bundle, state, t):
    path = Path(path).resolve()
    header = torch.load(path, map_location="cpu", weights_only=True)
    storage = {"entry_file": str(path), "entry_bytes": path.stat().st_size,
               "entry_sha256": sha256(path), "format": header["format"]}
    if header["format"] == REFERENCE_FORMAT:
        shared_path = path.parent / header["batch_file"]
        storage.update(batch_file=str(shared_path), batch_bytes=shared_path.stat().st_size,
                       batch_sha256=sha256(shared_path), batch_key=header["batch_key"],
                       integrity="verified_metadata_and_all_tensor_checksums")
    selected = selected_state(bundle, state)
    rng = state["rng"]
    return {"storage": storage, "target": bundle["target"], "molecule_index": bundle["molecule_index"],
            "batch_index": bundle["batch_index"], "batch_size": bundle["shared"]["batch_size"],
            "batch_layout": bundle["batch_layout"], "seed": bundle["seed"],
            "bundle_status": bundle["status"], "requested_t": time_key(t),
            "available_times": sorted(bundle["checkpoints"]), "step_index": state["step_index"],
            "capture": state["capture"], "created_utc": bundle.get("created_utc"),
            "completed_utc": bundle.get("completed_utc"), "wall_seconds": bundle.get("wall_seconds"),
            "recorded_verification": bundle["verification"],
            "read_operation": "checksum_validation_only_no_inference",
            "active_atom_count": len(selected["active_atom_indices"]),
            "selected_tensors": tensor_info({k: selected[k] for k in ("curr", "cond", "times", "coords_world")}),
            "rng": {"sha256": pack(rng)["sha256"], "families": list(rng),
                    "cuda_state_count": len(rng["cuda"]), "torch": tensor_info(rng["torch"]),
                    "cuda": tensor_info(rng["cuda"])},
            "generator_args": bundle["generator_args"], "provenance": bundle["provenance"],
            "final_molecule": {k: v for k, v in bundle.get("molecule", {}).items() if k != "sdf"},
            "notes": ["Indices are zero-based; batch_index selects a slot in the original full batch.",
                      "curr/cond are padded tensors in normalized model coordinates; coords_world uses saved scale and pocket COM.",
                      "t=1.00 is before the final prediction head; final.world_prediction is the final molecular output.",
                      "inference_seconds_this_segment is cumulative from this rollout start to capture, not just the previous 0.1 interval.",
                      "Exact continuation requires the full batch and saved environment; selected tensors are for analysis only."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="v1 trajectory or v2 molecule reference")
    parser.add_argument("--root", help="100-target output directory")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--molecule-index", type=int)
    parser.add_argument("--t", required=True, type=float)
    parser.add_argument("--json", help="Write information to a NEW JSON file instead of stdout")
    parser.add_argument("--export-selected", help="Write a NEW .pt with detached single-molecule tensors (analysis only)")
    options = parser.parse_args(argv)
    try:
        path = resolve_file(options.file, options.root, options.target_index, options.molecule_index)
        outputs = [Path(p).resolve() for p in (options.json, options.export_selected) if p]
        if len(outputs) != len(set(outputs)) or any(p.exists() for p in outputs):
            raise ValueError("Output files must be new and distinct; existing files are never replaced")
        bundle, state = read_at(path, options.t)
        summary = summarize(path, bundle, state, options.t)
        if options.export_selected:
            atomic_save(selected_state(bundle, state), options.export_selected)
        if options.json:
            atomic_save(summary, options.json, json_file=True)
            print(json.dumps({"status": "read", "json": str(Path(options.json).resolve()),
                              "target": bundle["target"]["target_id"], "t": time_key(options.t),
                              "molecule_index": bundle["molecule_index"]}))
        else:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
