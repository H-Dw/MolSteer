"""CLI for the Pocket2Mol/TargetDiff 100-target split and per-molecule bundles."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch

from .checkpoints import (FORMAT, TIMES, assert_equal, atomic_save, configure_numerics,
                          cpu_copy, environment, load_bundle, restore_rng, seed_all,
                          sha256, snapshot_rng, source_hashes, utc_now, validate_sources)
from .trajectory import Trajectory, verify_native, verify_suffix
from .storage import write_members


REPO = Path(__file__).resolve().parents[3]


def load_targets(split_path, data_root, expected_targets=100):
    """Read the official split in its original order; never sample from train."""
    split_path, data_root = Path(split_path).resolve(), Path(data_root).resolve()
    split = torch.load(split_path, map_location="cpu", weights_only=True)
    if not isinstance(split, dict) or "test" not in split:
        raise ValueError("split_by_name.pt must contain a test list")
    pairs = split["test"]
    if len(pairs) != expected_targets or expected_targets <= 0:
        raise ValueError(f"Expected {expected_targets} test targets, found {len(pairs)}")
    targets, seen = [], set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Test entry {index} is not a receptor/ligand pair")
        if not all(isinstance(x, str) for x in pair):
            raise ValueError(f"Test entry {index} contains a missing or invalid path")
        identity = tuple(pair)
        if identity in seen:
            raise ValueError(f"Duplicate test pair at index {index}")
        seen.add(identity)
        files = [(data_root / x).resolve() for x in pair]
        for path in files:
            if not path.is_relative_to(data_root) or not path.is_file():
                raise ValueError(f"Missing input or input outside data_root: {path}")
        receptor, ligand = files
        if receptor.suffix.lower() != ".pdb" or ligand.suffix.lower() != ".sdf":
            raise ValueError(f"Test entry {index} must reference a PDB and an SDF")
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", ligand.stem)
        targets.append({"target_id": f"{index:03d}_{slug}", "test_index": index,
                        "receptor": str(receptor), "reference_ligand": str(ligand),
                        "split_pair": list(pair), "receptor_sha256": sha256(receptor),
                        "reference_ligand_sha256": sha256(ligand)})
    return {"split_path": str(split_path), "split_sha256": sha256(split_path),
            "split_key": "test", "data_root": str(data_root), "targets": targets}


def target_seed(base_seed, target):
    # Stable under retries, subsets, or a change in target execution order.
    digest = hashlib.sha256(json.dumps(target["split_pair"]).encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**32)


def import_runner(model_root, stage_runner):
    model_root, stage_runner = Path(model_root).resolve(), Path(stage_runner).resolve()
    os.environ["FLOWR_ROOT"] = str(model_root)
    sys.path.insert(0, str(model_root))
    spec = importlib.util.spec_from_file_location("molsteer_dataset_stage_runner", stage_runner)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(model_root):
        raise ValueError("A different FLOWR installation is already imported")
    return runner


def json_args(args):
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    # Fail visibly if a new upstream argument cannot be serialized.
    json.dumps(result)
    return result


def build_args(runner, target, config, output):
    args = runner.build_args(target["receptor"], target["reference_ligand"], output,
                             config["checkpoint"], gpu=config["gpu"])
    args.seed = target_seed(config["seed"], target)
    args.integration_steps = config["integration_steps"]
    args.sample_n_molecules_per_target = 3
    # Use all chains by default: test receptors are not all chain A.
    args.chain_id = config.get("chain_id")
    args.num_workers = 0
    args.corrector_iters = 0
    args.ode_sampling_strategy = "linear"
    args.solver = "euler"
    args.categorical_strategy = "uniform-sample"
    args.gpus = 0 if config["device"] == "cpu" else 1
    args.mp_index = config["gpu"]
    return args


def input_bytes(target):
    return {name: Path(target[name]).read_bytes() for name in ("receptor", "reference_ligand")}


def decode_final(model, final):
    """Keep decoding failures as data; do not replace trajectories by resampling."""
    from rdkit import Chem
    results = []
    world = final["world_prediction"]
    batch_size = world["coords"].shape[0]
    for i in range(batch_size):
        def select(value):
            if torch.is_tensor(value):
                return value[i:i + 1] if value.ndim and value.shape[0] == batch_size else value
            if isinstance(value, dict):
                return {k: select(v) for k, v in value.items()}
            return value
        try:
            mol = model._generate_mols(select(world), sanitise=False)[0]
            if mol is None:
                raise ValueError("FLOWR decoder returned None")
            block = Chem.MolToMolBlock(mol)
            try:
                Chem.SanitizeMol(Chem.Mol(mol))
                results.append({"status": "decoded", "sanitized": True, "sdf": block + "\n$$$$\n"})
            except Exception as exc:
                results.append({"status": "decoded", "sanitized": False, "sdf": block + "\n$$$$\n",
                                "sanitization_error": str(exc)})
        except Exception as exc:
            results.append({"status": "decode_failed", "sdf": None, "error": str(exc)})
    return results


def generate_target(runner, model, hparams, vocabs, target, config, provenance, output, device):
    started = time.perf_counter()
    args = build_args(runner, target, config, output)
    seed_all(args.seed)
    transform, interpolant = runner.util.load_util(args, hparams, *vocabs[:4])
    system = runner.util.load_data_from_pdb(
        args, remove_hs=hparams["remove_hs"], remove_aromaticity=hparams["remove_aromaticity"],
        ligand_idx=args.ligand_idx, chain_id=args.chain_id, canonicalize_conformer=False)
    dataset = runner.get_dataset(system, transform, vocabs[0], interpolant, args, hparams)
    loader = runner.util.get_dataloader(args, dataset, interpolant, iter=0)
    # Resolve all prior draws before generation, recording the actual resulting batch layout.
    batches = list(loader)
    sizes = [int(batch[0]["coords"].shape[0]) for batch in batches]
    if not sizes or sum(sizes) != 3 or any(size <= 0 for size in sizes):
        raise ValueError(f"Expected exactly 3 original samples, got batch sizes {sizes}")
    offset, decoded_count, sanitized_count = 0, 0, 0
    for batch_id, (prior, data, _, _) in enumerate(batches):
        count = sizes[batch_id]
        ligand = model.builder.extract_ligand_from_complex(prior)
        for key in ("interactions", "fragment_mask", "fragment_mode"):
            ligand[key] = prior[key]
        pocket = model.builder.extract_pocket_from_complex(data)
        pocket.update(interactions=data["interactions"], complex=data["complex"])
        initial_rng = snapshot_rng()
        trajectory = Trajectory(model, ligand, pocket, args.integration_steps, device)
        bundle = {"format": FORMAT, "status": "running", "created_utc": utc_now(),
                  "target": target, "inputs": input_bytes(target), "seed": args.seed,
                  "batch_id": batch_id, "batch_layout": sizes, "generator_args": json_args(args),
                  "provenance": provenance, "shared": trajectory.shared(), "checkpoints": {},
                  "requested_times": list(TIMES), "verification": {"status": "not_run"}}

        def capture(key, state):
            bundle["checkpoints"][key] = state
            bundle["wall_seconds"] = time.perf_counter() - started
            write_members(bundle, output, offset, count)
            print(json.dumps({"target": target["target_id"], "batch": batch_id, "t": key,
                              "status": "checkpoint_saved"}), flush=True)

        try:
            _, final = trajectory.run(capture)
            bundle["final"] = final
            bundle["status"] = "generated"
            write_members(bundle, output, offset, count)
            verification_started = time.perf_counter()
            if config["verify"]:
                verify_native(model, trajectory, initial_rng, final)
                # Verify persisted bytes, not only in-memory objects.
                saved = load_bundle(Path(output) / f"molecule_{offset:03d}.pt")
                for key in TIMES:
                    verify_suffix(model, saved, key, device)
                    print(json.dumps({"target": target["target_id"], "batch": batch_id,
                                      "resumed_from": key, "status": "exact_match"}), flush=True)
                del saved
                bundle["verification"] = {"status": "passed", "native_generate": "exact_match",
                                          "resume_times": list(TIMES), "comparison": "torch.equal_all_tensors_and_rng"}
            else:
                bundle["verification"] = {"status": "not_run", "reason": "explicit --no-verify"}
            bundle["verification"]["seconds"] = time.perf_counter() - verification_started
            restore_rng(final["rng"])
            try:
                bundle["decoded"] = decode_final(model, final)
            finally:
                # Verification and chemical decoding must not change the next batch's stream.
                restore_rng(final["rng"])
            decoded_count += sum(x["status"] == "decoded" for x in bundle["decoded"])
            sanitized_count += sum(x.get("sanitized", False) for x in bundle["decoded"])
            bundle["status"] = "completed" if config["verify"] else "completed_unverified"
            bundle["completed_utc"] = utc_now()
            bundle["wall_seconds"] = time.perf_counter() - started
            write_members(bundle, output, offset, count)
        except BaseException as exc:
            bundle["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            bundle["error"] = {"type": type(exc).__name__, "message": str(exc), "utc": utc_now()}
            write_members(bundle, output, offset, count)
            raise
        offset += count
    return {"target_id": target["target_id"], "status": "completed" if config["verify"] else "completed_unverified",
            "generated": offset, "decoded": decoded_count, "sanitized": sanitized_count,
            "batch_sizes": sizes, "wall_seconds": time.perf_counter() - started}


def device_for(config):
    device = torch.device("cpu" if config["device"] == "cpu" else f"cuda:{config['gpu']}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA/DTK accelerator is available. Allocate a GPU or explicitly select --device cpu.")
    configure_numerics(device)
    return device


def generate(config, dry_run=False, target_index=None):
    manifest = load_targets(config["split"], config["data_root"], config["expected_targets"])
    if dry_run:
        report = {"status": "inputs_validated", "target_count": len(manifest["targets"]),
                  "molecule_count": 3 * len(manifest["targets"]), "checkpoints_per_molecule": len(TIMES),
                  "split_sha256": manifest["split_sha256"], "config": config, "dataset": manifest}
        if config.get("report"):
            atomic_save(report, config["report"], json_file=True)
        print(json.dumps({k: v for k, v in report.items() if k not in ("config", "dataset")}, indent=2))
        return 0
    targets = manifest["targets"]
    if target_index is not None:
        targets = [targets[target_index]]
    device = device_for(config)
    runner = import_runner(config["model_root"], config["stage_runner"])
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=False)
    run = {"status": "loading_model", "started_utc": utc_now(), "config": config,
           "dataset": manifest, "selected_test_indices": [t["test_index"] for t in targets], "targets": []}
    atomic_save(run, root / "run.json", json_file=True)
    try:
        args = build_args(runner, targets[0], config, root)
        seed_all(args.seed)
        model, hparams, *vocabs = runner.load_model(args)
        model = model.to(device).eval().requires_grad_(False)
        provenance = {"model_sha256": sha256(config["checkpoint"]),
                      "sources": source_hashes(config["model_root"], config["stage_runner"]),
                      "environment": environment(device), "split_sha256": manifest["split_sha256"],
                      "model_root": config["model_root"], "checkpoint": config["checkpoint"],
                      "stage_runner": config["stage_runner"]}
        run["provenance"] = provenance
        run["status"] = "running"
        for target in targets:
            output = root / target["target_id"]
            try:
                result = generate_target(runner, model, hparams, vocabs, target, config, provenance, output, device)
            except Exception as exc:
                result = {"target_id": target["target_id"], "status": "failed", "error": str(exc),
                          "traceback": traceback.format_exc()}
            run["targets"].append(result)
            atomic_save(run, root / "run.json", json_file=True)
            print(json.dumps(result), flush=True)
        run["status"] = "failed" if any(t["status"] == "failed" for t in run["targets"]) else (
            "completed" if config["verify"] else "completed_unverified")
    except BaseException as exc:
        run["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        run["error"] = str(exc)
        raise
    finally:
        run["finished_utc"] = utc_now()
        atomic_save(run, root / "run.json", json_file=True)
    return int(run["status"] == "failed")


def resume(options):
    bundle = load_bundle(options.file)
    key = f"{options.t:.2f}"
    if key not in TIMES or abs(float(key) - options.t) > 1e-8 or key not in bundle["checkpoints"]:
        raise ValueError(f"Checkpoint t={options.t} not present; available: {list(bundle['checkpoints'])}")
    output = Path(options.output)
    if output.exists():
        raise FileExistsError("Use a new resume output file")
    provenance = bundle["provenance"]
    model_root = options.model_root or provenance["model_root"]
    stage_runner = options.stage_runner or provenance["stage_runner"]
    checkpoint = options.checkpoint or provenance["checkpoint"]
    config = {"device": provenance["environment"]["device"].split(":")[0],
              "gpu": int(provenance["environment"]["device"].split(":")[-1])
              if provenance["environment"]["device"].startswith("cuda:") else 0}
    device = device_for(config)
    runner = import_runner(model_root, stage_runner)
    source_validation = validate_sources(provenance["sources"], source_hashes(model_root, stage_runner))
    assert_equal(provenance["model_sha256"], sha256(checkpoint), "model_sha256")
    args = argparse.Namespace(**bundle["generator_args"])
    args.ckpt_path, args.save_dir = str(checkpoint), str(output.parent)
    model, *_ = runner.load_model(args)
    model = model.to(device).eval().requires_grad_(False)
    assert_equal(provenance["environment"], environment(device), "environment")
    started = time.perf_counter()
    checkpoints, final = verify_suffix(model, bundle, key, device)
    result = dict(bundle)
    result["checkpoints"] = dict(bundle["checkpoints"], **checkpoints)
    result["final"] = final
    result["status"] = "resumed_verified" if "final" in bundle else "resumed_without_original_final"
    result["resume"] = {"from_t": key, "source_sha256": sha256(options.file),
                        "source_validation": source_validation,
                        "utc": utc_now(), "seconds": time.perf_counter() - started,
                        "original_final_exact_match": "final" in bundle}
    # A resumed partial artifact may not have had a decoded molecule yet.
    restore_rng(final["rng"])
    try:
        result["molecule"] = decode_final(model, final)[bundle["batch_index"]]
    finally:
        restore_rng(final["rng"])
    atomic_save(result, output)
    print(json.dumps(result["resume"], indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate 100 x 3 trajectories (8 checkpoints each)")
    gen.add_argument("--config", default=str(REPO / "configs/flowr_100target.json"))
    for key in ("split", "data-root", "model-root", "stage-runner", "checkpoint", "output", "report"):
        gen.add_argument("--" + key)
    gen.add_argument("--device", choices=("cuda", "cpu"))
    for key in ("gpu", "seed", "integration-steps", "expected-targets"):
        gen.add_argument("--" + key, type=int)
    gen.add_argument("--target-index", type=int, help="Explicit smoke-test subset; still validates the full split")
    gen.add_argument("--no-verify", action="store_true", help="Skip native/suffix checks; results are marked unverified")
    gen.add_argument("--dry-run", action="store_true", help="Validate split and all 200 input files without loading FLOWR")
    cont = commands.add_parser("resume", help="Resume one bundle without preprocessing or prefix replay")
    cont.add_argument("--file", required=True)
    cont.add_argument("--t", type=float, required=True)
    cont.add_argument("--output", required=True)
    for key in ("model-root", "stage-runner", "checkpoint"):
        cont.add_argument("--" + key, help="Relocate paths; source/weight hashes must still match")
    options = parser.parse_args(argv)
    if options.command == "resume":
        return resume(options)
    config_path = Path(options.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("split", "data_root", "model_root", "stage_runner", "checkpoint", "output", "report",
                "device", "gpu", "seed", "integration_steps", "expected_targets"):
        value = getattr(options, key)
        if value is not None:
            config[key] = value
    for key in ("split", "data_root", "model_root", "stage_runner", "checkpoint", "output", "report"):
        if config.get(key):
            path = Path(os.path.expandvars(config[key])).expanduser()
            config[key] = str((REPO / path).resolve() if not path.is_absolute() else path.resolve())
    config["verify"] = not options.no_verify
    if config["integration_steps"] < 10 or config["integration_steps"] % 10:
        parser.error("integration-steps must be a positive multiple of 10")
    if not 0 <= config["seed"] < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if options.target_index is not None and not 0 <= options.target_index < config["expected_targets"]:
        parser.error("target-index outside the test split")
    return generate(config, options.dry_run, options.target_index)


if __name__ == "__main__":
    raise SystemExit(main())
