"""Complete failed/missing targets without rewriting any completed trajectory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from molsteer.preprocessing.checkpoints import (
    TIMES, atomic_save, environment, load_bundle, seed_all, sha256, source_hashes, utc_now,
)
from molsteer.preprocessing.flowr_dataset import (
    build_args, device_for, generate_target, import_runner, load_targets, target_seed,
)


def audit_target(directory, target, config):
    """Fail closed on incomplete data; load_bundle also verifies v2 tensor checksums."""
    directory = Path(directory)
    expected = [directory / f"molecule_{i:03d}.pt" for i in range(3)]
    if set(directory.glob("molecule_*.pt")) != set(expected):
        raise ValueError(f"{target['target_id']}: expected exactly three molecule files")
    batches = {}
    decoded = sanitized = 0
    for index, path in enumerate(expected):
        bundle = load_bundle(path)
        if bundle["target"] != target or bundle["seed"] != target_seed(config["seed"], target):
            raise ValueError("Target identity or seed mismatch")
        if bundle["status"] != "completed" or bundle["verification"]["status"] != "passed":
            raise ValueError("Target has not completed exact verification")
        if (set(bundle["checkpoints"]) != set(TIMES) or
                bundle["verification"].get("resume_times") != list(TIMES) or
                bundle["verification"].get("native_generate") != "exact_match" or
                "final" not in bundle):
            raise ValueError("Missing saved times, final output or replay verification")
        if bundle["molecule_index"] != index or sum(bundle["batch_layout"]) != 3:
            raise ValueError("Incorrect molecule membership")
        batch_id = bundle["batch_id"]
        if (bundle["shared"]["batch_size"] != bundle["batch_layout"][batch_id] or
                index != sum(bundle["batch_layout"][:batch_id]) + bundle["batch_index"]):
            raise ValueError("Incorrect batch layout")
        batches.setdefault(batch_id, []).append(bundle["batch_index"])
        for key, state in bundle["checkpoints"].items():
            if state["step_index"] != round(float(key) * config["integration_steps"]):
                raise ValueError("Checkpoint integration step mismatch")
            for field in ("curr", "cond", "times", "rng", "capture"):
                if field not in state:
                    raise ValueError(f"Missing continuation field {field}")
        molecule = bundle["molecule"]
        decoded += molecule["status"] == "decoded"
        sanitized += bool(molecule.get("sanitized"))
    files = sorted(directory.glob("*.pt"))
    return {"target_id": target["target_id"], "status": "completed", "generated": 3,
            "decoded": decoded, "sanitized": sanitized,
            "batch_sizes": [len(batches[k]) for k in sorted(batches)],
            "checkpoint_count": 24, "file_bytes": sum(p.stat().st_size for p in files),
            "sha256": {p.name: sha256(p) for p in files}}


def classify(root, run, manifest):
    records = {t["target_id"]: t for t in run["targets"]}
    completed, pending = [], []
    for target in manifest["targets"]:
        if records.get(target["target_id"], {}).get("status") == "completed":
            # Never silently regenerate a supposedly completed but corrupt target.
            completed.append(audit_target(root / target["target_id"], target, run["config"]))
        else:
            pending.append(target)
    return completed, pending


def complete(root, expected_pending=None, audit_only=False):
    root = Path(root).resolve()
    original_run = json.loads((root / "run.json").read_text())
    if original_run["status"] in ("running", "loading_model"):
        raise RuntimeError("Run is marked active; inspect its process before recovery")
    config = dict(original_run["config"], verify=True, output=str(root))
    manifest = load_targets(config["split"], config["data_root"], config["expected_targets"])
    if manifest != original_run["dataset"]:
        raise ValueError("Original split/input manifest changed")
    completed, pending = classify(root, original_run, manifest)
    if expected_pending is not None and len(pending) != expected_pending:
        raise ValueError(f"Expected {expected_pending} pending targets, found {len(pending)}")
    print(json.dumps({"completed": len(completed), "pending": [t["test_index"] for t in pending]}), flush=True)
    if audit_only or not pending:
        return 0
    session = root.with_name(root.name + "_completion_" + utc_now().replace(":", "").replace("-", ""))
    session.mkdir(exist_ok=False)
    atomic_save(original_run, session / "original_run.json", json_file=True)
    atomic_save(completed, session / "preserved_targets.json", json_file=True)
    run = dict(original_run)
    records = {t["target_id"]: t for t in original_run["targets"]}
    run["completion_history"] = list(run.get("completion_history", [])) + [str(session)]
    report = {"status": "loading_model", "started_utc": utc_now(), "session": str(session),
              "preserved_count": len(completed), "pending_indices": [t["test_index"] for t in pending],
              "targets": []}

    def save(status):
        run["status"] = report["status"] = status
        run["targets"] = [records[t["target_id"]] for t in manifest["targets"] if t["target_id"] in records]
        atomic_save(report, session / "completion.json", json_file=True)
        atomic_save(run, root / "run.json", json_file=True)

    save("loading_model")
    try:
        device = device_for(config)
        runner = import_runner(config["model_root"], config["stage_runner"])
        args = build_args(runner, pending[0], config, session)
        seed_all(args.seed)
        model, hparams, *vocabs = runner.load_model(args)
        model = model.to(device).eval().requires_grad_(False)
        provenance = {"model_sha256": sha256(config["checkpoint"]),
                      "sources": source_hashes(config["model_root"], config["stage_runner"]),
                      "environment": environment(device), "split_sha256": manifest["split_sha256"],
                      "model_root": config["model_root"], "checkpoint": config["checkpoint"],
                      "stage_runner": config["stage_runner"]}
        if provenance["model_sha256"] != original_run["provenance"]["model_sha256"]:
            raise ValueError("Model weights differ from the original run")
        report["provenance"] = provenance
        # Mixed environments are recorded per bundle; the original run provenance stays intact.
        save("running")
        for target in pending:
            name = target["target_id"]
            staged = session / "generated" / name
            try:
                result = generate_target(runner, model, hparams, vocabs, target, config, provenance, staged, device)
                result["storage_audit"] = audit_target(staged, target, config)
                destination = root / name
                if destination.exists():
                    archive = session / "previous_incomplete" / name
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(destination, archive)
                os.rename(staged, destination)
            except Exception as exc:
                result = {"target_id": name, "status": "failed", "error": str(exc),
                          "traceback": traceback.format_exc()}
            records[name] = result
            report["targets"].append(result)
            save("running")
            print(json.dumps(result), flush=True)
        # Prove that all originally completed artifacts remained byte-for-byte unchanged.
        for target in completed:
            for filename, digest in target["sha256"].items():
                if sha256(root / target["target_id"] / filename) != digest:
                    raise ValueError("Previously completed artifact changed")
        report["preserved_sha256_verified"] = True
        report["finished_utc"] = run["finished_utc"] = utc_now()
        save("completed" if all(t["status"] == "completed" for t in records.values()) else "failed")
    except BaseException as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        save("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    return int(report["status"] != "completed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-pending", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    # Advisory process lock is released by the OS after crashes; no stale lock deletion.
    import fcntl  # Completion runs on the Linux inference host.
    with (Path(args.root) / ".completion.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return complete(args.root, args.expected_pending, args.audit_only)


if __name__ == "__main__":
    raise SystemExit(main())
