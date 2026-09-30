"""Audit or losslessly migrate complete v1 target directories to shared batch storage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoints import atomic_save, utc_now
from .storage import migrate_target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="A target directory or run root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="Read-only field/identity/size audit")
    mode.add_argument("--output", help="New root; leave original files untouched")
    mode.add_argument("--in-place", action="store_true", help="Verify shared file, then atomically replace old files with indices")
    parser.add_argument("--report", help="Optional JSON report (including rejected partial targets)")
    options = parser.parse_args(argv)
    source = Path(options.source).resolve()
    single = any(source.glob("molecule_*.pt"))
    run_path = (source.parent if single else source) / "run.json"
    if run_path.exists() and not options.audit:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("status") in ("running", "loading_model"):
            parser.error("Generation is marked active; stop its writer before migration")
    targets = [source] if single else sorted(p for p in source.iterdir() if p.is_dir() and any(p.glob("molecule_*.pt")))
    if not targets:
        parser.error("No target directories with molecule files found")
    if options.output and Path(options.output).resolve() == source:
        parser.error("Use --in-place explicitly to migrate the source directory")
    report = {"started_utc": utc_now(), "source": str(source), "mode": "audit" if options.audit else "migration",
              "targets": [], "errors": []}
    for target in targets:
        dest = (Path(options.output) / (target.name if not single else "")) if options.output else target
        try:
            result = migrate_target(target, dest, audit_only=options.audit)
            report["targets"].append(result)
            print(json.dumps({"target": target.name, "status": result["status"],
                              "saved_bytes": result.get("saved_bytes"),
                              "batch_sharing_saved_tensor_bytes": result.get("batch_sharing_saved_tensor_bytes")}), flush=True)
        except Exception as exc:
            error = {"target": target.name, "error": str(exc)}
            report["errors"].append(error)
            print(json.dumps(error), flush=True)
    report["finished_utc"] = utc_now()
    report["summary"] = {"processed_targets": len(report["targets"]), "rejected_targets": len(report["errors"])}
    for key in ("original_file_bytes", "compact_file_bytes", "saved_bytes", "original_tensor_bytes",
                "shared_tensor_bytes", "batch_sharing_saved_tensor_bytes", "pooled_tensor_bytes"):
        report["summary"][key] = sum(t.get(key, 0) for t in report["targets"])
    if options.report:
        atomic_save(report, options.report, json_file=True)
    print(json.dumps(report["summary"], indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
