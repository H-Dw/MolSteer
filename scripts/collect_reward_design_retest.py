"""Collect exact-restart MolReader evidence for the RewardDesignIR retest.

The script deliberately keeps each saved stage independent.  Native future
stages are joined only in a separate comparison record, so the t=0.50
DiagnosticReport cannot leak information from t=0.75 or final.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import RDLogger

from molreader.io import load_stage
from molreader.localized_report import make_localized_report, validate_localized_report
from molreader.packet import build_packet
from molsteer.common import file_hash, write_json
from molsteer.molreader import enrich_packet
from molsteer.molreader.reporting import render_diagnostic


TARGETS = {
    "2pqw_A__2rhy_MLZ": {
        "receptor": "2pqw_A_rec_2rhy_mlz_lig_tt_min_0_pocket10.pdb",
        "reference": "2pqw_A_rec_2rhy_mlz_lig_tt_min_0.sdf",
    },
    "5i0b_A__5vef_M77": {
        "receptor": "5i0b_A_rec_5vef_m77_lig_tt_min_0_pocket10.pdb",
        "reference": "5i0b_A_rec_5vef_m77_lig_tt_min_0.sdf",
    },
}
STAGES = ("t_0.50", "t_0.75", "final")


def metric_map(packet):
    return {(row["metric_id"], row["view"]): row for row in packet["observations"]}


def compact(packet, report, stage_dir):
    metrics = metric_map(packet)

    def values(name, views=("sdf", "prediction", "state")):
        for view in views:
            row = metrics.get((name, view), {})
            if row.get("status") == "ok":
                return row.get("values", {})
        return None

    findings = report.get("findings", [])
    raw = report.get("raw_state_findings", [])
    return {
        "packet_id": packet["packet_id"],
        "stage": packet["identity"].get("stage"),
        "stage_t": packet["identity"].get("stage_t"),
        "sdf_sha256": file_hash(stage_dir / "ligand.sdf"),
        "finding_count": len(findings),
        "raw_state_finding_count": len(raw),
        "findings": [
            {
                "finding_id": row["finding_id"],
                "category": row["category"],
                "severity": row.get("severity"),
                "atom_ids": row.get("atom_ids", []),
                "scope": row.get("scope"),
                "summary": row.get("summary"),
            }
            for row in findings
        ],
        "raw_state_findings": [
            {
                "finding_id": row["finding_id"],
                "category": row["category"],
                "severity": row.get("severity"),
                "atom_ids": row.get("atom_ids", []),
                "scope": row.get("scope"),
                "summary": row.get("summary"),
            }
            for row in raw
        ],
        "affinity": values("affinity", ("prediction",)),
        "mmff_strain": values("mmff_strain"),
        "mmff_local_geometry": values("mmff_local_geometry"),
        "protein_clashes": values("protein_clashes"),
        "intramolecular_clashes": values("intramolecular_clashes"),
        "posebusters": values("posebusters"),
        "structural_alerts": values("structural_alerts"),
        "coverage": packet.get("coverage"),
        "graph_signatures": packet.get("steering", {}).get("graph_signatures"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    manifest = []
    for target, target_config in TARGETS.items():
        receptor = root / "inputs" / target_config["receptor"]
        reference = root / "inputs" / target_config["reference"]
        for ligand_index in range(3):
            ligand = f"ligand_{ligand_index:03d}"
            case_rows = []
            for stage in STAGES:
                source = root / target / ligand / stage
                destination = output / "cases" / target / ligand / stage
                contexts = [
                    load_stage(source, view, receptor=receptor, posebusters=(view == "sdf"))
                    for view in ("state", "prediction", "sdf")
                ]
                packet = enrich_packet(build_packet(contexts), contexts)
                report = make_localized_report(packet)
                validate_localized_report(report, packet)
                write_json(destination / "StatePacket.json", packet)
                write_json(destination / "DiagnosticReport.json", report)
                for language in ("en", "zh"):
                    (destination / f"DiagnosticReport.{language}.md").write_text(
                        render_diagnostic(report, language), encoding="utf-8"
                    )
                row = compact(packet, report, source)
                write_json(destination / "summary.json", row)
                case_rows.append(row)
                print(f"COLLECTED {target}/{ligand}/{stage}", flush=True)
            case = {
                "target_id": target,
                "ligand_id": ligand,
                "ligand_index": ligand_index,
                "receptor": str(receptor),
                "reference_ligand": str(reference),
                "resume_checkpoint": str(root / target / ligand / "t_0.50" / "runtime.pt"),
                "stages": case_rows,
            }
            write_json(output / "cases" / target / ligand / "native_stage_comparison.json", case)
            manifest.append(case)
    write_json(output / "collection_manifest.json", manifest)
    print(f"COLLECTION_COMPLETE {len(manifest)} cases", flush=True)


if __name__ == "__main__":
    main()
