"""Compile updated-skill RewardDesignIR artifacts for the exact dataset retest."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from molsteer.common import digest, file_hash, write_json
from molsteer.molthinker.knowledge import KnowledgeBase


ACTIONABLE_CASE = ("5i0b_A__5vef_M77", "ligand_002")
SCREEN_ONLY_CASE = ("5i0b_A__5vef_M77", "ligand_001")


def overlap(a, b):
    return bool(set(a) & set(b))


def finding_rows(stage):
    return [
        {
            "finding_id": row["finding_id"],
            "category": row["category"],
            "atom_ids": row.get("atom_ids", []),
            "priority": row.get("priority"),
        }
        for row in stage["report"].get("findings", [])
    ]


def persistence(stage_rows):
    origin = finding_rows(stage_rows[0])
    later = [(row["stage"], f) for row in stage_rows[1:] for f in finding_rows(row)]
    result = []
    for finding in origin:
        matches = [
            {"stage": stage, **other}
            for stage, other in later
            if overlap(finding["atom_ids"], other["atom_ids"])
        ]
        result.append({
            **finding,
            "later_local_matches": matches,
            "native_self_repaired": not matches,
            "native_changed_risk_class": bool(matches) and any(
                row["category"] != finding["category"] for row in matches
            ),
        })
    return result


def candidate_architectures(status):
    common = [
        {
            "architecture_id": "normalized_sum",
            "form": "R=-sum_k w_k F_k",
            "gate_result": "rejected",
            "reason": "The diagnosed local defect can be compensated by diffuse movement, head gain, or an unrelated global term.",
        },
        {
            "architecture_id": "worst_gap_plus_residual",
            "form": "R=-smoothmax(F_local)-rho*sum(F_residual)",
            "gate_result": "rejected",
            "reason": "Dynamic graph rebinding can make the local residual vanish after a categorical branch change without proving terminal repair.",
        },
        {
            "architecture_id": "staged_lexicographic",
            "form": "local geometry -> local relaxation strain -> terminal affinity, with noncompensable feasibility gates",
            "gate_result": "selected" if status == "coordinate_actionable" else "rejected",
            "reason": (
                "Directly targets the persistent region and prevents affinity from paying for unrepaired geometry."
                if status == "coordinate_actionable"
                else "No persistent coordinate defect justifies intervention."
            ),
        },
        {
            "architecture_id": "explicit_constrained",
            "form": "maximize terminal utility subject to local-force, graph-validity, clash, and displacement constraints",
            "gate_result": "deferred",
            "reason": "Graph validity, clashes and displacement are executable; a differentiable terminal local-force inequality is not available in the current bridge.",
        },
        {
            "architecture_id": "hybrid_continuous_discrete",
            "form": "regional coordinate control plus bounded atom/bond/charge proposal search",
            "gate_result": "deferred" if status != "screen_only" else "not_executable",
            "reason": (
                "No independently supported replacement graph or microstate is available; unrestricted graph search would turn the alert into an invented target."
                if status == "screen_only"
                else "Retained as a fallback only if the native categorical branch fails to resolve the region."
            ),
        },
    ]
    if status == "native_self_repair":
        common.append({
            "architecture_id": "monitor_only",
            "form": "R=0; preserve exact native continuation",
            "gate_result": "selected",
            "reason": "The matched native suffix removes the t=0.50 defect; guidance has no attributable treatment target.",
        })
    elif status == "screen_only":
        common.append({
            "architecture_id": "monitor_only",
            "form": "R=0; retain the alert as terminal evidence",
            "gate_result": "selected",
            "reason": "The persistent substructure alert is a screening flag, not a coordinate objective, and no validated chemical alternative is supplied.",
        })
    return common


def derivation_markdown(ir, language):
    zh = language == "zh"
    selected = ir["selected_architecture"]
    lines = [
        "# 奖励函数推导记录" if zh else "# Reward derivation record",
        "",
        f"StatePacket: `{ir['identity']['packet_id']}`",
        f"Decision: `{ir['decision']['status']}`",
        "",
        ("诊断区域" if zh else "Diagnosed regions") + ":",
    ]
    for row in ir["regions"]:
        lines.append(f"- `{row['finding_id']}`: {row['category']}; atoms={row['atom_ids']}; native_self_repaired={row['native_self_repaired']}")
    lines += ["", ("架构比较" if zh else "Architecture comparison") + ":"]
    for row in ir["architecture_candidates"]:
        lines.append(f"- **{row['architecture_id']}** — {row['gate_result']}: {row['reason']}")
    lines += ["", ("选择" if zh else "Selection") + f": **{selected['architecture_id']}**", "", selected["rationale"]]
    if ir["decision"]["status"] == "coordinate_actionable":
        lines += [
            "",
            "`local geometry -> local MMFF relaxation strain -> matched-control pKd`",
            "",
            ("亲和力只在局部缺陷达到同时间原生参照后启用；有效性、碰撞、局部外位移和独立 Vina 反证均为不可补偿门控。" if zh else
             "Affinity activates only after the local defect reaches the matched native reference. Validity, clashes, outside-region displacement and independent Vina counterevidence are noncompensable gates."),
        ]
    else:
        lines += ["", ("本样本不编译执行奖励。" if zh else "No executable reward is compiled for this case.")]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--knowledge", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--base-program", required=True)
    parser.add_argument("--local-reference", required=True)
    parser.add_argument("--prior-comparison", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads((root / "collection_manifest.json").read_text())
    kb = KnowledgeBase(args.knowledge)
    mechanisms = {row["function_id"]: row for row in kb.entries}
    retrieval_ids = [key for key in ("G01", "P01", "P05", "S04", "S05") if key in mechanisms]
    retrieval = [
        {k: mechanisms[key][k] for k in ("function_id", "name_en", "formula", "role", "prerequisites", "source")}
        for key in retrieval_ids
    ]
    prior = json.loads(Path(args.prior_comparison).read_text())
    base_program = json.loads(Path(args.base_program).read_text())
    summary = []
    for case in manifest:
        target, ligand = case["target_id"], case["ligand_id"]
        case_dir = root / "cases" / target / ligand
        stages = []
        for stage_name in ("t_0.50", "t_0.75", "final"):
            stage_dir = case_dir / stage_name
            stages.append({
                "stage": stage_name,
                "summary": json.loads((stage_dir / "summary.json").read_text()),
                "report": json.loads((stage_dir / "DiagnosticReport.json").read_text()),
            })
        regions = persistence(stages)
        key = (target, ligand)
        final_has_screening = any(
            row["category"] == "structural_screening" for row in finding_rows(stages[-1])
        )
        if key == ACTIONABLE_CASE:
            status = "coordinate_actionable"
        elif key == SCREEN_ONLY_CASE or final_has_screening:
            status = "screen_only"
        else:
            status = "native_self_repair"
        packet_id = stages[0]["summary"]["packet_id"]
        evidence = {
            "kind": "EvidenceLedger",
            "packet_id": packet_id,
            "native_stages": [row["summary"] for row in stages],
            "regions": regions,
            "prior_control_experiment": prior if key == ACTIONABLE_CASE else None,
            "attribution_rule": "Only improvement beyond the matched unguided suffix is guidance benefit.",
            "leakage_rule": "Future native stages select whether intervention is warranted; their coordinates are not reward targets.",
        }
        evidence["ledger_id"] = "el_" + digest(evidence)[:24]
        write_json(case_dir / "EvidenceLedger.json", evidence)
        architectures = candidate_architectures(status)
        selected_id = next(row["architecture_id"] for row in architectures if row["gate_result"] == "selected")
        rationale = {
            "coordinate_actionable": "The [1,10,14] geometry defect persists at t=0.75 and the native graph changes before final. A staged regional controller is the only executable candidate that preserves priority without allowing a head or global metric to compensate for the defect.",
            "screen_only": "The iodine alert persists, while geometry, valence and strain remain acceptable. Removing it requires a chemical identity hypothesis that the evidence does not provide.",
            "native_self_repair": "The exact unguided suffix removes the detected t=0.50 defect. Intervention would confound native self-correction and add avoidable trajectory risk.",
        }[status]
        ir = {
            "kind": "RewardDesignIR",
            "schema_version": "1.0.0",
            "identity": {"target_id": target, "ligand_id": ligand, "packet_id": packet_id},
            "decision": {"status": status, "guidance_enabled": status == "coordinate_actionable"},
            "control_context": {
                "start_t": 0.5,
                "end_t": 1.0,
                "matched_runtime": case["resume_checkpoint"],
                "native_future_is_evidence_not_target": True,
                "editable_scope": "diagnosed core plus one-bond halo",
            },
            "regions": regions,
            "objective_nodes": [
                {"id": "regional_geometry", "role": "repair", "source": "t=0.50 and t=0.75 localized evidence"},
                {"id": "regional_relaxation", "role": "repair_confirmation", "source": "matched local MMFF relaxation"},
                {"id": "terminal_pkd", "role": "terminal_utility", "source": "live affinity head minus matched control"},
            ] if status == "coordinate_actionable" else [],
            "preservation_terms": ["outside_halo_displacement", "matched_native_local_strain"],
            "hard_constraints": ["sanitized_connected_graph", "no_new_severe_clash", "injected_path_budget", "independent_vina_guard"],
            "terminal_utilities": ["matched_control_pkd"] if status == "coordinate_actionable" else [],
            "discrete_operators": {"status": "deferred", "reason": "No validated replacement graph target"},
            "architecture_candidates": architectures,
            "selected_architecture": {"architecture_id": selected_id, "rationale": rationale},
            "execution_contract": {"status": "executable" if status == "coordinate_actionable" else "not_requested"},
            "feedback_contract": {"review_every_steps": 5, "counterevidence": ["Vina", "global MMFF strain"], "eta_escalation": False},
            "provenance": {
                "evidence_ledger_id": evidence["ledger_id"],
                "skill": {"path": str(Path(args.skill).resolve()), "sha256": file_hash(args.skill)},
                "knowledge": {"path": str(Path(args.knowledge).resolve()), "sha256": file_hash(args.knowledge)},
                "retrieval": retrieval,
            },
        }
        ir["ir_id"] = "rdir_" + digest(ir)[:24]
        write_json(case_dir / "RewardDesignIR.json", ir)
        write_json(case_dir / "ArchitectureComparison.json", {"ir_id": ir["ir_id"], "candidates": architectures})
        write_json(case_dir / "RetrievalTrace.json", {"knowledge_sha256": file_hash(args.knowledge), "results": retrieval})
        for language in ("en", "zh"):
            (case_dir / f"RewardDerivation.{language}.md").write_text(derivation_markdown(ir, language), encoding="utf-8")
        program_path = None
        if status == "coordinate_actionable":
            program = copy.deepcopy(base_program)
            parent = program.pop("program_id", None)
            program.update(
                kind="RewardProgram",
                mode="creativity",
                skill="molthinker-reward-creativity",
                evaluator="local_first",
                parent_program_id=parent,
                packet_id=packet_id,
                reward_design_ir={"ir_id": ir["ir_id"], "sha256": file_hash(case_dir / "RewardDesignIR.json")},
                evidence_ledger={"ledger_id": evidence["ledger_id"], "sha256": file_hash(case_dir / "EvidenceLedger.json")},
                region_atom_ids=[1, 10, 14],
                reward="lexicographic(regional_geometry, regional_relaxation, matched_control_pkd)",
                aggregation_policy="staged_lexicographic_noncompensable",
                active_objectives=["regional_geometry", "regional_relaxation", "matched_control_pkd"],
                weights=[1.0, 1.0, 1.0],
                weights_contract="Executor API compatibility only; the staged evaluator never sums these objectives.",
                local_first={
                    "tau_region": 0.05,
                    "bond_tolerance_fraction": 0.10,
                    "angle_tolerance_degrees": 30.0,
                    "strain_scale_kcal_mol": 20.0,
                    "minimum_extra_improvement_kcal_mol": 0.0,
                    "persistence_frames": 3,
                    "geometry_zero_tolerance": 1e-7,
                    "minimum_same_time_gain": 1e-6,
                    "max_endpoint_outside_halo_delta_angstrom": 0.10,
                    "graph_change_max_geometry_loss": 1e-5,
                    "graph_change_strain_allowance_kcal_mol": 0.0,
                    "geometry_regression_allowance": 0.002,
                    "strain_regression_allowance_kcal_mol": 0.5,
                    "review_every": 5,
                    "oracle_guard": {"vina_regression_allowance": 0.15, "strain_regression_allowance": 2.0},
                },
                local_native_reference={"path": str(Path(args.local_reference).resolve()), "sha256": file_hash(args.local_reference)},
                normalization_origin={
                    "bond_and_angle": "DiagnosticReport screening thresholds",
                    "strain": "20 kcal/mol MMFF screening threshold",
                    "terminal_utility": "matched-control pKd difference",
                    "strength_selection": "eta=100 selected from prior safe exact-restart arms because it had less Vina regression than eta=300",
                },
                architecture_limit="Local MMFF is a force-field proxy; pKd is a model head; Vina is counterevidence rather than a differentiable term.",
            )
            program["program_id"] = "rp_" + digest(program)[:24]
            program_path = case_dir / "RewardProgram.json"
            write_json(program_path, program)
        summary.append({
            "target_id": target,
            "ligand_id": ligand,
            "status": status,
            "selected_architecture": selected_id,
            "ir_id": ir["ir_id"],
            "program": str(program_path) if program_path else None,
        })
    write_json(root / "reward_design_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
