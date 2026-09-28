"""Execute the RewardDesignIR-selected exact continuation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

from molreader.io import load_config, parse_pdb
from molsteer.common import file_hash, write_json
from molsteer.molexecutor.flowr import FlowrRootAdapter
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molexecutor.local_first_engine import run_local_first_suffix
from molsteer.molexecutor.program import make_reward


def different(left, right):
    if isinstance(left, torch.Tensor):
        return not torch.equal(left, right)
    if isinstance(left, dict):
        return set(left) != set(right) or any(different(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) != len(right) or any(different(a, b) for a, b in zip(left, right))
    return left != right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--base-execution", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = Path(args.source_root).resolve()
    target, ligand, ligand_index = "5i0b_A__5vef_M77", "ligand_002", 2
    case_dir = root / "cases" / target / ligand
    program_path = case_dir / "RewardProgram.json"
    program = json.loads(program_path.read_text())
    config = json.loads(Path(args.base_execution).read_text())
    saved_stage = source / "output/crossdocked_100target_stage_test_exact_20260923" / target / ligand / "t_0.50"
    native_final_runtime = source / "output/crossdocked_100target_stage_test_exact_20260923" / target / ligand / "final/runtime.pt"
    config.update(
        output=str(root / "continuations" / target / ligand / "staged_lexicographic_eta100"),
        saved_stage=str(saved_stage),
        resume_checkpoint=str(saved_stage / "runtime.pt"),
        reward_reference_stage=str(saved_stage),
        target_id=target,
        ligand_index=ligand_index,
        start_step=50,
        reward_programs={"creativity": str(program_path)},
        guidance_interval=[0.5, 1.0],
        budget={"strength": 100.0, "max_step_angstrom": 0.04, "max_path_angstrom": 2.0},
        record_tensor_trace=True,
    )
    config.pop("monitor", None)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution.json", config)
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    checkpoint = torch.load(config["resume_checkpoint"], weights_only=True, map_location="cpu")
    adapter = FlowrRootAdapter(config)
    adapter.restore(checkpoint)
    baseline = torch.load(saved_stage / "world_prediction.pt", weights_only=True, map_location=adapter.device)
    baseline = {key: value[0] for key, value in baseline.items() if torch.is_tensor(value)}
    receptor, _ = parse_pdb(config["receptor"])
    periodic = Chem.GetPeriodicTable()
    for atom in receptor:
        atom["vdw_radius"] = periodic.GetRvdw(atom["atomic_number"])
    controls = json.loads(Path(config["control_trajectory"]).read_text())
    reward = make_reward(program, baseline, receptor, load_config(), controls[program["affinity_head"]])
    summary = run_local_first_suffix(
        adapter,
        reward,
        output / "creativity",
        GuidanceBudget(**config["budget"]),
        "staged_lexicographic_eta100",
    )
    write_json(root / "runs.json", [{"target_id": target, "ligand_id": ligand, "label": "staged_lexicographic_eta100", "summary": summary}])

    guided = torch.load(output / "creativity/resume_final.pt", weights_only=True, map_location="cpu")
    native = torch.load(native_final_runtime, weights_only=True, map_location="cpu")
    non_target = {}
    for field in ("coords", "atomics", "bonds", "charges", "mask"):
        non_target[field] = torch.equal(guided["curr"][field][:ligand_index], native["curr"][field][:ligand_index])
    fidelity = {
        "origin_runtime_sha256": file_hash(saved_stage / "runtime.pt"),
        "native_final_runtime_sha256": file_hash(native_final_runtime),
        "program_sha256": file_hash(program_path),
        "rng_exact": not different(guided["rng"], native["rng"]),
        "times_exact": not different(guided["times"], native["times"]),
        "non_target_batch_exact": non_target,
        "all_non_target_exact": all(non_target.values()),
        "self_condition_expected_to_diverge_for_target": True,
        "same_random_seed_sweep": False,
    }
    write_json(root / "runtime_fidelity.json", fidelity)
    if not fidelity["rng_exact"] or not fidelity["times_exact"] or not fidelity["all_non_target_exact"]:
        raise ValueError("Exact continuation fidelity check failed")
    print(json.dumps({"summary": summary, "fidelity": fidelity}, indent=2), flush=True)


if __name__ == "__main__":
    main()
