"""FLOWR suffix executor for an adaptive augmented-Lagrangian reward."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch

from molsteer.common import file_hash, write_json
from .interfaces import bounded_displacement
from .oracles import VinaOracle


def _oracle_failures(candidate, base, guard):
    failures = []
    if candidate["vina"] > base["vina"] + float(guard["vina_regression_allowance"]):
        failures.append("independent_vina_regression")
    if candidate["strain"] > base["strain"] + float(guard["strain_regression_allowance"]):
        failures.append("independent_global_strain_regression")
    return failures


def run_augmented_lagrangian_suffix(adapter, reward, output, budget, arm="creativity"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    saved = adapter.guidance_state
    if saved:
        if saved.get("arm") != arm or saved.get("program_id") != reward.spec["program_id"]:
            raise ValueError("Augmented-Lagrangian restart lineage mismatch")
        if "augmented_lagrangian" not in saved:
            raise ValueError("Guided checkpoint is missing augmented-Lagrangian dual state")
        reward.restore_controller(saved["augmented_lagrangian"])
    bindings = reward.reference.get("bindings", {})
    origin = bindings.get("origin_runtime_sha256")
    if not saved and origin and file_hash(adapter.config["resume_checkpoint"]) != origin:
        raise ValueError("Augmented-Lagrangian experiment must begin at its matched native runtime")

    guard = reward.al_cfg.get("oracle_guard")
    oracle = VinaOracle(reward.reference["oracle_config"], reward.observables, reward.vocab) if guard else None
    path = saved.get("path_used", torch.zeros_like(adapter.curr["mask"], dtype=adapter.curr["coords"].dtype)).clone()
    adapter.execution_arm = arm
    rows = []
    torch.cuda.reset_peak_memory_stats(adapter.device)
    start_t = adapter.step_index / adapter.args.integration_steps
    adapter.save_stage(output, f"t_{start_t:.2f}", start_t)
    interval = adapter.config.get("guidance_interval", [.5, 1.])
    review_every = int(reward.al_cfg.get("review_every", 5))

    for step in range(adapter.step_index, adapter.args.integration_steps):
        t = step / adapter.args.integration_steps
        next_t = (step + 1) / adapter.args.integration_steps
        reward.set_time(float(adapter.times[0][0]))
        dt = adapter.grid[step + 1] - adapter.grid[step]
        active = arm != "unguided" and interval[0] <= t < interval[1]
        row = {"step": step, "t": t, "next_t": next_t, "accepted": False, "active": active}
        gradient = None
        editable = None
        if active:
            x = adapter.curr["coords"].detach().requires_grad_(True)
            with torch.enable_grad():
                prediction, condition = adapter.predict(coordinates=x)
                try:
                    endpoint = adapter.endpoint(prediction)
                    editable = reward.editable_mask(endpoint)
                    gradient, gradient_detail = reward.gradient(endpoint, x)
                    row["gradient"] = gradient_detail
                except (ValueError, RuntimeError) as exc:
                    row["unavailable"] = str(exc)
        else:
            with torch.no_grad():
                prediction, condition = adapter.predict()

        adapter.native_step(prediction, condition, dt)
        del prediction, condition
        comparison_times = adapter.model._update_times(adapter.times, -1e-4) if next_t == 1. else adapter.times
        reward.set_time(float(comparison_times[0][0]))
        review = (step + 1) % review_every == 0 or next_t == 1.
        final_detail = None
        with torch.no_grad():
            base_prediction, _ = adapter.predict(times=comparison_times)
            base = adapter.endpoint(base_prediction)
            try:
                base_value, base_detail = reward.evaluate(base)
                final_detail = base_detail
                row["native_proposal_endpoint"] = base_detail
                row["proposal_attempts"] = []
                base_oracle = oracle.score(base) if oracle and review and active else None
                if gradient is not None:
                    mask = torch.zeros_like(adapter.curr["mask"], dtype=torch.bool)
                    mask[adapter.index] = editable
                    raw_delta = budget.strength * adapter.inject(gradient, float(dt))
                    delta = bounded_displacement(raw_delta, mask, path, budget, adapter.model.coord_scale)
                    row["raw_guidance_l2_angstrom"] = float(raw_delta[adapter.index].norm()) * adapter.model.coord_scale
                    row["bounded_guidance_l2_angstrom"] = float(delta[adapter.index].norm()) * adapter.model.coord_scale
                    row["clipped_atom_count"] = int((raw_delta[adapter.index].norm(dim=-1) * adapter.model.coord_scale > budget.max_step_angstrom + 1e-9).sum())
                    for backtrack in range(5):
                        proposal_delta = delta * .5 ** backtrack
                        proposal_prediction, _ = adapter.predict(coordinates=adapter.curr["coords"] + proposal_delta, times=comparison_times)
                        candidate = adapter.endpoint(proposal_prediction)
                        failures = reward.feasible(candidate, base)
                        graph_changed = reward.graph(candidate) != reward.graph(base)
                        candidate_oracle = None
                        if oracle and (review or graph_changed) and not failures:
                            candidate_oracle = oracle.score(candidate)
                            comparison = base_oracle or oracle.score(base)
                            failures += _oracle_failures(candidate_oracle, comparison, guard)
                        accepted, gain, reason, candidate_detail, _ = reward.compare(candidate, base)
                        if not accepted:
                            failures.append(reason)
                        row["proposal_attempts"].append({
                            "backtrack": backtrack,
                            "failures": failures,
                            "gain": gain if math.isfinite(gain) else None,
                            "graph_changed": graph_changed,
                            "oracle": candidate_oracle,
                        })
                        if not failures:
                            adapter.curr["coords"] = adapter.curr["coords"] + proposal_delta
                            path += proposal_delta.norm(dim=-1) * adapter.model.coord_scale
                            final_detail = candidate_detail
                            row.update({
                                "accepted": True,
                                "accepted_gain": gain,
                                "after": candidate_detail,
                                "injected_max_angstrom": float(proposal_delta[adapter.index].norm(dim=-1).max()) * adapter.model.coord_scale,
                            })
                            break
                if active and final_detail is not None:
                    row["dual_update"] = reward.update_duals(final_detail, step + 1)
                if review:
                    final_prediction, _ = adapter.predict(times=comparison_times)
                    current = adapter.endpoint(final_prediction)
                    row["review_oracle"] = oracle.score(current) if oracle else None
                    _, row["review_detail"] = reward.evaluate(current)
            except (ValueError, RuntimeError) as exc:
                row["comparison_unavailable"] = str(exc)
        row["injected_path_max_angstrom"] = float(path.max())
        row["dual_state_after"] = dict(reward.controller.duals)
        rows.append(row)
        adapter.guidance_state = {
            "arm": arm,
            "program_id": reward.spec["program_id"],
            "path_used": path,
            "augmented_lagrangian": reward.controller_state(),
        }
        with (output / "augmented_lagrangian_trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if adapter.step_index % 5 == 0 and next_t < 1.:
            adapter.save_stage(output, f"t_{next_t:.2f}", next_t)
            torch.save(adapter.checkpoint(), output / f"resume_t_{next_t:.2f}.pt")
        if adapter.step_index % 10 == 0:
            print(json.dumps({
                "arm": arm,
                "step": adapter.step_index,
                "accepted": sum(item["accepted"] for item in rows),
                "duals": reward.controller.duals,
            }), flush=True)

    adapter.save_stage(output, "final", 1.)
    torch.save(adapter.checkpoint(), output / "resume_final.pt")
    result = {
        "arm": arm,
        "status": "complete",
        "steps": len(rows),
        "accepted_steps": sum(row["accepted"] for row in rows),
        "gradient_steps": sum("gradient" in row for row in rows),
        "max_injected_path_angstrom": float(path.max()),
        "oracle_calls": oracle.calls if oracle else 0,
        "wall_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device),
        "any_model_parameter_grad": any(parameter.grad is not None for parameter in adapter.model.parameters()),
        "final_is_native_model_head": True,
        "posthoc_coordinate_minimization": False,
        "edited_batch_indices": [adapter.index],
        "controller": reward.controller_state(),
        "reward_architecture": "task utility with persistent augmented-Lagrangian constraints",
    }
    write_json(output / "execution_summary.json", result)
    return result
