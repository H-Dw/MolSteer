"""Adaptive augmented-Lagrangian reward with checkpointable dual state.

The scalar reward is a task utility minus one Lagrange and one quadratic
penalty term per declared inequality.  Constraint implementations are kept in
a registry so model integrations can add differentiable residuals without
changing the controller or the execution engine.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import torch
from rdkit import Chem

from molsteer.common import file_hash
from .chemistry import decode_endpoint, signature
from .local_first_reward import LocalMMFFRelaxation, local_geometry, region_halo
from .outcome_reward import OutcomeAwareReward


ConstraintEvaluator = Callable[["AugmentedLagrangianReward", dict, Chem.Mol, dict], tuple[torch.Tensor, dict]]
CONSTRAINT_EVALUATORS: dict[str, ConstraintEvaluator] = {}


def register_constraint(kind: str, evaluator: ConstraintEvaluator) -> None:
    """Register one differentiable inequality residual implementation."""
    if not kind or kind in CONSTRAINT_EVALUATORS:
        raise ValueError("Duplicate or empty augmented-Lagrangian constraint kind")
    CONSTRAINT_EVALUATORS[kind] = evaluator


def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"Invalid {name}")
    return number


class AugmentedLagrangianController:
    """Persistent nonnegative multipliers for h(x) <= 0 constraints."""

    schema_version = 1

    def __init__(self, program_id: str, constraints: list[dict], update_every: int = 1):
        if not program_id or not constraints:
            raise ValueError("Augmented-Lagrangian control needs a program id and constraints")
        self.program_id = program_id
        self.update_every = int(update_every)
        if self.update_every <= 0:
            raise ValueError("dual_update_every must be positive")
        definitions = []
        self.duals = {}
        self.satisfied_frames = {}
        for spec in constraints:
            cid = str(spec.get("id", ""))
            if not cid or cid in self.duals:
                raise ValueError("Constraint ids must be nonempty and unique")
            penalty = _positive(spec.get("penalty"), f"{cid}.penalty")
            initial = _positive(spec.get("lambda_initial", 0.0), f"{cid}.lambda_initial", allow_zero=True)
            maximum = _positive(spec.get("lambda_max"), f"{cid}.lambda_max")
            if initial > maximum:
                raise ValueError(f"{cid}.lambda_initial exceeds lambda_max")
            tolerance = _positive(spec.get("satisfaction_tolerance", 0.0), f"{cid}.satisfaction_tolerance", allow_zero=True)
            frames = int(spec.get("satisfaction_frames", 1))
            decay = float(spec.get("satisfied_decay", 1.0))
            if frames <= 0 or not 0 < decay <= 1:
                raise ValueError(f"Invalid persistence/decay for {cid}")
            definitions.append({
                "id": cid,
                "kind": spec.get("kind"),
                "penalty": penalty,
                "lambda_max": maximum,
                "satisfaction_tolerance": tolerance,
                "satisfaction_frames": frames,
                "satisfied_decay": decay,
            })
            self.duals[cid] = initial
            self.satisfied_frames[cid] = 0
        payload = json.dumps(
            {"constraints": constraints, "dual_update_every": self.update_every},
            sort_keys=True, separators=(",", ":"),
        )
        self.definition_sha256 = hashlib.sha256(payload.encode()).hexdigest()
        self.update_count = 0
        self.last_updated_step = None
        self.history = []
        self._definitions = {row["id"]: row for row in definitions}

    def update(self, violations: dict[str, float], step: int) -> dict:
        step = int(step)
        if self.last_updated_step == step:
            return {"updated": False, "reason": "already_updated", "step": step}
        if step % self.update_every:
            return {"updated": False, "reason": "update_interval", "step": step}
        if set(violations) != set(self.duals):
            raise ValueError("Dual update constraint set differs from the reward definition")
        before = dict(self.duals)
        rows = {}
        for cid, value in violations.items():
            h = float(value)
            if not math.isfinite(h) or h < 0:
                raise ValueError(f"Invalid nonnegative violation for {cid}")
            cfg = self._definitions[cid]
            if h <= cfg["satisfaction_tolerance"]:
                self.satisfied_frames[cid] += 1
                if self.satisfied_frames[cid] >= cfg["satisfaction_frames"]:
                    self.duals[cid] *= cfg["satisfied_decay"]
            else:
                self.satisfied_frames[cid] = 0
                self.duals[cid] = min(cfg["lambda_max"], self.duals[cid] + cfg["penalty"] * h)
            rows[cid] = {
                "violation": h,
                "lambda_before": before[cid],
                "lambda_after": self.duals[cid],
                "satisfied_frames": self.satisfied_frames[cid],
            }
        self.update_count += 1
        self.last_updated_step = step
        event = {"updated": True, "step": step, "constraints": rows}
        self.history.append(event)
        max_history = 256
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        return event

    def state_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "program_id": self.program_id,
            "definition_sha256": self.definition_sha256,
            "duals": dict(self.duals),
            "satisfied_frames": dict(self.satisfied_frames),
            "update_count": self.update_count,
            "last_updated_step": self.last_updated_step,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("schema_version", -1)) != self.schema_version:
            raise ValueError("Unsupported augmented-Lagrangian dual-state schema")
        if state.get("program_id") != self.program_id:
            raise ValueError("Dual-state reward program lineage mismatch")
        if state.get("definition_sha256") != self.definition_sha256:
            raise ValueError("Dual-state constraint definition mismatch")
        if set(state.get("duals", {})) != set(self.duals):
            raise ValueError("Dual-state constraint ids mismatch")
        if set(state.get("satisfied_frames", {})) != set(self.satisfied_frames):
            raise ValueError("Dual-state persistence ids mismatch")
        duals = {key: float(value) for key, value in state["duals"].items()}
        if any(not math.isfinite(value) or value < 0 or value > self._definitions[key]["lambda_max"] for key, value in duals.items()):
            raise ValueError("Dual-state multiplier is outside its declared bounds")
        frames = {key: int(value) for key, value in state["satisfied_frames"].items()}
        if any(value < 0 for value in frames.values()):
            raise ValueError("Dual-state persistence counter is negative")
        self.duals = duals
        self.satisfied_frames = frames
        self.update_count = int(state.get("update_count", 0))
        self.last_updated_step = state.get("last_updated_step")
        if self.last_updated_step is not None:
            self.last_updated_step = int(self.last_updated_step)
        self.history = list(state.get("history", []))[-256:]


def _local_geometry_constraint(reward, pred, mol, spec):
    value, detail = local_geometry(
        pred["coords"], mol, reward.core,
        spec["bond_tolerance_fraction"], spec["angle_tolerance_degrees"], spec["tau_region"],
    )
    native = reward.native_local()
    margin = _positive(spec["improvement_margin"], f"{spec['id']}.improvement_margin")
    scale = _positive(spec["scale"], f"{spec['id']}.scale")
    target = max(0.0, float(native["local_geometry_loss"]) - margin)
    violation = torch.relu((value - target) / scale)
    return violation, {
        "raw": float(value.detach()), "target": target, "scale": scale,
        "native": float(native["local_geometry_loss"]), "geometry": detail,
        "native_reference_mode": native.get("reference_mode", "unknown"),
        "native_reference_bracket": native.get("reference_bracket"),
    }


def _local_strain_constraint(reward, pred, mol, spec):
    value, detail = reward.local_mmff.tensor(pred["coords"], mol, reward.core)
    native = reward.native_local()
    margin = _positive(spec["improvement_margin_kcal_mol"], f"{spec['id']}.improvement_margin_kcal_mol")
    scale = _positive(spec["scale_kcal_mol"], f"{spec['id']}.scale_kcal_mol")
    target = max(0.0, float(native["local_strain_kcal_mol"]) - margin)
    violation = torch.relu((value - target) / scale)
    return violation, {
        "raw": float(value.detach()), "target": target, "scale": scale,
        "native": float(native["local_strain_kcal_mol"]),
        "native_reference_mode": native.get("reference_mode", "unknown"),
        "native_reference_bracket": native.get("reference_bracket"),
        "local_relaxation": {key: val for key, val in detail.items() if key != "gradient"},
    }


def _outside_anchor_constraint(reward, pred, mol, spec):
    halo = set(region_halo(mol, reward.core))
    outside = [idx for idx in range(len(pred["coords"])) if idx not in halo]
    if not outside:
        zero = pred["coords"].sum() * 0
        return zero, {"raw": 0.0, "target": float(spec["tolerance_angstrom"]), "outside_atom_count": 0}
    delta = pred["coords"][outside] - reward.x0[outside]
    rmsd = torch.sqrt(delta.square().sum(-1).mean().clamp(min=1e-20))
    tolerance = _positive(spec["tolerance_angstrom"], f"{spec['id']}.tolerance_angstrom")
    scale = _positive(spec["scale_angstrom"], f"{spec['id']}.scale_angstrom")
    violation = torch.relu((rmsd - tolerance) / scale)
    return violation, {
        "raw": float(rmsd.detach()), "target": tolerance, "scale": scale,
        "outside_atom_count": len(outside), "reference": "reward_start_endpoint",
    }


register_constraint("local_geometry_advantage", _local_geometry_constraint)
register_constraint("local_strain_advantage", _local_strain_constraint)
register_constraint("outside_anchor", _outside_anchor_constraint)


class AugmentedLagrangianReward(OutcomeAwareReward):
    """Task utility under localized, adaptively weighted inequality constraints."""

    def __init__(self, program, baseline, receptor, vocabulary, control):
        super().__init__(program, baseline, receptor, vocabulary, control)
        self.al_cfg = program["augmented_lagrangian"]
        self.core = tuple(int(i) for i in program["region_atom_ids"])
        if not self.core:
            raise ValueError("Localized augmented-Lagrangian reward requires a diagnosed core")
        path = Path(program["local_native_reference"]["path"])
        if file_hash(path) != program["local_native_reference"]["sha256"]:
            raise ValueError("Local native reference changed")
        reference = json.loads(path.read_text(encoding="utf-8"))
        self.local_frames = sorted(reference["frames"], key=lambda row: row["time"])
        self.local_times = [float(row["time"]) for row in self.local_frames]
        self.local_mmff = LocalMMFFRelaxation()
        constraints = list(self.al_cfg["constraints"])
        unknown = sorted({row.get("kind") for row in constraints} - set(CONSTRAINT_EVALUATORS))
        if unknown:
            raise ValueError("Unknown augmented-Lagrangian constraint kinds: " + ", ".join(unknown))
        self.constraint_specs = constraints
        self.controller = AugmentedLagrangianController(
            program["program_id"], constraints, self.al_cfg.get("dual_update_every", 1),
        )

    def native_local(self):
        if not self.local_frames:
            raise ValueError("Empty local native reference")
        tolerance = float(self.al_cfg.get("native_time_tolerance", .002))
        nearest = min(self.local_frames, key=lambda row: abs(float(row["time"]) - self.time))
        if abs(float(nearest["time"]) - self.time) <= tolerance and nearest.get("status") == "ok":
            return {**nearest, "reference_mode": "exact"}
        valid = [row for row in self.local_frames if row.get("status") == "ok"]
        before = [row for row in valid if float(row["time"]) < self.time]
        after = [row for row in valid if float(row["time"]) > self.time]
        if not before or not after:
            raise ValueError("Local native reference unavailable at this boundary")
        a, b = before[-1], after[0]
        span = float(b["time"]) - float(a["time"])
        max_span = float(self.al_cfg.get("native_interpolation_max_gap", 0.0))
        if span <= 0 or max_span <= 0 or span > max_span:
            raise ValueError("Local native reference gap exceeds the declared interpolation limit")
        fraction = (self.time - float(a["time"])) / span
        keys = ("local_geometry_loss", "local_geometry_max_residual", "local_strain_kcal_mol")
        row = {
            key: float(a[key]) + fraction * (float(b[key]) - float(a[key]))
            for key in keys
        }
        row.update({
            "time": self.time,
            "status": "ok",
            "reference_mode": "linear_scalar_interpolation",
            "reference_bracket": [float(a["time"]), float(b["time"])],
        })
        return row

    def editable_mask(self, pred):
        mol = decode_endpoint(pred, self.vocab)
        mask = torch.zeros(len(pred["coords"]), dtype=torch.bool, device=pred["coords"].device)
        scope = self.al_cfg.get("editable_scope", "region_halo")
        if scope == "region_halo":
            mask[list(region_halo(mol, self.core))] = True
        elif scope == "all_ligand_atoms":
            mask[:] = True
        else:
            raise ValueError("Unknown augmented-Lagrangian editable scope")
        return mask

    def terms(self, pred):
        mol = decode_endpoint(pred, self.vocab)
        utility, affinity = self.affinity_utility(pred)
        weighted_utility = float(self.al_cfg.get("utility_weight", 1.0)) * utility
        violations = {}
        details = {}
        penalties = {}
        for spec in self.constraint_specs:
            cid = spec["id"]
            h, detail = CONSTRAINT_EVALUATORS[spec["kind"]](self, pred, mol, spec)
            if h.ndim != 0 or not torch.isfinite(h) or float(h.detach()) < 0:
                raise ValueError(f"Constraint {cid} did not produce a finite nonnegative scalar")
            lam = pred["coords"].new_tensor(self.controller.duals[cid])
            kappa = pred["coords"].new_tensor(float(spec["penalty"]))
            contribution = lam * h + .5 * kappa * h.square()
            violations[cid] = h
            penalties[cid] = contribution
            details[cid] = {
                **detail,
                "kind": spec["kind"], "violation": float(h.detach()),
                "lambda": float(lam), "penalty": float(kappa),
                "augmented_penalty": float(contribution.detach()),
            }
        reward = weighted_utility - sum(penalties.values(), pred["coords"].sum() * 0)
        detail = {
            "architecture": "adaptive_augmented_lagrangian",
            "reward": float(reward.detach()),
            "affinity": float(affinity.detach()),
            "control_affinity": self.control_value(),
            "affinity_utility": float(utility.detach()),
            "weighted_task_utility": float(weighted_utility.detach()),
            "constraints": details,
            "duals": dict(self.controller.duals),
            "time": self.time,
            "smiles": Chem.MolToSmiles(mol),
            "graph_signature": signature(mol),
        }
        return reward, violations, penalties, detail

    def evaluate(self, pred):
        reward, _, _, detail = self.terms(pred)
        return reward, detail

    def gradient(self, pred, source_coordinates):
        reward, violations, penalties, detail = self.terms(pred)
        gradient, = torch.autograd.grad(reward, source_coordinates, retain_graph=True, allow_unused=False)
        component_gradients = {}
        utility, _ = self.affinity_utility(pred)
        pieces = {"task_utility": float(self.al_cfg.get("utility_weight", 1.0)) * utility}
        pieces.update({f"constraint:{key}": -value for key, value in penalties.items()})
        for key, value in pieces.items():
            component, = torch.autograd.grad(value, source_coordinates, retain_graph=True, allow_unused=False)
            component_gradients[key] = float(component.detach().norm())
        if not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite augmented-Lagrangian gradient")
        return gradient.detach(), {
            "norm": float(gradient.detach().norm()),
            "component_gradient_norms": component_gradients,
            "violations": {key: float(value.detach()) for key, value in violations.items()},
            "detail": detail,
        }

    def compare(self, candidate, base):
        candidate_value, candidate_detail = self.evaluate(candidate)
        base_value, base_detail = self.evaluate(base)
        gain = float((candidate_value - base_value).detach())
        threshold = float(self.al_cfg.get("minimum_same_time_gain", 0.0))
        return gain > threshold, gain, "augmented_reward_gain" if gain > threshold else "no_augmented_reward_gain", candidate_detail, base_detail

    def feasible(self, candidate, reference):
        failures = super().feasible(candidate, reference)
        try:
            candidate_mol = decode_endpoint(candidate, self.vocab)
            reference_mol = decode_endpoint(reference, self.vocab)
            halo = set(region_halo(candidate_mol, self.core))
            outside = [idx for idx in range(len(candidate["coords"])) if idx not in halo]
            limit = self.al_cfg.get("max_endpoint_outside_halo_delta_angstrom")
            if limit is not None and outside:
                shift = (candidate["coords"][outside] - reference["coords"][outside]).norm(dim=-1).max()
                if float(shift) > float(limit):
                    failures.append("outside_halo_endpoint_displacement")
            if signature(candidate_mol) != signature(reference_mol):
                policy = self.al_cfg.get("graph_change_policy", "allow_if_feasible")
                if policy == "reject":
                    failures.append("graph_change_rejected")
                elif policy != "allow_if_feasible":
                    failures.append("unknown_graph_change_policy")
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
        return failures

    def update_duals(self, detail: dict, step: int):
        violations = {cid: float(detail["constraints"][cid]["violation"]) for cid in self.controller.duals}
        return self.controller.update(violations, step)

    def controller_state(self):
        return self.controller.state_dict()

    def restore_controller(self, state):
        self.controller.load_state_dict(state)
