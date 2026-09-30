"""Native rigid-pocket FLOWR Euler rollout, with explicit continuation boundaries."""
from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from .checkpoints import TIMES, assert_equal, cpu_copy, restore_rng, snapshot_rng, tree_map, utc_now


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_model(model):
    for name in ("inpainting_mode", "graph_inpainting", "_inpaint_self_condition"):
        if getattr(model, name, False):
            raise ValueError(f"This preprocessing recipe requires unmasked rigid-pocket generation: {name}")
    if getattr(model.integrator, "use_sde_simulation", False):
        raise ValueError("SDE generation requires a separate continuation recipe")


def model_signature(model):
    return cpu_copy({"self_condition": bool(model.self_condition), "sc_charges": bool(model.sc_charges),
                     "sc_recycle_steps": int(getattr(model, "sc_recycle_steps", 1)),
                     "coord_scale": model.coord_scale, "integrator": model.integrator.hparams})


class Trajectory:
    """Owns the whole original batch. Never slice it before categorical sampling."""

    def __init__(self, model, prior, pocket, steps, device):
        validate_model(model)
        if steps < 10 or steps % 10:
            raise ValueError("integration_steps must be a positive multiple of 10")
        self.model, self.device = model, device
        self.prior = tree_map(lambda t: t.to(device), prior)
        # Complex instances are only used to recover the coordinate origin.
        self.com = torch.stack([torch.as_tensor(s.com) for s in pocket["complex"]]).cpu()
        self.pocket = tree_map(lambda t: t.to(device), {k: v for k, v in pocket.items() if k != "complex"})
        self.grid = torch.linspace(0, 1, steps + 1)
        self.curr = tree_map(lambda t: t.clone(), self.prior)
        self.cond = {"coords": self.prior["coords"], "atomics": torch.zeros_like(self.prior["atomics"]),
                     "bonds": torch.zeros_like(self.prior["bonds"])}
        if model.sc_charges:
            self.cond["charges"] = torch.zeros_like(self.prior["charges"])
        self.times = [torch.zeros(self.prior["coords"].shape[0], device=device) for _ in range(2)]
        self.times.append(torch.zeros(self.pocket["coords"].shape[0], device=device))
        with torch.no_grad():
            self.equis, self.invs = model.gen.get_pocket_encoding(
                self.pocket["coords"], self.pocket["atom_names"],
                pocket_atom_charges=self.pocket["charges"].argmax(-1),
                pocket_bond_types=self.pocket["bonds"].argmax(-1),
                pocket_res_types=self.pocket["res_names"], pocket_atom_mask=self.pocket["mask"])
        self.step_index = 0

    @classmethod
    def restore(cls, model, shared, state, device):
        validate_model(model)
        assert_equal(shared["model_signature"], model_signature(model), "model_signature")
        self = cls.__new__(cls)
        self.model, self.device = model, device
        for key in ("prior", "pocket", "equis", "invs"):
            setattr(self, key, tree_map(lambda t: t.to(device).clone(), shared[key]))
        self.com, self.grid = shared["com"].clone(), shared["grid"].clone()
        for key in ("curr", "cond", "times"):
            setattr(self, key, tree_map(lambda t: t.to(device).clone(), state[key]))
        self.step_index = state["step_index"]
        if not 0 <= self.step_index < len(self.grid):
            raise ValueError("Invalid integration step in checkpoint")
        # Restore LAST: loading the model and moving tensors must precede RNG restoration.
        restore_rng(state["rng"])
        return self

    def shared(self):
        return cpu_copy({"prior": self.prior, "pocket": self.pocket, "com": self.com,
                         "equis": self.equis, "invs": self.invs, "grid": self.grid,
                         "model_signature": model_signature(self.model),
                         "batch_size": self.curr["coords"].shape[0]})

    def state(self):
        sync(self.device)
        return cpu_copy({"curr": self.curr, "cond": self.cond, "times": self.times,
                         "step_index": self.step_index, "rng": snapshot_rng()})

    @torch.no_grad()
    def predict(self, times=None, cond=None):
        out = self.model(self.curr, self.pocket, self.times if times is None else times,
                         training=False, cond_batch=(self.cond if cond is None else cond)
                         if self.model.self_condition else None,
                         pocket_equis=self.equis, pocket_invs=self.invs)
        return self.model._get_predictions(out)

    @torch.no_grad()
    def step(self):
        predicted, cond = self.predict()
        for _ in range(getattr(self.model, "sc_recycle_steps", 1) - 1):
            if self.model.self_condition:
                predicted, cond = self.predict(cond=cond)
        dt = self.grid[self.step_index + 1] - self.grid[self.step_index]
        self.curr = self.model.integrator.step(self.curr, predicted, self.prior, self.times, dt)
        self.cond = cond
        self.times = self.model._update_times(self.times, dt)
        self.step_index += 1

    def finish(self):
        # t=1.00 stores the live integrator state BEFORE the native final head.
        # The head's 1e-4 offset never replaces the saved integration time.
        prediction, _ = self.predict(times=self.model._update_times(self.times, -1e-4))
        prediction = cpu_copy(prediction)
        world = cpu_copy(prediction)
        world["coords"] = world["coords"] * self.model.coord_scale
        world["coords"] = self.model.builder.undo_zero_com_batch(
            world["coords"], world["mask"], com_list=list(self.com))
        return {"prediction": prediction, "world_prediction": world, "rng": snapshot_rng()}

    def run(self, on_checkpoint=None):
        checkpoints = {}
        steps = len(self.grid) - 1
        stage_steps = {steps * i // 10: f"{i / 10:.2f}" for i in range(3, 11)}
        inference_seconds = 0.0
        while self.step_index < steps:
            sync(self.device)
            started = time.perf_counter()
            self.step()
            sync(self.device)
            inference_seconds += time.perf_counter() - started
            if self.step_index in stage_steps:
                key = stage_steps[self.step_index]
                state = self.state()
                state["capture"] = {"utc": utc_now(), "inference_seconds_this_segment": inference_seconds,
                                    "status": "awaiting_final_head" if key == "1.00" else "in_progress",
                                    "boundary": "after_native_step_before_next_forward",
                                    "requested_t": float(key), "grid_t": float(self.grid[self.step_index])}
                checkpoints[key] = state
                if on_checkpoint:
                    on_checkpoint(key, state)
        sync(self.device)
        started = time.perf_counter()
        final = self.finish()
        sync(self.device)
        final["head_seconds"] = time.perf_counter() - started
        final["completed_utc"] = utc_now()
        return checkpoints, final


def state_fields(state):
    return {key: state[key] for key in ("curr", "cond", "times", "step_index", "rng")}


def verify_suffix(model, bundle, key, device):
    """Compare every remaining saved boundary, final structure/affinity and all RNGs."""
    trajectory = Trajectory.restore(model, bundle["shared"], bundle["checkpoints"][key], device)
    checkpoints, final = trajectory.run()
    for time_key, state in checkpoints.items():
        if time_key in bundle["checkpoints"]:
            assert_equal(state_fields(bundle["checkpoints"][time_key]), state_fields(state), time_key)
    if "final" in bundle:
        for field in ("prediction", "world_prediction", "rng"):
            assert_equal(bundle["final"][field], final[field], "final." + field)
    return checkpoints, final


def verify_native(model, trajectory, initial_rng, final):
    """Independently execute upstream _generate on exactly the same prior and RNG."""
    pocket = tree_map(lambda t: t.clone(), trajectory.pocket)
    pocket["complex"] = [SimpleNamespace(com=c) for c in trajectory.com]
    times = [torch.zeros_like(t) for t in trajectory.times]
    restore_rng(initial_rng)
    with torch.no_grad():
        expected = model._generate(tree_map(lambda t: t.clone(), trajectory.prior), pocket,
                                   steps=len(trajectory.grid) - 1, times=times, strategy="linear",
                                   solver="euler", corr_iters=0, save_traj=False,
                                   final_corr_pred=True, final_inpaint=False, apply_guidance=False)
    assert_equal(cpu_copy(expected), final["world_prediction"], "upstream._generate")
    assert_equal(snapshot_rng(), final["rng"], "upstream.rng")
