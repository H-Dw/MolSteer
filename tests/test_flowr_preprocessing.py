"""Exercise disk checkpoints with stochastic categorical draws and live self-conditioning."""
import ast
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from molsteer.preprocessing.checkpoints import (
    FORMAT, TIMES, assert_equal, atomic_save, cpu_copy, load_bundle, restore_rng, seed_all, snapshot_rng,
)
from molsteer.preprocessing.flowr_dataset import load_targets, target_seed, write_members
from molsteer.preprocessing.trajectory import Trajectory, verify_native, verify_suffix


class ToyIntegrator:
    use_sde_simulation = False
    hparams = {"steps": 100, "type_strategy": "uniform-sample"}

    def step(self, curr, pred, prior, times, dt):
        # Independent RNG families and batch-shaped categorical draws all affect the suffix.
        atoms = torch.multinomial(torch.ones(curr["coords"].shape[:2] + (4,)).flatten(0, 1), 1)
        atoms = atoms.reshape(curr["coords"].shape[:2]).float()
        noise = torch.rand_like(curr["coords"]) + np.random.random() + random.random()
        return {**curr, "coords": curr["coords"] + dt * (pred["coords"] + noise), "atomics": atoms}


class ToyModel:
    self_condition = True
    sc_charges = True
    sc_recycle_steps = 2
    inpainting_mode = False
    graph_inpainting = False
    _inpaint_self_condition = False
    coord_scale = 2.0

    def __init__(self):
        self.integrator = ToyIntegrator()
        self.builder = SimpleNamespace(undo_zero_com_batch=lambda coords, mask, com_list:
                                       coords + torch.stack(com_list).unsqueeze(1))
        self.gen = SimpleNamespace(get_pocket_encoding=lambda coords, names, **kwargs:
                                   (coords * 2, names.float()))

    def __call__(self, curr, pocket, times, *, cond_batch, **kwargs):
        return {**curr, "coords": curr["coords"] * .2 + cond_batch["coords"] * .3,
                "affinity": {"pic50": curr["coords"].sum(dim=(1, 2))}}

    def _get_predictions(self, out):
        return out, {k: out[k] for k in ("coords", "atomics", "bonds", "charges")}

    def _update_times(self, times, dt):
        return [t + dt for t in times]


def inputs():
    prior = {"coords": torch.randn(3, 4, 3), "atomics": torch.zeros(3, 4),
             "bonds": torch.zeros(3, 4, 4), "charges": torch.zeros(3, 4, 2), "mask": torch.ones(3, 4)}
    pocket = {"coords": torch.randn(3, 5, 3), "atom_names": torch.ones(3, 5),
              "charges": torch.ones(3, 5, 2), "bonds": torch.ones(3, 5, 5, 2),
              "res_names": torch.zeros(3, 5), "mask": torch.ones(3, 5),
              "complex": [SimpleNamespace(com=torch.tensor([i, 2., 3.])) for i in range(3)]}
    return prior, pocket


def record():
    seed_all(19)
    model = ToyModel()
    prior, pocket = inputs()
    initial_rng = snapshot_rng()
    trajectory = Trajectory(model, prior, pocket, 100, torch.device("cpu"))
    checkpoints, final = trajectory.run()
    bundle = {"format": FORMAT, "shared": trajectory.shared(), "checkpoints": checkpoints, "final": final}
    return model, trajectory, bundle, initial_rng


@pytest.mark.parametrize("key", TIMES)
def test_every_saved_time_resumes_exactly_from_disk(tmp_path, key):
    model, _, bundle, _ = record()
    atomic_save(bundle, tmp_path / "molecule.pt")
    saved = load_bundle(tmp_path / "molecule.pt")
    seed_all(9182)  # Model loading / other jobs can consume arbitrary RNG before restoration.
    verify_suffix(ToyModel(), saved, key, torch.device("cpu"))
    assert saved["checkpoints"]["1.00"]["step_index"] == 100
    assert all(torch.equal(t, torch.ones_like(t)) for t in saved["checkpoints"]["1.00"]["times"])


def test_upstream_generate_matches_checkpoint_loop():
    # Exercise the actual upstream loop without requiring its heavy import-time dependencies.
    source = Path(__file__).parents[1] / "flowr_root/flowr/models/fm_pocket.py"
    if not source.exists():
        pytest.skip("FLOWR checkout unavailable for native-loop parity test")
    module = ast.parse(source.read_text(encoding="utf-8"))
    native = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "_generate")
    namespace = {"torch": torch, "np": np, "raise_if_cancelled": lambda *a: None}
    exec(compile(ast.Module(body=[native], type_ignores=[]), str(source), "exec"), namespace)
    model, trajectory, bundle, rng = record()
    model._generate = namespace["_generate"].__get__(model)
    verify_native(model, trajectory, rng, bundle["final"])


def test_losing_self_condition_or_rng_is_detected():
    model, _, bundle, _ = record()
    broken = cpu_copy(bundle)
    broken["checkpoints"]["0.30"]["cond"]["coords"].zero_()
    with pytest.raises(ValueError, match="Exact replay mismatch"):
        verify_suffix(model, broken, "0.30", torch.device("cpu"))
    broken = cpu_copy(bundle)
    seed_all(3)
    broken["checkpoints"]["0.30"]["rng"] = snapshot_rng()
    with pytest.raises(ValueError, match="Exact replay mismatch"):
        verify_suffix(model, broken, "0.30", torch.device("cpu"))


def test_checkpoints_own_cpu_storage_and_rng_roundtrip():
    value = torch.ones(3)
    saved = cpu_copy({"x": value})
    value.zero_()
    assert saved["x"].sum() == 3
    seed_all(42)
    state = snapshot_rng()
    first = [random.random(), np.random.random(), torch.rand(5)]
    restore_rng(state)
    assert_equal(first, [random.random(), np.random.random(), torch.rand(5)])


def test_three_references_resolve_to_one_full_batch(tmp_path):
    _, _, bundle, _ = record()
    write_members(bundle, tmp_path, 0, 3)
    assert len(list(tmp_path.glob("molecule_*.pt"))) == 3
    assert len(list(tmp_path.glob("batch_*.pt"))) == 1
    for i, path in enumerate(sorted(tmp_path.glob("molecule_*.pt"))):
        member = load_bundle(path)
        assert member["batch_index"] == i
        assert tuple(member["checkpoints"]) == TIMES
        assert member["checkpoints"]["0.30"]["curr"]["coords"].shape[0] == 3


def test_capture_callback_does_not_change_trajectory(tmp_path):
    _, _, baseline, _ = record()
    seed_all(19)
    prior, pocket = inputs()
    trajectory = Trajectory(ToyModel(), prior, pocket, 100, torch.device("cpu"))
    _, final = trajectory.run(lambda key, state: atomic_save(state, tmp_path / f"{key}.pt"))
    for key in ("prediction", "world_prediction", "rng"):
        assert_equal(baseline["final"][key], final[key])


def test_split_validates_count_order_duplicates_and_paths(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("b.pdb", "b.sdf", "a.pdb", "a.sdf"):
        (data / name).write_text(name)
    split = tmp_path / "split.pt"
    pairs = [("b.pdb", "b.sdf"), ("a.pdb", "a.sdf")]
    torch.save({"train": [], "test": pairs}, split)
    manifest = load_targets(split, data, 2)
    assert [t["split_pair"] for t in manifest["targets"]] == [list(p) for p in pairs]
    with pytest.raises(ValueError, match="Expected 100"):
        load_targets(split, data)
    torch.save({"test": [pairs[0], pairs[0]]}, split)
    with pytest.raises(ValueError, match="Duplicate"):
        load_targets(split, data, 2)
    torch.save({"test": [("../escape.pdb", "a.sdf")]}, split)
    with pytest.raises(ValueError, match="Missing input"):
        load_targets(split, data, 1)
    first, second = manifest["targets"]
    assert target_seed(42, first) == target_seed(42, dict(first, test_index=55))
    assert target_seed(42, first) != target_seed(42, second)


def test_partial_trajectory_can_finish_without_original_final():
    model, _, bundle, _ = record()
    del bundle["final"]
    bundle["checkpoints"] = {"0.30": bundle["checkpoints"]["0.30"]}
    checkpoints, final = verify_suffix(model, bundle, "0.30", torch.device("cpu"))
    assert "1.00" in checkpoints and "prediction" in final


def test_unsupported_modes_grid_and_model_changes_fail_closed():
    model, _, bundle, _ = record()
    model.sc_recycle_steps = 1
    with pytest.raises(ValueError, match="model_signature"):
        verify_suffix(model, bundle, "0.30", torch.device("cpu"))
    prior, pocket = inputs()
    with pytest.raises(ValueError, match="multiple of 10"):
        Trajectory(ToyModel(), prior, pocket, 99, torch.device("cpu"))
    model.graph_inpainting = True
    with pytest.raises(ValueError, match="unmasked"):
        Trajectory(model, prior, pocket, 100, torch.device("cpu"))
