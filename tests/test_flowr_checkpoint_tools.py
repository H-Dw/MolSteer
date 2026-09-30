"""Real persisted-format tests for checkpoint inspection and safe recovery selection."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from molsteer.preprocessing.checkpoints import (
    TIMES, assert_equal, atomic_save, load_bundle, sha256, snapshot_rng,
)
from molsteer.preprocessing.flowr_dataset import target_seed
from molsteer.preprocessing.storage import migrate_target
from test_checkpoint_storage import legacy_target


def import_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = import_script("read_flowr_checkpoint")
completion = import_script("complete_flowr_100targets")


def fixture_target(root):
    target = {"target_id": "068_fixture", "test_index": 68, "split_pair": ["a.pdb", "a.sdf"]}
    config = {"seed": 20260930, "integration_steps": 100}
    directory = root / target["target_id"]
    files = legacy_target(directory)
    for file in files:
        b = load_bundle(file)
        b.update(target=target, seed=target_seed(config["seed"], target),
                 verification={"status": "passed", "resume_times": list(TIMES), "native_generate": "exact_match"})
        b["molecule"]["status"] = "decoded"
        atomic_save(b, file)
    return target, config, directory


@pytest.mark.parametrize("compact", [False, True])
def test_reader_exact_time_and_member_without_rng_or_file_mutation(tmp_path, compact):
    _, _, directory = fixture_target(tmp_path)
    if compact:
        migrate_target(directory, directory)
    originals = {p: sha256(p) for p in directory.glob("*.pt")}
    rng = snapshot_rng()
    for member in range(3):
        path = reader.resolve_file(root=tmp_path, target_index=68, molecule_index=member)
        for key in TIMES:
            bundle, state = reader.read_at(path, float(key))
            selected = reader.selected_state(bundle, state)
            assert selected["resume_supported"] is False
            assert selected["step_index"] == round(float(key) * 100)
            assert_equal(selected["curr"]["coords"], state["curr"]["coords"][member])
            expected = state["curr"]["coords"][member] * 2. + bundle["shared"]["com"][member]
            assert_equal(selected["coords_world"], expected)
            # Editing analysis copies cannot corrupt shared mapped tensors.
            saved = state["curr"]["coords"][member].clone()
            selected["curr"]["coords"].zero_()
            assert_equal(saved, state["curr"]["coords"][member])
    assert_equal(rng, snapshot_rng())
    assert originals == {p: sha256(p) for p in originals}


@pytest.mark.parametrize("value", [0.299, 0.301, 0.355, 0.0, 1.01, float("nan"), float("inf")])
def test_reader_rejects_rounding_to_a_different_saved_time(value):
    with pytest.raises(ValueError):
        reader.time_key(value)


def test_reader_json_export_mask_and_existing_output_protection(tmp_path):
    _, _, directory = fixture_target(tmp_path)
    path = directory / "molecule_002.pt"
    bundle = load_bundle(path)
    bundle["checkpoints"]["0.50"]["curr"]["mask"][2, 3] = 0
    atomic_save(bundle, path)
    output, tensors = tmp_path / "info.json", tmp_path / "selected.pt"
    assert reader.main(["--file", str(path), "--t", "0.5", "--json", str(output),
                        "--export-selected", str(tensors)]) == 0
    report = json.loads(output.read_text())
    assert report["active_atom_count"] == 3
    assert report["rng"]["families"] == ["python", "numpy", "torch", "cuda"]
    selected = torch.load(tensors, weights_only=True)
    assert not selected["coords_world"][3].any()
    assert selected["active_coords_world"].shape == (3, 3)
    before = sha256(output)
    with pytest.raises(SystemExit):
        reader.main(["--file", str(path), "--t", "0.5", "--json", str(output)])
    assert sha256(output) == before


def test_recovery_preserves_completed_and_selects_only_failed(tmp_path):
    target, config, directory = fixture_target(tmp_path)
    migrate_target(directory, directory)
    failed = dict(target, target_id="069_pending", test_index=69)
    run = {"config": config, "targets": [{"target_id": target["target_id"], "status": "completed"},
                                          {"target_id": failed["target_id"], "status": "failed"}]}
    completed, pending = completion.classify(tmp_path, run, {"targets": [target, failed]})
    assert len(completed) == 1 and pending == [failed]
    assert len(completed[0]["sha256"]) == 4
    # A completed entry with damaged data must raise, never silently enter pending.
    (directory / "molecule_001.pt").unlink()
    with pytest.raises(ValueError, match="three molecule"):
        completion.classify(tmp_path, run, {"targets": [target, failed]})


def test_reader_rejects_ambiguous_and_out_of_range_selection(tmp_path):
    (tmp_path / "068_a").mkdir()
    (tmp_path / "068_b").mkdir()
    with pytest.raises(ValueError, match="one target"):
        reader.resolve_file(root=tmp_path, target_index=68, molecule_index=0)
    with pytest.raises(ValueError, match="0..2"):
        reader.resolve_file(root=tmp_path, target_index=68, molecule_index=3)
    with pytest.raises(ValueError, match="OR"):
        reader.resolve_file(file="x.pt", root=tmp_path)
