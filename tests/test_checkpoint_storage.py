import json
from pathlib import Path

import pytest
import torch

from molsteer.preprocessing import storage
from molsteer.preprocessing.checkpoints import (
    TIMES, assert_equal, atomic_save, load_bundle, seed_all, sha256, validate_sources,
)
from molsteer.preprocessing.trajectory import Trajectory, verify_suffix
from test_flowr_preprocessing import ToyModel, inputs, record


def legacy_target(path):
    _, _, bundle, _ = record()
    bundle.update(status="completed", batch_layout=[3], batch_id=0,
                  target={"target_id": "fixture"}, seed=19,
                  provenance={"model_sha256": "fixture"}, generator_args={"seed": 19})
    path.mkdir(parents=True)
    files = []
    for i in range(3):
        # Members deliberately differ: deleting two as identical copies loses data.
        member = dict(bundle, molecule_index=i, batch_index=i,
                      molecule={"sdf": f"unique molecule {i}", "sanitized": i != 2})
        file = path / f"molecule_{i:03d}.pt"
        atomic_save(member, file)
        files.append(file)
    return files


def test_lossless_pool_keeps_bits_dtypes_strides_and_rng():
    noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    value = {"a": noncontiguous, "b": noncontiguous.clone(),
             "nan": torch.tensor([float("nan")]), "plus_zero": torch.tensor([0.]),
             "minus_zero": torch.tensor([-0.]), "bytes": b"\x00\xff", "tuple": (1, True, None),
             "view": torch.arange(1000)[100:110], "expanded": torch.ones(1).expand(4)}
    packed = storage.pack(value)
    restored = storage.unpack(packed)
    storage.assert_storage_equal(value, restored)
    assert restored["a"].data_ptr() == restored["b"].data_ptr()
    assert restored["a"].stride() == value["a"].stride()
    assert restored["plus_zero"].data_ptr() != restored["minus_zero"].data_ptr()
    assert restored["view"].untyped_storage().nbytes() == 10 * 8


def test_saved_pool_and_member_metadata_roundtrip_and_replay(tmp_path):
    files = legacy_target(tmp_path / "v1")
    original_hashes = [sha256(p) for p in files]
    report = storage.migrate_target(tmp_path / "v1", tmp_path / "v2")
    assert report["status"] == "completed"
    assert report["compact_file_bytes"] < report["original_file_bytes"] * .6
    assert [sha256(p) for p in files] == original_hashes
    for file in files:
        original = load_bundle(file)
        converted = load_bundle(tmp_path / "v2" / file.name)
        storage.assert_storage_equal(original, converted)
        for key in TIMES:
            verify_suffix(ToyModel(), converted, key, torch.device("cpu"))
        # Restoration clones tensor-pool aliases before advancing.
        storage.assert_storage_equal(original, converted)
    assert report["replay_status"] == "not_run"  # migration does not invent a GPU check


def test_in_place_migration_is_lossless_and_retryable(tmp_path):
    files = legacy_target(tmp_path / "v1")
    originals = [load_bundle(p) for p in files]
    report = storage.migrate_target(tmp_path / "v1", tmp_path / "v1")
    for file, before in zip(files, originals):
        assert torch.load(file, weights_only=True)["format"] == storage.REFERENCE_FORMAT
        storage.assert_storage_equal(before, load_bundle(file))
    repeat = storage.migrate_target(tmp_path / "v1", tmp_path / "v1")
    assert repeat["original_sha256"] == report["original_sha256"]
    assert repeat["original_file_bytes"] == report["original_file_bytes"]


def test_different_shared_data_is_not_deduplicated_or_deleted(tmp_path):
    files = legacy_target(tmp_path / "v1")
    changed = load_bundle(files[1])
    changed["checkpoints"]["0.30"]["cond"]["coords"].add_(.1)
    atomic_save(changed, files[1])
    hashes = [sha256(p) for p in files]
    with pytest.raises(ValueError, match="differ"):
        storage.migrate_target(tmp_path / "v1", tmp_path / "v2")
    assert [sha256(p) for p in files] == hashes
    assert not (tmp_path / "v2").exists()


def test_partial_target_and_duplicate_index_are_rejected(tmp_path):
    files = legacy_target(tmp_path / "v1")
    changed = load_bundle(files[1])
    changed["batch_index"] = 0
    atomic_save(changed, files[1])
    with pytest.raises(ValueError, match="slot"):
        storage.migrate_target(tmp_path / "v1", tmp_path / "v2")
    changed["batch_index"] = 1
    changed["status"] = "failed"
    atomic_save(changed, files[1])
    with pytest.raises(ValueError, match="Incomplete"):
        storage.migrate_target(tmp_path / "v1", tmp_path / "v2")


def test_interrupted_reference_publication_can_resume(tmp_path, monkeypatch):
    files = legacy_target(tmp_path / "v1")
    originals = [load_bundle(p) for p in files]
    real_save = storage.atomic_save

    def interrupted(value, path, **kwargs):
        if Path(path).name == "molecule_001.pt" and value.get("format") == storage.REFERENCE_FORMAT:
            raise OSError("simulated disk failure")
        return real_save(value, path, **kwargs)

    monkeypatch.setattr(storage, "atomic_save", interrupted)
    with pytest.raises(OSError, match="simulated"):
        storage.migrate_target(tmp_path / "v1", tmp_path / "v1")
    for file, before in zip(files, originals):
        storage.assert_storage_equal(before, load_bundle(file))
    monkeypatch.setattr(storage, "atomic_save", real_save)
    storage.migrate_target(tmp_path / "v1", tmp_path / "v1")
    for file, before in zip(files, originals):
        storage.assert_storage_equal(before, load_bundle(file))


def test_tensor_corruption_wrong_member_and_path_traversal_fail(tmp_path):
    files = legacy_target(tmp_path / "v1")
    storage.migrate_target(tmp_path / "v1", tmp_path / "v2")
    member_path = tmp_path / "v2" / files[0].name
    ref = torch.load(member_path, weights_only=True)
    atomic_save(dict(ref, batch_index=1), member_path)
    with pytest.raises(ValueError, match="Molecule index"):
        load_bundle(member_path)
    atomic_save(dict(ref, batch_file="../batch_000.pt"), member_path)
    with pytest.raises(ValueError, match="sibling"):
        load_bundle(member_path)
    atomic_save(ref, member_path)
    batch_path = tmp_path / "v2" / "batch_000.pt"
    batch = torch.load(batch_path, weights_only=True)
    tensor = next(t for t in batch["payload"]["tensors"].values() if t.is_floating_point() and t.numel())
    tensor.add_(1)
    atomic_save(batch, batch_path)
    with pytest.raises(ValueError, match="tensor checksum"):
        load_bundle(member_path)


def test_incremental_capture_has_stable_refs_and_one_shared_file(tmp_path):
    _, _, bundle, _ = record()
    original_stages = bundle["checkpoints"]
    bundle["checkpoints"] = {"0.30": original_stages["0.30"]}
    storage.write_members(bundle, tmp_path, 0, 3)
    ref_hashes = [sha256(p) for p in sorted(tmp_path.glob("molecule_*.pt"))]
    bundle["checkpoints"] = original_stages
    bundle["decoded"] = [{"sdf": str(i)} for i in range(3)]
    storage.write_members(bundle, tmp_path, 0, 3)
    assert ref_hashes == [sha256(p) for p in sorted(tmp_path.glob("molecule_*.pt"))]
    for i in range(3):
        loaded = load_bundle(tmp_path / f"molecule_{i:03d}.pt")
        assert tuple(loaded["checkpoints"]) == TIMES
        assert loaded["molecule"]["sdf"] == str(i)


def test_audit_does_not_modify_files(tmp_path):
    files = legacy_target(tmp_path / "v1")
    hashes = [sha256(p) for p in files]
    result = storage.migrate_target(tmp_path / "v1", audit_only=True)
    assert result["batch_sharing_saved_tensor_bytes"] > 0
    assert [sha256(p) for p in files] == hashes
    assert len(list((tmp_path / "v1").iterdir())) == 3
    storage.migrate_target(tmp_path / "v1", tmp_path / "v2")
    compact_audit = storage.migrate_target(tmp_path / "v2", audit_only=True)
    assert compact_audit["batch_sharing_saved_tensor_bytes"] == 0
    assert compact_audit["original_file_bytes"] == sum(p.stat().st_size for p in (tmp_path / "v2").glob("*.pt"))


def test_shared_capture_is_rng_neutral_and_preserves_native_output(tmp_path):
    _, _, baseline, _ = record()
    seed_all(19)
    prior, pocket = inputs()
    trajectory = Trajectory(ToyModel(), prior, pocket, 100, torch.device("cpu"))
    bundle = {"format": baseline["format"], "shared": trajectory.shared(), "checkpoints": {}}

    def capture(key, state):
        bundle["checkpoints"][key] = state
        storage.write_members(bundle, tmp_path, 0, 3)

    _, final = trajectory.run(capture)
    for field in ("prediction", "world_prediction", "rng"):
        assert_equal(baseline["final"][field], final[field])


def test_failed_shared_file_update_keeps_old_references_readable(tmp_path, monkeypatch):
    _, _, bundle, _ = record()
    final_stages = bundle["checkpoints"]
    bundle["checkpoints"] = {"0.30": final_stages["0.30"]}
    storage.write_members(bundle, tmp_path, 0, 3)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "atomic_save", fail)
    bundle["checkpoints"] = final_stages
    with pytest.raises(OSError, match="disk full"):
        storage.write_members(bundle, tmp_path, 0, 3)
    for i in range(3):
        assert list(load_bundle(tmp_path / f"molecule_{i:03d}.pt")["checkpoints"]) == ["0.30"]


def test_source_compatibility_is_specific_to_audited_storage_upgrade():
    old = {"molsteer/preprocessing/checkpoints.py": "8a16ef453ce867151dc0a59c4576845633bf09246e2c3eae2b3de90c5c0a24a7",
           "molsteer/preprocessing/flowr_dataset.py": "a3efd591f2b6cef9aa128a7f893ba3e4e59b987e84d93afe5a920ef02ac3d16b",
           "molsteer/preprocessing/trajectory.py": "same", "flowr/fm_pocket.py": "same"}
    new = dict(old)
    new.update({"molsteer/preprocessing/checkpoints.py": "updated", "molsteer/preprocessing/flowr_dataset.py": "updated",
                "molsteer/preprocessing/storage.py": "new", "molsteer/preprocessing/migrate.py": "new"})
    assert validate_sources(old, new) == "audited_v1_storage_upgrade"
    new["flowr/fm_pocket.py"] = "changed_numerics"
    with pytest.raises(ValueError, match="source_hashes"):
        validate_sources(old, new)


def test_unrelated_existing_batch_is_not_overwritten(tmp_path):
    files = legacy_target(tmp_path / "v1")
    common, _ = storage.split_member(load_bundle(files[0]))
    common["seed"] = 999
    members = [storage.split_member(load_bundle(p))[1] for p in files]
    batch_path = tmp_path / "v1" / "batch_000.pt"
    storage.write_batch(common, members, batch_path)
    before = sha256(batch_path)
    with pytest.raises(ValueError, match="differ"):
        storage.migrate_target(tmp_path / "v1", tmp_path / "v1")
    assert sha256(batch_path) == before
