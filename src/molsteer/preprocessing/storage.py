"""Lossless batch storage: one tensor pool, small molecule references, atomic commits.

Decoded tensors can share storage and must be treated as read-only. Trajectory.restore
clones the selected live state before any sampling mutation. No dtype conversion,
quantization, batch slicing, or RNG reconstruction is performed here.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

import torch

from .checkpoints import FORMAT, TIMES, atomic_save, sha256, utc_now


BATCH_FORMAT = "molsteer.flowr.batch_trajectory.v2"
REFERENCE_FORMAT = "molsteer.flowr.molecule_reference.v2"
MEMBER_FIELDS = frozenset(("molecule_index", "batch_index", "molecule"))


def json_hash(value):
    def canonical(node):
        kind, data = node
        if kind == "dict":
            return [kind, [[k, canonical(v)] for k, v in sorted(data)]]
        if kind in ("list", "tuple"):
            return [kind, [canonical(v) for v in data]]
        return node
    return hashlib.sha256(json.dumps(canonical(value), ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def tensor_digest(tensor):
    if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("Checkpoint storage requires dense, unquantized CPU tensors")
    digest = hashlib.sha256()
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape), list(tensor.stride())]).encode())
    # Hash exact bytes, including signed zero and NaN payloads. memoryview avoids a
    # second full-size Python bytes allocation; only noncontiguous tensors need a copy.
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def pack(value):
    pool = {}
    logical_bytes = 0

    def encode(item):
        nonlocal logical_bytes
        if torch.is_tensor(item):
            key = tensor_digest(item)
            logical_bytes += item.numel() * item.element_size()
            if key not in pool:
                # Do not serialize an oversized backing storage belonging to a view.
                # Preserve strides: a contiguous conversion could change kernel choice.
                overlapping = any(size > 1 and stride == 0 for size, stride in zip(item.shape, item.stride()))
                if not overlapping and (item.storage_offset() or item.untyped_storage().nbytes() > item.numel() * item.element_size()):
                    saved = torch.empty_strided(item.shape, item.stride(), dtype=item.dtype)
                    saved.copy_(item)
                else:
                    saved = item.detach()
                pool[key] = saved
            return ["tensor", key]
        if isinstance(item, dict):
            if not all(isinstance(k, str) for k in item):
                raise TypeError("Checkpoint mappings require string keys")
            return ["dict", [[k, encode(v)] for k, v in item.items()]]
        if isinstance(item, list):
            return ["list", [encode(v) for v in item]]
        if isinstance(item, tuple):
            return ["tuple", [encode(v) for v in item]]
        if isinstance(item, bytes):
            return ["bytes", base64.b64encode(item).decode("ascii")]
        if type(item) in (str, int, float, bool, type(None)):
            return ["scalar", item]
        raise TypeError(f"Unsupported storage value: {type(item).__name__}")

    tree = encode(value)
    return {"tree": tree, "tensors": pool, "sha256": json_hash(tree),
            "statistics": {"logical_tensor_bytes": logical_bytes,
                           "unique_tensor_bytes": sum(t.untyped_storage().nbytes() for t in pool.values()),
                           "unique_tensors": len(pool)}}


def unpack(packed):
    if json_hash(packed["tree"]) != packed["sha256"]:
        raise ValueError("Batch metadata checksum mismatch")
    for key, tensor in packed["tensors"].items():
        if tensor_digest(tensor) != key:
            raise ValueError("Batch tensor checksum mismatch: " + key)

    def decode(node):
        kind, value = node
        if kind == "tensor":
            return packed["tensors"][value]
        if kind == "dict":
            return {k: decode(v) for k, v in value}
        if kind in ("tuple", "list"):
            values = [decode(v) for v in value]
            return tuple(values) if kind == "tuple" else values
        if kind == "bytes":
            return base64.b64decode(value, validate=True)
        if kind == "scalar":
            return value
        raise ValueError(f"Unknown batch node kind: {kind}")

    return decode(packed["tree"])


def assert_storage_equal(first, second):
    """Compare all values and exact tensor bytes/layout, not whole .pt file hashes."""
    if pack(first)["sha256"] != pack(second)["sha256"]:
        raise ValueError("Checkpoint fields/tensor bytes differ")


def split_member(bundle):
    common = {k: v for k, v in bundle.items() if k not in MEMBER_FIELDS}
    member = {k: bundle[k] for k in MEMBER_FIELDS if k in bundle}
    if not all(type(member.get(k)) is int for k in ("batch_index", "molecule_index")):
        raise ValueError("Missing or invalid molecule/batch index")
    return common, member


def batch_key(common):
    # Stable across incremental writes; distinguishes different priors, targets and runs.
    identity = {k: common.get(k) for k in ("shared", "target", "seed", "batch_id", "generator_args")}
    return pack(identity)["sha256"]


def validate_members(common, members):
    size = common["shared"]["batch_size"]
    indices = [m.get("batch_index") for m in members]
    mol_indices = [m.get("molecule_index") for m in members]
    if sorted(indices) != list(range(size)) or len(set(mol_indices)) != size:
        raise ValueError("Batch membership must cover every original slot exactly once")
    if any(type(i) is not int or i < 0 for i in mol_indices):
        raise ValueError("Invalid molecule index")


def write_batch(common, members, path):
    validate_members(common, members)
    packed = pack({"common": common, "members": members})
    key = batch_key(common)
    envelope = {"format": BATCH_FORMAT, "batch_key": key, "payload": packed}
    atomic_save(envelope, path)
    return key, packed["statistics"]


def read_batch(path):
    envelope = torch.load(path, map_location="cpu", weights_only=True, mmap=os.name != "nt")
    if envelope.get("format") != BATCH_FORMAT:
        raise ValueError("Unsupported batch file format")
    values = unpack(envelope["payload"])
    validate_members(values["common"], values["members"])
    if batch_key(values["common"]) != envelope["batch_key"]:
        raise ValueError("Batch identity checksum mismatch")
    return values, envelope["batch_key"]


def reference_for(batch_path, key, member):
    return {"format": REFERENCE_FORMAT, "batch_file": Path(batch_path).name, "batch_key": key,
            "batch_index": member["batch_index"], "molecule_index": member["molecule_index"]}


def resolve_reference(reference, path):
    # References are intentionally local, portable and cannot escape to another target.
    name = reference["batch_file"]
    if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError("Batch reference must be a sibling filename")
    parent = Path(path).resolve().parent
    batch_path = (parent / name).resolve()
    if batch_path.parent != parent:
        raise ValueError("Batch reference escapes target directory")
    values, key = read_batch(batch_path)
    if key != reference["batch_key"]:
        raise ValueError("Reference points to a different sampling batch")
    member = next((m for m in values["members"] if m["batch_index"] == reference["batch_index"]), None)
    if member is None or member["molecule_index"] != reference["molecule_index"]:
        raise ValueError("Molecule index does not match batch membership")
    return dict(values["common"], **member)


def write_members(bundle, output, offset, count):
    """A single atomic batch replacement commits all molecular metadata together.

    References are immutable identities, not hashes of a changing file. Content
    checksums live in the atomically replaced batch envelope. A crash between
    reference writes leaves existing references valid for the latest complete batch.
    """
    output = Path(output)
    common = {k: v for k, v in bundle.items() if k not in MEMBER_FIELDS and k != "decoded"}
    members = [{"molecule_index": offset + j, "batch_index": j} for j in range(count)]
    if "decoded" in bundle:
        for member, decoded in zip(members, bundle["decoded"], strict=True):
            member["molecule"] = decoded
    batch_path = output / f"batch_{bundle.get('batch_id', offset):03d}.pt"
    # Check all identities BEFORE replacing an existing shared file.
    key = batch_key(common)
    for member in members:
        path = output / f"molecule_{member['molecule_index']:03d}.pt"
        if path.exists():
            previous = torch.load(path, map_location="cpu", weights_only=True)
            if previous != reference_for(batch_path, key, member):
                raise ValueError(f"Refusing to overwrite unrelated molecule: {path}")
    write_batch(common, members, batch_path)
    for member in members:
        path = output / f"molecule_{member['molecule_index']:03d}.pt"
        if not path.exists():
            atomic_save(reference_for(batch_path, key, member), path)


def migrate_target(source, destination=None, *, audit_only=False):
    """Migrate a COMPLETE target. Same-directory mode atomically replaces members.

    All shared fields and each member's distinct fields are validated first.
    The committed shared file is reloaded and compared before any source file is
    replaced. A journal with original hashes makes interrupted migrations retryable.
    No source is unlinked and incomplete/failed targets are rejected.
    """
    from .checkpoints import load_bundle

    source = Path(source).resolve()
    destination = Path(destination).resolve() if destination is not None else source
    files = sorted(source.glob("molecule_*.pt"))
    if not files:
        raise ValueError("No molecule files found")
    source_hashes = {p.name: sha256(p) for p in files}
    # mmap-backed CPU trees keep original files on disk instead of copying 3 batches.
    bundles = [load_bundle(p) for p in files]
    physical_files = set(files)
    tensor_files = {}
    for path in files:
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=os.name != "nt")
        backing = path.parent / raw["batch_file"] if raw.get("format") == REFERENCE_FORMAT else path
        tensor_files[path] = backing
        physical_files.add(backing)
    del raw
    groups = {}
    indices = []
    for path, bundle in zip(files, bundles, strict=True):
        if bundle.get("status") not in ("completed", "completed_unverified") or set(bundle["checkpoints"]) != set(TIMES):
            raise ValueError(f"Incomplete or failed molecule: {path.name}")
        if "final" not in bundle:
            raise ValueError(f"Missing final state: {path.name}")
        common, member = split_member(bundle)
        indices.append(member["molecule_index"])
        group = groups.setdefault(common.get("batch_id", 0), [])
        group.append((path, common, member))
    expected = sum(len(group) for group in groups.values())
    layout = bundles[0]["batch_layout"]
    if sum(layout) != expected or sorted(indices) != list(range(expected)):
        raise ValueError("Missing/duplicate target members; refusing partial-target migration")
    if set(groups) != set(range(len(layout))):
        raise ValueError("Batch IDs disagree with the original batch layout")
    for bundle in bundles[1:]:
        for field in ("target", "seed", "batch_layout", "provenance", "generator_args"):
            assert_storage_equal(bundles[0].get(field), bundle.get(field))
    for batch_id, group in groups.items():
        common = group[0][1]
        validate_members(common, [item[2] for item in group])
        if len(group) != layout[batch_id]:
            raise ValueError("Missing members from the original batch")
        for path, _, member in group:
            if (path.name != f"molecule_{member['molecule_index']:03d}.pt" or
                    member["molecule_index"] != sum(layout[:batch_id]) + member["batch_index"]):
                raise ValueError("Molecule filename/index disagrees with original batch layout")
        for _, other, _ in group[1:]:
            assert_storage_equal(common, other)
    if audit_only:
        def tensor_bytes(path):
            with zipfile.ZipFile(path) as archive:
                return sum(i.file_size for i in archive.infolist() if re.search(r"/data/\d+$", i.filename))
        original_tensors = sum(tensor_bytes(p) for p in physical_files)
        shared_tensors = sum(tensor_bytes(tensor_files[group[0][0]]) for group in groups.values())
        pooled_tensors = sum(pack({"common": group[0][1], "members": [x[2] for x in group]})
                             ["statistics"]["unique_tensor_bytes"] for group in groups.values())
        return {"status": "audited", "source": str(source), "molecules": len(files),
                "original_file_bytes": sum(p.stat().st_size for p in physical_files),
                "original_tensor_bytes": original_tensors, "shared_tensor_bytes": shared_tensors,
                "batch_sharing_saved_tensor_bytes": original_tensors - shared_tensors,
                "pooled_tensor_bytes": pooled_tensors, "all_common_fields_equal": True,
                "original_sha256": source_hashes}
    destination.mkdir(parents=True, exist_ok=True)
    # Even in-place mode must not overwrite a pre-existing, unrelated batch file.
    for batch_id, group in groups.items():
        batch_path = destination / f"batch_{batch_id:03d}.pt"
        if batch_path.exists():
            saved, _ = read_batch(batch_path)
            assert_storage_equal({"common": group[0][1], "members": [x[2] for x in group]}, saved)
    if source != destination:
        for path, original in zip(files, bundles, strict=True):
            dest = destination / path.name
            if dest.exists():
                assert_storage_equal(original, load_bundle(dest))
    journal_path = destination / "storage_migration.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal["source"] != str(source):
            raise ValueError("Migration destination belongs to another source")
    else:
        journal = {"format": "molsteer.flowr.storage_migration.v2", "source": str(source),
                   "started_utc": utc_now(), "original_sha256": source_hashes,
                   "original_file_bytes": sum(p.stat().st_size for p in physical_files), "batches": []}
    journal["status"] = "validating"
    atomic_save(journal, journal_path, json_file=True)
    committed = []
    for batch_id, group in groups.items():
        common = group[0][1]
        members = [item[2] for item in group]
        batch_path = destination / f"batch_{batch_id:03d}.pt"
        key, statistics = write_batch(common, members, batch_path)
        restored, _ = read_batch(batch_path)
        for path, original, member in group:
            actual_member = next(m for m in restored["members"] if m["batch_index"] == member["batch_index"])
            assert_storage_equal(dict(original, **member), dict(restored["common"], **actual_member))
        committed.append({"batch": batch_path.name, "batch_key": key, **statistics,
                          "members": [p.name for p, _, _ in group], "all_fields_bitwise_equal": True})
    # Check for concurrent writers before committing ANY molecule references.
    if source_hashes != {p.name: sha256(p) for p in files}:
        raise ValueError("Source files changed during migration; references not committed")
    journal.update(status="batch_verified", batches=committed)
    atomic_save(journal, journal_path, json_file=True)
    for batch_id, group in groups.items():
        batch_path = destination / f"batch_{batch_id:03d}.pt"
        key = batch_key(group[0][1])
        for path, _, member in group:
            dest = destination / path.name
            if dest.exists() and source != destination:
                existing = load_bundle(dest)
                assert_storage_equal(existing, load_bundle(path))
            atomic_save(reference_for(batch_path, key, member), dest)
    # Verify all public load paths after publication, including per-molecule SDFs.
    for path, before in zip(files, bundles, strict=True):
        assert_storage_equal(before, load_bundle(destination / path.name))
    compact_paths = [destination / p.name for p in files] + [destination / f"batch_{i:03d}.pt" for i in groups]
    journal.update(status="completed", completed_utc=utc_now(), replay_status="not_run",
                   replay_note="Lossless state/RNG equality verified; this is not a GPU suffix execution.",
                   compact_file_bytes=sum(p.stat().st_size for p in compact_paths))
    journal["saved_bytes"] = journal["original_file_bytes"] - journal["compact_file_bytes"]
    atomic_save(journal, journal_path, json_file=True)
    return journal
