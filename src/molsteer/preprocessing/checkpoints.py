"""Tensor-only checkpoint primitives; importing this module does not import FLOWR."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


TIMES = tuple(f"{i / 10:.2f}" for i in range(3, 11))
FORMAT = "molsteer.flowr.molecule_trajectory.v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def tree_map(fn, value):
    if torch.is_tensor(value):
        return fn(value)
    if hasattr(value, "items"):
        return {k: tree_map(fn, v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_map(fn, v) for v in value)
    if isinstance(value, list):
        return [tree_map(fn, v) for v in value]
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def cpu_copy(value):
    # clone is necessary even on CPU: later in-place updates must not alter a snapshot.
    return tree_map(lambda t: t.detach().cpu().clone(), value)


def snapshot_rng():
    state = np.random.get_state()
    return cpu_copy({"python": random.getstate(),
                     "numpy": [state[0], state[1].tolist(), *state[2:]],
                     "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})


def restore_rng(state):
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Visible CUDA device count differs from the saved RNG layout")
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(model_root, stage_runner):
    paths = sorted((Path(model_root) / "flowr").rglob("*.py"))
    if not paths:
        raise ValueError("FLOWR source tree is empty")
    result = {"flowr/" + p.relative_to(Path(model_root) / "flowr").as_posix(): sha256(p) for p in paths}
    result["stage_runner.py"] = sha256(stage_runner)
    for p in sorted(Path(__file__).parent.glob("*.py")):
        result["molsteer/preprocessing/" + p.name] = sha256(p)
    return result


def configure_numerics(device):
    # Set before the first CUDA context / BLAS handle is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def environment(device):
    gpu = []
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        gpu.append({"index": i, "name": prop.name, "capability": [prop.major, prop.minor],
                    "total_memory": prop.total_memory, "uuid": str(getattr(prop, "uuid", "unavailable"))})
    # Whitelist runtime controls; never save arbitrary environment variables or credentials.
    controls = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "NVIDIA_TF32_OVERRIDE", "FLOWR_DEVICE",
                "HIPBLAS_WORKSPACE_CONFIG", "HSA_OVERRIDE_GFX_VERSION")
    driver_files = ("/proc/driver/nvidia/version", "/sys/module/amdgpu/version",
                    "/sys/module/hydcu/version", "/sys/module/hycu/version")
    drivers = {path: Path(path).read_text().strip() for path in driver_files if Path(path).is_file()}
    packages = {p.metadata["Name"].lower(): p.version for p in importlib.metadata.distributions()
                if p.metadata.get("Name")}
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "torch": str(torch.__version__),
            "torch_build": torch.__config__.show(), "cuda": torch.version.cuda,
            "hip": torch.version.hip, "cudnn": torch.backends.cudnn.version(),
            "device": str(device), "gpu": gpu, "driver_versions": drivers,
            "packages": dict(sorted(packages.items())),
            "controls": {key: os.environ.get(key) for key in controls},
            "numerics": {"deterministic": torch.are_deterministic_algorithms_enabled(),
                         "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                         "precision": torch.get_float32_matmul_precision(),
                         "default_dtype": str(torch.get_default_dtype()),
                         "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                         "cudnn_benchmark": torch.backends.cudnn.benchmark,
                         "cudnn_deterministic": torch.backends.cudnn.deterministic,
                         "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                         "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                         "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
                         "math_sdp": torch.backends.cuda.math_sdp_enabled(),
                         "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
                         "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                         "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}}


def assert_equal(first, second, path="state"):
    """Exact shape/dtype/value equality, recursively, with the failing field named."""
    if torch.is_tensor(first):
        if not torch.is_tensor(second) or first.dtype != second.dtype or first.shape != second.shape:
            raise ValueError(f"Exact replay mismatch: {path} (tensor layout)")
        if not torch.equal(first.detach().cpu(), second.detach().cpu()):
            raise ValueError(f"Exact replay mismatch: {path}")
    elif isinstance(first, dict):
        if not isinstance(second, dict) or first.keys() != second.keys():
            raise ValueError(f"Exact replay mismatch: {path} (keys)")
        for key in first:
            assert_equal(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            raise ValueError(f"Exact replay mismatch: {path} (sequence)")
        for i, (a, b) in enumerate(zip(first, second)):
            assert_equal(a, b, f"{path}[{i}]")
    elif type(first) is not type(second) or first != second:
        raise ValueError(f"Exact replay mismatch: {path}")


def atomic_save(value, path, *, json_file=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            if json_file:
                stream.write(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))
            else:
                torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_bundle(path):
    # Linux uses memory mapping for large read-only histories. Windows cannot replace
    # files with outstanding mappings during an in-place storage migration.
    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=os.name != "nt")
    from .storage import REFERENCE_FORMAT, resolve_reference
    if bundle.get("format") == REFERENCE_FORMAT:
        return resolve_reference(bundle, path)
    if bundle.get("format") != FORMAT:
        raise ValueError("Unsupported molecule checkpoint format")
    return bundle


def validate_sources(saved, current):
    """Accept only the audited v1 storage upgrade, preserving all numerical code checks."""
    if saved == current:
        return "exact_source_match"
    legacy = {
        "molsteer/preprocessing/checkpoints.py": "8a16ef453ce867151dc0a59c4576845633bf09246e2c3eae2b3de90c5c0a24a7",
        "molsteer/preprocessing/flowr_dataset.py": "a3efd591f2b6cef9aa128a7f893ba3e4e59b987e84d93afe5a920ef02ac3d16b",
    }
    new_files = {"molsteer/preprocessing/storage.py", "molsteer/preprocessing/migrate.py"}
    # FLOWR, stage runner, trajectory.py, RNG functions and numerical settings remain
    # unchanged by this migration. Do not allow arbitrary saved source hashes.
    if (all(saved.get(k) == v for k, v in legacy.items()) and
            set(current) == set(saved) | new_files and
            all(current.get(k) == v for k, v in saved.items() if k not in legacy)):
        return "audited_v1_storage_upgrade"
    assert_equal(saved, current, "source_hashes")
