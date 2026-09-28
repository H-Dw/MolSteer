#!/usr/bin/env python3
"""Stage-aware CrossDocked target-conditioned FLOWR.root generation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import numpy as np
from rdkit import Chem

import flowr.gen.utils as util
from flowr.gen.generate_from_pdb import get_args, get_dataset
from flowr.scriptutil import load_model
from flowr.util.device import resolve_device

EXPECTED_STAGES = (0.25, 0.50, 0.75)


def snapshot_rng():
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                python=random.getstate(), numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]])


def write_runtime(target_dir, ligand_offset, stage_name, step_index, curr, cond, prior,
                  times, pocket, equis, invs, args, model):
    """Capture the live batch after the stage head, without drawing random values.

    Each ligand receives a complete batch checkpoint: slicing a batch cannot
    reproduce the random draws consumed by the original categorical sampler.
    """
    metadata = args.runtime_metadata
    com = torch.stack([torch.as_tensor(system.com) for system in pocket['complex']])
    checkpoint = dict(format='flowr_root_live_runtime', schema=1, curr=curr, cond=cond,
        prior=prior, times=times, step_index=step_index, rng=snapshot_rng(),
        pocket_tensors={k:v for k,v in pocket.items() if k!='complex'},
        pocket_com=com, pocket_equis=equis, pocket_invs=invs,
        grid=torch.linspace(0,1,args.integration_steps+1),
        source_hash=metadata['stage_runner_sha256'], model_checkpoint=metadata['checkpoint'],
        model_checkpoint_sha256=metadata['checkpoint_sha256'], precision=metadata['precision'],
        cuda_device_index=metadata['cuda_device_index'],
        config=dict(receptor=str(Path(args.pdb_file).resolve()),
            reference_ligand=str(Path(args.ligand_file).resolve()),target_id=target_dir.name,
            integration_steps=args.integration_steps,seed=args.seed,
            categorical_strategy=args.categorical_strategy,ode_sampling_strategy=args.ode_sampling_strategy),
        guidance_state={}, self_condition_enabled=bool(model.self_condition),
        resume_fidelity='Live original trajectory; no replay or state replacement',
        capture_boundary='After extra reporting head and ligand serialization; before next native step')
    checkpoint=cpu_copy(checkpoint)
    directory=target_dir/'runtime';directory.mkdir(exist_ok=True)
    canonical=directory/f'batch_{ligand_offset:03d}_{stage_name}.pt'
    torch.save(checkpoint,canonical)
    sha=hashlib.sha256(canonical.read_bytes()).hexdigest()
    for j in range(curr['coords'].shape[0]):
        dest=target_dir/f'ligand_{ligand_offset+j:03d}'/stage_name
        path=dest/'runtime.pt'
        try:os.link(canonical,path)
        except OSError:shutil.copy2(canonical,path)
        (dest/'runtime.json').write_text(json.dumps(dict(format=checkpoint['format'],
            batch_index=j,batch_size=int(curr['coords'].shape[0]),step_index=step_index,
            sha256=sha,capture_boundary=checkpoint['capture_boundary'],
            self_condition_enabled=checkpoint['self_condition_enabled'],
            resume_fidelity=checkpoint['resume_fidelity']),indent=2))


def cpu_copy(value: Any):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if hasattr(value, "items"):
        try:
            return {k: cpu_copy(v) for k, v in value.items()}
        except Exception:
            pass
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    # Complex objects are not needed in serialized stage state.
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def tensor_fields(value: dict[str, Any]):
    out = {}
    for k, v in value.items():
        if torch.is_tensor(v) or isinstance(v, (dict, list, tuple, str, int, float, bool)):
            out[k] = cpu_copy(v)
    return out


def slice_batch(value: Any, index: int, batch_size: int):
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value
    if isinstance(value, dict):
        return {k: slice_batch(v, index, batch_size) for k, v in value.items()}
    if hasattr(value, "items"):
        try:
            return {k: slice_batch(v, index, batch_size) for k, v in value.items()}
        except Exception:
            pass
    if isinstance(value, list):
        return [slice_batch(v, index, batch_size) for v in value]
    if isinstance(value, tuple):
        return tuple(slice_batch(v, index, batch_size) for v in value)
    return value


def scalarize(value: Any):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.reshape(-1).tolist()
    if isinstance(value, dict):
        return {str(k): scalarize(v) for k, v in value.items()}
    if hasattr(value, "items"):
        try:
            return {str(k): scalarize(v) for k, v in value.items()}
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [scalarize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def cuda_sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def mem_stats(device):
    if not str(device).startswith("cuda"):
        return {}
    d = torch.device(device)
    return {
        "device": str(d),
        "allocated_bytes": torch.cuda.memory_allocated(d),
        "reserved_bytes": torch.cuda.memory_reserved(d),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(d),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(d),
    }


def prepare_stage_prediction(model, curr, pocket, times, cond_batch, pocket_equis, pocket_invs):
    cond = cond_batch if model.self_condition else None
    with torch.no_grad():
        out = model(
            curr,
            pocket,
            times,
            training=False,
            cond_batch=cond,
            pocket_equis=pocket_equis,
            pocket_invs=pocket_invs,
        )
    predicted, _ = model._get_predictions(out)
    return predicted


def world_prediction(model, predicted, pocket):
    result = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in predicted.items()}
    if torch.is_tensor(result.get("coords")):
        result["coords"] = result["coords"] * model.coord_scale
        if "complex" in pocket:
            com_list = []
            for system in pocket["complex"]:
                com = system.com
                if torch.is_tensor(com):
                    com = com.to(result["coords"].device)
                com_list.append(com)
            batch = result["coords"].shape[0]
            if len(com_list) != batch:
                com_list = com_list[:1] * batch
            result["coords"] = model.builder.undo_zero_com_batch(
                result["coords"], result["mask"], com_list=com_list
            )
    return result


def write_stage(target_dir: Path, ligand_index: int, stage_name: str, stage_t: float,
                state: dict, prediction: dict, model, pocket, metrics: dict):
    stage_dir = target_dir / f"ligand_{ligand_index:03d}" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_fields(state), stage_dir / "state.pt")
    torch.save(cpu_copy(prediction), stage_dir / "structure_affinity_prediction.pt")
    world = world_prediction(model, prediction, pocket)
    torch.save(cpu_copy(world), stage_dir / "world_prediction.pt")

    mol = None
    try:
        mols = model._generate_mols(world, sanitise=False)
        mol = mols[0] if mols else None
    except Exception as exc:  # preserve tensors even if a stage cannot decode
        metrics["decode_error"] = repr(exc)
    sdf_path = stage_dir / "ligand.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    if mol is not None:
        mol.SetProp("stage_t", str(stage_t))
        mol.SetProp("ligand_index", str(ligand_index))
        writer.write(mol)
    writer.close()

    affinity = prediction.get("affinity")
    metrics["stage_t"] = stage_t
    metrics["affinity"] = scalarize(slice_batch(affinity, 0, 1)) if affinity is not None else None
    metrics["prediction_keys"] = sorted(prediction.keys())
    metrics["state_keys"] = sorted(state.keys())
    (stage_dir / "predictions.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def generate_batch(model, prior, data, args, device, target_dir: Path, ligand_offset: int):
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["interactions"] = prior["interactions"]
    lig_prior["fragment_mask"] = prior["fragment_mask"]
    lig_prior["fragment_mode"] = prior["fragment_mode"]
    lig_prior = {k: v.to(device) if torch.is_tensor(v) else v for k, v in lig_prior.items()}
    pocket = model.builder.extract_pocket_from_complex(data)
    pocket["interactions"] = data["interactions"]
    pocket["complex"] = data["complex"]
    pocket = {k: v.to(device) if torch.is_tensor(v) else v for k, v in pocket.items()}

    batch_size = prior["coords"].shape[0]
    times = [
        torch.zeros(batch_size, device=device),
        torch.zeros(batch_size, device=device),
        torch.zeros(pocket["coords"].shape[0], device=device),
    ]
    curr = {k: (v.clone() if torch.is_tensor(v) else v.copy() if isinstance(v, list) else v) for k, v in lig_prior.items()}
    cond_batch = {
        "coords": lig_prior["coords"],
        "atomics": torch.zeros_like(lig_prior["atomics"]),
        "bonds": torch.zeros_like(lig_prior["bonds"]),
    }
    if model.sc_charges:
        cond_batch["charges"] = torch.zeros_like(lig_prior["charges"])
    if model._inpaint_self_condition:
        if model.inpainting_mode:
            cond_batch = model.builder.inpaint_molecule(lig_prior, cond_batch, pocket_mask=pocket["mask"].bool(), keep_interactions=model.flow_interactions)
        elif model.graph_inpainting:
            cond_batch = model.builder.inpaint_graph(lig_prior, cond_batch, feature_keys=["coords", "atomics", "bonds"], overwrite_with_zeros=True)

    steps = args.integration_steps
    time_points = torch.linspace(0, 1, steps + 1)
    step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
    stage_steps = {round(int(steps * t) - 1): (t, f"t_{t:.2f}") for t in EXPECTED_STAGES}
    cuda_sync(device)
    pocket_equis, pocket_invs = model.gen.get_pocket_encoding(
        pocket["coords"], pocket["atom_names"],
        pocket_atom_charges=torch.argmax(pocket["charges"], dim=-1),
        pocket_bond_types=torch.argmax(pocket["bonds"], dim=-1),
        pocket_res_types=pocket["res_names"], pocket_atom_mask=pocket["mask"],
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    stage_start = time.perf_counter()
    stage_mem_start = mem_stats(device)
    for i, step_size in enumerate(step_sizes):
        with torch.no_grad():
            out = model(curr, pocket, times, training=False,
                        cond_batch=cond_batch if model.self_condition else None,
                        pocket_equis=pocket_equis, pocket_invs=pocket_invs)
        predicted, cond_batch = model._get_predictions(out)
        curr = model.integrator.step(curr, predicted, lig_prior, times, step_size)
        if model.inpainting_mode and model.inpainting_mode_inf == "fragment":
            curr = model.builder.inpaint_molecule(lig_prior, curr, pocket_mask=pocket["mask"].bool(), keep_interactions=model.flow_interactions)
        elif model.graph_inpainting:
            curr = model.builder.inpaint_graph(lig_prior, curr, feature_keys=model.feature_keys)
        if model._inpaint_self_condition:
            if model.inpainting_mode:
                cond_batch = model.builder.inpaint_molecule(lig_prior, cond_batch, pocket_mask=pocket["mask"].bool(), keep_interactions=model.flow_interactions)
            elif model.graph_inpainting:
                cond_batch = model.builder.inpaint_graph(lig_prior, cond_batch, feature_keys=["coords", "atomics", "bonds"], overwrite_with_zeros=True)
        times = model._update_times(times, step_size)

        if i in stage_steps:
            stage_t, stage_name = stage_steps[i]
            cuda_sync(device)
            head_start = time.perf_counter()
            stage_pred = prepare_stage_prediction(model, curr, pocket, times, cond_batch, pocket_equis, pocket_invs)
            cuda_sync(device)
            now = time.perf_counter()
            mem = mem_stats(device)
            for j in range(batch_size):
                state_j = slice_batch(curr, j, batch_size)
                pred_j = slice_batch(stage_pred, j, batch_size)
                m = {
                    "integration_and_head_wall_seconds": now - stage_start,
                    "head_wall_seconds": now - head_start,
                    "memory_start": stage_mem_start,
                    "memory_end_and_peak": mem,
                }
                write_stage(target_dir, ligand_offset + j, stage_name, stage_t, state_j, pred_j, model, pocket, m)
            write_runtime(target_dir,ligand_offset,stage_name,i+1,curr,cond_batch,lig_prior,
                          times,pocket,pocket_equis,pocket_invs,args,model)
            print(json.dumps(dict(target=target_dir.name,stage=stage_name,runtime_saved=True)),flush=True)
            stage_start = time.perf_counter()
            stage_mem_start = mem_stats(device)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)

    # Match the standard generator's final t=1 corrector prediction.
    eps = -1e-4
    final_times = model._update_times(times, eps)
    cuda_sync(device)
    head_start = time.perf_counter()
    final_pred = prepare_stage_prediction(model, curr, pocket, final_times, cond_batch, pocket_equis, pocket_invs)
    cuda_sync(device)
    now = time.perf_counter()
    mem = mem_stats(device)
    for j in range(batch_size):
        state_j = slice_batch(curr, j, batch_size)
        pred_j = slice_batch(final_pred, j, batch_size)
        m = {
            "integration_and_head_wall_seconds": now - stage_start,
            "head_wall_seconds": now - head_start,
            "memory_start": stage_mem_start,
            "memory_end_and_peak": mem,
        }
        write_stage(target_dir, ligand_offset + j, "final", 1.0, state_j, pred_j, model, pocket, m)
    write_runtime(target_dir,ligand_offset,'final',steps,curr,cond_batch,lig_prior,
                  times,pocket,pocket_equis,pocket_invs,args,model)
    return batch_size


def build_args(pdb_file, ligand_file, save_dir, ckpt, gpu=0):
    sys.argv = ["stage_runner", "--pdb_file", str(pdb_file), "--ligand_file", str(ligand_file),
                "--arch", "pocket", "--pocket_type", "holo", "--chain_id", "A",
                "--cut_pocket", "--pocket_cutoff", "7", "--gpus", "1", "--mp_index", str(gpu),
                "--num_workers", "0", "--batch_cost", "4", "--ckpt_path", str(ckpt),
                "--save_dir", str(save_dir), "--sample_n_molecules_per_target", "3",
                "--max_sample_iter", "1", "--integration_steps", "100", "--categorical_strategy", "uniform-sample",
                "--ode_sampling_strategy", "linear", "--seed", "20260922"]
    return get_args()


def run_target(model, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic,
               vocab_pocket_atoms, vocab_pocket_res, spec, root, device, runtime_metadata):
    target_dir = root / spec["key"]
    target_dir.mkdir(parents=True, exist_ok=True)
    args = build_args(root / "inputs" / spec["receptor_file"], root / "inputs" / spec["ligand_file"],
                      target_dir, runtime_metadata['checkpoint'], runtime_metadata['cuda_device_index'])
    args.runtime_metadata=runtime_metadata
    args.mp_index=runtime_metadata['cuda_device_index']
    # Build target-specific preprocessing and dataloader using the project code.
    transform, interpolant = util.load_util(args, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic)
    system = util.load_data_from_pdb(args, remove_hs=hparams["remove_hs"], remove_aromaticity=hparams["remove_aromaticity"], ligand_idx=0, chain_id="A", canonicalize_conformer=False)
    dataset = get_dataset(system, transform, vocab, interpolant, args, hparams)
    loader = util.get_dataloader(args, dataset, interpolant, iter=0)
    count = 0
    t0 = time.perf_counter()
    for batch in loader:
        prior, data, _, _ = batch
        count += generate_batch(model, prior, data, args, device, target_dir, count)
    summary = {"target_key": spec["key"], "requested": 3, "generated": count, "wall_seconds": time.perf_counter() - t0}
    (target_dir / "target_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description='Stage generation with complete live continuation checkpoints')
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--input-root',required=True,help='Directory containing the receptor and ligand input files')
    parser.add_argument('--checkpoint',required=True,help='FLOWR.ROOT model checkpoint')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--precision',choices=['highest','high'],default='highest')
    options=parser.parse_args()
    root=Path(options.output_root).resolve()
    if root.exists():raise ValueError('Use a new output root; historical generation files are preserved')
    root.mkdir(parents=True)
    shutil.copytree(options.input_root,root/'inputs')
    shutil.copy2(__file__,root/'stage_runner.py')
    ckpt=Path(options.checkpoint).resolve()
    specs = [
        {"key": "2pqw_A__2rhy_MLZ", "receptor_file": "2pqw_A_rec_2rhy_mlz_lig_tt_min_0_pocket10.pdb", "ligand_file": "2pqw_A_rec_2rhy_mlz_lig_tt_min_0.sdf"},
        {"key": "5i0b_A__5vef_M77", "receptor_file": "5i0b_A_rec_5vef_m77_lig_tt_min_0_pocket10.pdb", "ligand_file": "5i0b_A_rec_5vef_m77_lig_tt_min_0.sdf"},
    ]
    # Use the parser solely to construct a complete args namespace for checkpoint loading.
    args = build_args(root / "inputs" / specs[0]["receptor_file"], root / "inputs" / specs[0]["ligand_file"],
                      root, ckpt, options.gpu)
    args.mp_index=options.gpu
    torch.cuda.set_device(options.gpu)
    torch.set_float32_matmul_precision(options.precision)
    torch.backends.cuda.matmul.allow_tf32=options.precision!='highest'
    model, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic, vocab_pocket_atoms, vocab_pocket_res = load_model(args)
    device = torch.device('cuda',options.gpu)
    model = model.to(device)
    model.eval().requires_grad_(False)
    metadata=dict(device=str(device),stages=[0.25,0.5,0.75,1.0],checkpoint=str(ckpt),
        checkpoint_bytes=ckpt.stat().st_size,checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        stage_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),precision=options.precision,
        cuda_device_index=torch.cuda.current_device(),
        runtime_fields=['curr','cond','prior','times','rng','pocket_tensors','pocket_com','pocket_equis','pocket_invs','grid'],
        resume_fidelity='Live capture with full batch, self-conditioning and RNG')
    (root/'runner_metadata.json').write_text(json.dumps(metadata,indent=2))
    (root/'experiment_manifest.json').write_text(json.dumps(dict(dataset=dict(inputs={s['key']:dict(
        receptor='inputs/'+s['receptor_file'],reference_ligand='inputs/'+s['ligand_file']) for s in specs}),runtime=metadata),indent=2))
    summaries = []
    for spec in specs:
        summaries.append(run_target(model, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic, vocab_pocket_atoms, vocab_pocket_res, spec, root, device,metadata))
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
