# Linux setup and execution

Use Python 3.10 or newer. For CUDA generation, install a PyTorch build suitable for the host driver and CUDA environment before installing MolSteer.

```bash
git clone https://github.com/H-Dw/MolSteer.git
cd MolSteer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest tests -q
python -m molsteer --help
```

Use an editable installation because agent configuration, knowledge and Skill files remain in the source tree. Git ignores `outputs/`, `.venv/` and local secret files. API agents require local credentials and a selected model ID; tests do not call model APIs.

## FLOWR.ROOT generation

Live generation also requires a separate FLOWR.ROOT installation, model weights and inputs. Set these paths for the Linux host and use the Python environment containing FLOWR.ROOT. The input directory must contain the two receptor and ligand pairs used by this stage runner. Choose a new output directory.

```bash
export FLOWR_ROOT=/path/to/flowr_root
export MOLSTEER_ROOT=/path/to/MolSteer
python "$MOLSTEER_ROOT/integrations/flowr_root/stage_runner.py" \
  --input-root "$FLOWR_ROOT/output/crossdocked_100target_stage_test/inputs" \
  --checkpoint "$FLOWR_ROOT/checkpoints/flowr_root_v2.2.ckpt" \
  --output-root "$FLOWR_ROOT/output/new_exact_stages" \
  --gpu 0 --precision highest
```

Continue with the [exact checkpoint workflow](EXACT_RESTART.md). Files under `experiments/guidance/*.json` record the original experiment and contain that host's absolute paths. Copy a configuration and replace its model, input, checkpoint, reward program and output paths before running on a new host. Historical runtime checkpoints bind to model and stage-runner hashes; do not treat them as fresh checkpoints for a different host.

For independent scoring, install the optional dependencies with `python -m pip install -e '.[evaluation]'`. `prepare_independent_eval.py` finds `mk_prepare_receptor.py` beside the active Python executable or on `PATH`, or accepts `--receptor-preparer`. Pass the test log to `finalize_researcher_audit.py` with `--tests-log`.
