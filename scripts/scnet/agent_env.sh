#!/usr/bin/env bash
# Source this inside an allocated Slurm job on SCNet.
_molsteer_scnet_root="${MOLSTEER_ROOT:-$HOME/MolSteer}"
export MOLSTEER_ROOT="$_molsteer_scnet_root"
export PATH="$MOLSTEER_ROOT/tools/cpython-3.12.14-linux-x86_64-gnu/bin:$PATH"
export PYTHONPATH="$MOLSTEER_ROOT/agent_site:$MOLSTEER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
unset _molsteer_scnet_root
