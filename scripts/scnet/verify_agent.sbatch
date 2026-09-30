#!/usr/bin/env bash
#SBATCH --job-name=molsteer-agent-check
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=scnet-agent-check-%j.out

set -euo pipefail
cd "$HOME/MolSteer"
source scripts/scnet/agent_env.sh
printf 'NODE=%s\n' "$(hostname)"
python3.12 -c 'import molsteer, rdkit, torch, langchain_core, langgraph; print("Python agent stack OK", torch.__version__, rdkit.__version__)'
python3.12 -m pytest -q tests/test_agent_config.py tests/test_agent_assets.py tests/test_agent_runtime.py tests/test_agent_flowr_bridge.py
