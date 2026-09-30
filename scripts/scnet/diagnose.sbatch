#!/usr/bin/env bash
#SBATCH --job-name=molsteer-probe
#SBATCH --partition=ksagnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=scnet-probe-%j.out

set -euo pipefail
cd "$HOME/MolSteer"
printf 'NODE=%s\n' "$(hostname)"
printf 'PYTHON=%s\n' "$(python3 --version 2>&1)"
printf 'HOME_FS\n'
df -h "$HOME"
printf 'GPU\n'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf 'NETWORK\n'
for url in https://pypi.org/simple/uv/ https://conda.anaconda.org/conda-forge/ https://download.pytorch.org/whl/cu121/; do
  curl -IL --connect-timeout 5 --max-time 10 -s -o /dev/null -w '%{http_code} %{url_effective}\n' "$url" || true
done
