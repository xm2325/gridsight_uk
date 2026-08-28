#!/bin/bash
# Submit from the staged repository root, with --account=project_2012997.
#SBATCH --job-name=gridsight-demo
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --output=gridsight-roihu-%j.log

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the GridSight repository root}"
# CSC skips non-interactive initialization unless explicitly requested.
# Its site script reads optional variables, so enable nounset afterwards.
if ! type module >/dev/null 2>&1; then
    export CSC_ENV_INIT_NON_INTERACTIVE=yes
    source /etc/profile.d/zz-csc-env.sh
fi
module load python-pytorch/2.10
set -u
if [[ -n "${GRIDSIGHT_VENV:-}" ]]; then
    source "${GRIDSIGHT_VENV}/bin/activate"
fi
export PYTHONUNBUFFERED=1
# Install the CLIP/tokenizer dependency during setup, never during a GPU job.
export YOLO_AUTOINSTALL=false
srun python3 scripts/roihu_demo_ablation.py "$@"
