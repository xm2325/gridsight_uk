#!/bin/bash
#SBATCH --job-name=gridsight-material-head-v2
#SBATCH --partition=gputest
#SBATCH --account=project_2012997
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00
#SBATCH --output=runtime/material_head_v2_%j.out
#SBATCH --error=runtime/material_head_v2_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
if ! type module >/dev/null 2>&1; then
  export CSC_ENV_INIT_NON_INTERACTIVE=yes
  source /etc/profile.d/zz-csc-env.sh
fi
module load python-pytorch/2.10
export PYTHONPATH="${SLURM_SUBMIT_DIR}/runtime/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
srun python3 scripts/roihu_material_head_v2.py --config configs/material_head_v2.json
