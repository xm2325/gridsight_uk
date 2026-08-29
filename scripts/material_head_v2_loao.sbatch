#!/bin/bash
#SBATCH --job-name=gridsight-mat-v2-loao
#SBATCH --partition=gputest
#SBATCH --account=project_2012997
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH --output=runtime/material_head_v2_loao_%j.out
#SBATCH --error=runtime/material_head_v2_loao_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
if ! type module >/dev/null 2>&1; then
  export CSC_ENV_INIT_NON_INTERACTIVE=yes
  source /etc/profile.d/zz-csc-env.sh
fi
module load python-pytorch/2.10
export PYTHONPATH="${SLURM_SUBMIT_DIR}/runtime/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
srun python3 scripts/roihu_material_head_v2_loao.py --config configs/material_head_v2_loao.json
