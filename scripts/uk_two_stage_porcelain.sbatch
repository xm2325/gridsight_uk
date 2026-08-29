#!/bin/bash
#SBATCH --job-name=uk_mat_2stage
#SBATCH --account=project_2012997
#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runtime/uk_two_stage_porcelain_%j.out
#SBATCH --error=runtime/uk_two_stage_porcelain_%j.err
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
srun python3 scripts/roihu_uk_two_stage_porcelain.py --config configs/uk_two_stage_porcelain_v1.json
