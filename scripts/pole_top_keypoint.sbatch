#!/bin/bash
#SBATCH --job-name=gridsight-pole-top-audit
#SBATCH --partition=gputest
#SBATCH --account=project_2012997
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH --output=runtime/pole_top_keypoint_%j.out
#SBATCH --error=runtime/pole_top_keypoint_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
if ! type module >/dev/null 2>&1; then
  export CSC_ENV_INIT_NON_INTERACTIVE=yes
  source /etc/profile.d/zz-csc-env.sh
fi
module load python-pytorch/2.10
export PYTHONPATH="${SLURM_SUBMIT_DIR}/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
srun python3 scripts/audit_pole_top_keypoint.py --config configs/pole_top_keypoint_v1.json
