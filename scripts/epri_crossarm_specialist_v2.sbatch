#!/bin/bash
#SBATCH --job-name=gridsight-crossarm-v2
#SBATCH --account=project_2012997
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --mem=96G
#SBATCH --output=runtime/epri_crossarm_specialist_v2_%j.out
#SBATCH --error=runtime/epri_crossarm_specialist_v2_%j.err

# CSC's non-interactive environment initializer reads PS1 before defining it.
# Keep error and pipe failure handling, but do not enable nounset here.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
if ! type module >/dev/null 2>&1; then
  export CSC_ENV_INIT_NON_INTERACTIVE=yes
  source /etc/profile.d/zz-csc-env.sh
fi
module load python-pytorch/2.10
export PYTHONPATH="${SLURM_SUBMIT_DIR}/scripts:${SLURM_SUBMIT_DIR}/runtime/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_AUTOINSTALL=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
srun python3 scripts/roihu_epri_crossarm_specialist_v2.py
