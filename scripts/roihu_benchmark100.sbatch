#!/bin/bash
# Submit from /scratch/project_2012997/keen_ai after data/weights are verified.
#SBATCH --job-name=gridsight100
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00
#SBATCH --output=gridsight100-%j.log

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged repository root}"
if ! type module >/dev/null 2>&1; then
    export CSC_ENV_INIT_NON_INTERACTIVE=yes
    source /etc/profile.d/zz-csc-env.sh
fi
module load python-pytorch/2.10
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}/runtime/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export OMP_NUM_THREADS=8
srun python3 scripts/roihu_benchmark100.py "$@"
