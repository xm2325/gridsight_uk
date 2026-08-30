#!/bin/bash
#SBATCH --job-name=gridsight-uk-loc-v1
#SBATCH --account=project_2012997
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=runtime/uk_insulator_localisation_v1_%j.out
#SBATCH --error=runtime/uk_insulator_localisation_v1_%j.err
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
srun python3 scripts/roihu_uk_insulator_localisation_v1.py --config configs/uk_insulator_localisation_prospective_v1.json
