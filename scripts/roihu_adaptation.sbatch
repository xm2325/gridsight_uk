#!/bin/bash
# Submit from the staged repository; pass one allowlisted stage and its arguments.
#SBATCH --job-name=gridsight-adapt
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --output=gridsight-adapt-%j.log
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if ! type module >/dev/null 2>&1; then
    export CSC_ENV_INIT_NON_INTERACTIVE=yes
    source /etc/profile.d/zz-csc-env.sh
fi
module load python-pytorch/2.10
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}/runtime/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR="${SLURM_SUBMIT_DIR}/runtime/ultralytics_config"
export OMP_NUM_THREADS=8
mkdir -p "$YOLO_CONFIG_DIR"
stage="${1:?Require prompts, overfit, or adapt}"
shift
case "$stage" in
    prompts) srun python3 scripts/roihu_prompt_ablation.py "$@" ;;
    resource-probe) srun python3 scripts/roihu_training_throughput.py "$@" ;;
    overfit|adapt) srun python3 scripts/roihu_insplad_train.py --stage "$stage" "$@" ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
