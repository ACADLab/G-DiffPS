#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=standard
#SBATCH --account=sa2564
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=gdiffps_train
#SBATCH --output=train_diff.out
#SBATCH --error=train_diff.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Edit these for your site
# module load cuDNN/... CUDA/...
# conda activate gdiffps
# export PATH="/path/to/ngspice/bin:$PATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python train_diffusion.py --total-timesteps 20000 --actor cfm --sizing log --gpus 1
