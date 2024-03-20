#!/bin/bash
#SBATCH -J MSAT
#SBATCH --gpus=1
#SBATCH -o slurm-%j.log
#SBATCH -e slurm-%j.log

module purge
module load anaconda/2021.11
source activate mhypy39
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
OUTPUT_LOG=train_script.log

script -q -f "${OUTPUT_LOG}" -c "deepspeed main.py"

