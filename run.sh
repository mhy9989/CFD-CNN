#!/bin/bash
#SBATCH -J msta
#SBATCH --gpus=1
#SBATCH -o slurm-%j.log
#SBATCH -e slurm-%j.log

module purge
module load compilers/cuda/11.8 compilers/gcc/9.3.0 anaconda/2021.11 cudnn/8.4.0.27_cuda11.x
source activate mhypy
OUTPUT_LOG=train_script.log

#script -q -f "${OUTPUT_LOG}" -c "deepspeed main.py"
deepspeed --master_port=29349 main.py



  
