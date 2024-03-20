#!/bin/bash
#DSUB -n MSTA
#DSUB -A root.bingxing2.gpuuser661
#DSUB -q root.default
#DSUB -l wuhanG5500
#DSUB --job_type cosched
#DSUB -R 'cpu=48;gpu=8;mem=360000'
#DSUB -N 1
#DSUB -e dsub-%J.out
#DSUB -o dsub-%J.out

module load anaconda/2020.11
#module load cuda/11.8 cudnn/8.8.1_cuda11.x
source activate py38

OUTPUT_LOG=train_"${BATCH_JOB_ID}".log

nvidia-smi

deepspeed main.py >> "${OUTPUT_LOG}" 2>&1