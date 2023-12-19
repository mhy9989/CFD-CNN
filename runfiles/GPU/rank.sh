#!/bin/bash                

module load anaconda/2020.11
#module load cuda/11.8 cudnn/8.8.1_cuda11.x
source activate py38

export NCCL_DEBUG=INFO 
export NCCL_IB_DISABLE=0 
export PYTHONUNBUFFERED=1

NODE_RANK="${1}"
NODES="${2}"

NPROC_PER_NODE=8 #8 num gpu of use for pre node


MASTER_ADDR="${3}" 
MASTER_PORT="29501"
OUTPUT_LOG=train_"${BATCH_JOB_ID}".log

torchrun \
    --nnodes="${NODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --max_restarts=3 \
    --device 0,1,2,3,4,5,6,7 >> "${OUTPUT_LOG}" 2>&1