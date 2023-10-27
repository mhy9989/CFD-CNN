#! /bin/bash

source "/work1/xdsc0029/torch.sh"

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

script_path="main.py "

DEEPSPEED_ARGS=" \
    --deepspeed \
    --local_rank $lrank \
    --zero_stage 0
    "

APP="python3 -u $script_path \
    ${DEEPSPEED_ARGS} \
    "

case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac
