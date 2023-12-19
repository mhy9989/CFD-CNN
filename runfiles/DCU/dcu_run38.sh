#!/bin/bash                
#SBATCH -J CFD-ConvLSTM
#SBATCH -n 48
#SBATCH -N 12
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=1
#SBATCH -p blcy
#SBATCH --exclusive
#SBATCH -o slurm.log.%j
#SBATCH -e wrong.log.%j
#SBATCH --mem=64G

set -x -e

module purge
source "/work1/xdsc0029/torch38.sh"

ulimit -u 200000

echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log.txt"

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
# Adding OMPI runtime parameters
export OMPI_MCA_pml=ob1

# so processes know who to talk to
rm -f hostfile
for i in `scontrol show hostnames $SLURM_JOB_NODELIST`
do
 echo "$i slots=4" >> hostfile
done

np=$(($NNODES*4))
#module list
echo $np
nodename=$(scontrol show hostnames $SLURM_JOB_NODELIST |sed -n "1p")
dist_url=`echo $nodename | awk '{print $1}'`

# mpirun -np $np --allow-run-as-root --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/dcu_single.sh $dist_url
mpirun -np $np -mca pml ob1 -mca btl self,vader,tcp -hostfile ./hostfile --bind-to none `pwd`/dcu_single.sh $dist_url

