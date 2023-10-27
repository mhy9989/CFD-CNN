#!/bin/bash                
#SBATCH -J CFD-test
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=1
#SBATCH -p blcy
#SBATCH --exclusive
#SBATCH -o slurm.log.%j
#SBATCH -e wrong.log.%j
#SBATCH --mem=64G

module purge
source "/work1/xdsc0029/torch.sh"

ulimit -u 200000

echo "START TIME: $(date)"

# mpirun -np $np --allow-run-as-root --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/dcu_single.sh $dist_url
python3 -u ./test.py