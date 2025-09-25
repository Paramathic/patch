#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=23
#SBATCH --nodes 1
#SBATCH -t 24:00:00
#SBATCH --account=rrg-mmehride

echo export SLURM_TMPDIR=$SLURM_TMPDIR

sleep 10000000000