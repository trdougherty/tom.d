#!/bin/bash

#SBATCH --job-name=p1
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 4
#SBATCH --time=00:10:00
#SBATCH --output=slurm-p1.out
#SBATCH --error=slurm-p1.err

ml system
ml cairo
ml julia

julia --project=@. --machine-file <(srun hostname -s) p1.jl
