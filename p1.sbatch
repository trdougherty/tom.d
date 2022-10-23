#!/bin/bash

#SBATCH --job-name=p1
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 4
#SBATCH --time=00:10:00
#SBATCH --output=slurm-output.out
#SBATCH --error=slurm-error.err

ml julia
julia --machine-file <(srun hostname -s) ./p1.jl
