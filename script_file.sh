#!/bin/bash
#SBATCH --account=chem033284
#SBATCH --job-name=image_NMPImpi
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:01
#SBATCH --mem-per-cpu=500M


module add languages/Intel-OneAPI
cd "${SLURM_SUBMIT_DIR}"
srun --mpi=pmi2 ./forest_fire file input_grid.txt > timings.txt