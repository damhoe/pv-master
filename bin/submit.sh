#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --mem=10gb
#SBATCH --job-name=hoedtke_sim_pv
#SBATCH --account=compacts
#SBATCH --output=sim_%j.log
#SBATCH --error=sim_%j.err
#SBATCH --workdir=/home/damianho/projects/git/pv-master/

module load anaconda/2020.07
python3 src/pop_random.py 100
mv sim_${SLURM_JOB_ID}.* data/sim100k/history-sparse/rnd/
