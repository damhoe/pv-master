#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=1-00:00:00
#SBATCH --mem=30gb
#SBATCH --job-name=hoedtke_sim_pv
#SBATCH --account=compacts
#SBATCH --output=sim_%j.log
#SBATCH --error=sim_%j.err
#SBATCH --workdir=/home/damianho/projects/git/pv-master/

module load anaconda/2020.07
python3 src/pop_model_efficient.py 1000 1overr2
mv sim_${SLURM_JOB_ID}.* data/sim1000k/history-sparse/1overr2/
