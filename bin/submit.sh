#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=300
#SBATCH --mem=20gb
#SBATCH --job-name=hoedtke_sim_pv_e
#SBATCH --account=compacts
#SBATCH --output=sim_%j_seed_1.log
#SBATCH --error=sim_%j_seed_1.err
#SBATCH --workdir=/home/damianho/projects/git/PV_master/

module load anaconda/2020.07
python3 src/pop_model_efficient.py 1000 1overr2
mv sim_${SLURM_JOB_ID}_seed_1.* data/sim1000k/history-sparse/1overr2/
