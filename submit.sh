#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=300
#SBATCH --mem=20gb
#SBATCH --job-name=hoedtke_sim_pv_e
#SBATCH --account=compacts
#SBATCH --output=data/sim1000k/history-sparse/pop.log
#SBATCH --error=data/sim1000k/history-sparse/pop.err
#SBATCH --workdir=/home/damianho/projects/git/PV_master/

module load anaconda/2020.07
python3 src/pop_model_efficient.py
