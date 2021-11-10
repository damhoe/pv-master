#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=300
#SBATCH --mem=60gb
#SBATCH --job-name=hoedtke_sim_pv_e
#SBATCH --account=compacts
#SBATCH --output=data/sim1000k/single/pop_1overr.log
#SBATCH --error=data/sim1000k/single/pop_1overr.err
#SBATCH --workdir=/home/damianho/projects/git/PV_master/

module load anaconda/2020.07
python3 src/pop_model_single_1overr.py
