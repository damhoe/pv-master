#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=90
#SBATCH --mem=60gb
#SBATCH --job-name=hoedtke_sim_pv_e
#SBATCH --account=compacts
#SBATCH --output=data/sim500k/pop_1overr.log
#SBATCH --error=data/sim500k/pop_1overr.err
#SBATCH --workdir=/home/damianho/projects/git/PV_master/

module load anaconda/2020.07
python3 src/pop_model_efficient_1overr.py
