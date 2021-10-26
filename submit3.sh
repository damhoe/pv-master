#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=400
#SBATCH --mem=60gb
#SBATCH --job-name=hoedtke_sim_pv_e
#SBATCH --account=compacts
#SBATCH --output=data/sim1000k/pop_1overr2.log
#SBATCH --error=data/sim1000k/pop_1overr2.err
#SBATCH --workdir=/home/damianho/projects/git/PV_master/

module load anaconda/2020.07
python3 src/pop_model_efficient_1overr2.py
