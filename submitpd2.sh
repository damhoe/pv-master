#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=300
#SBATCH --mem=20gb
#SBATCH --job-name=hoedtke_sim_pv_e
#SBATCH --account=compacts
#SBATCH --output=data/sim1000k/pop_panel_distr.log
#SBATCH --error=data/sim1000k/pop_panel_distr.err
#SBATCH --workdir=/home/damianho/projects/git/PV_master/

module load anaconda/2020.07
python3 src/pop_model_panel_distr_1overr2.py
