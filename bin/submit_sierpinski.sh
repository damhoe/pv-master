#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --mem=30gb
#SBATCH --job-name=hoedtke_sierpinski
#SBATCH --account=compacts
#SBATCH --output=sier_%j.log
#SBATCH --error=sier_%j.err
#SBATCH --workdir=/home/damianho/projects/git/pv-master/

module load anaconda/2020.07
python3 src/sierpinski_construction.py 5000000
mv sier_${SLURM_JOB_ID}.* data/sierpinski/
