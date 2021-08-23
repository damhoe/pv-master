#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=1
#SBATCH --job-name=test_job
#SBATCH --account=compacts
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
#SBATCH --workdir=/home/damianho/projects/git/PV_master/tmp/

module load anaconda/5.0.0_py3
python test_script_00.py
