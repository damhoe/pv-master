#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=mTestPyPrint2File_job
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1GB
#SBATCH --time=3:00
#SBATCH --account=compacts
#SBATCH --output=test01-%j.out
#SBATCH --error=test01-%j.err
#SBATCH --workdir=/home/damianho/projects/git/PV_master/tmp/

module load anaconda/5.0.0_py3
python3 test_script_01.py
