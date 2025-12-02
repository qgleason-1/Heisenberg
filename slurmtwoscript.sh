#!/bin/bash
#
#SBATCH --job-name=heisenbergtest
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --output=heisenberg_tests%j.log
#SBATCH --error=err_heisenberg_tests%j.log

pwd; hostname; date

module load anaconda

python HeisenbergProb.py

date
