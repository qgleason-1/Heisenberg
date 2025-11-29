#!/bin/bash
#
#SBATCH --job-name=testheisenberg # Assign a name to the job
#SBATCH --partition=defq # Which partition to use
#SBATCH --nodes=1 # Number of nodes
#SBATCH --output=testpy%j.log # Output that would print to the screen
 # can be saved here
#SBATCH --error=err_testpy%j.log # Errors that arise will be saved here

pwd; hostname; date

module load anaconda
./mycode
date
