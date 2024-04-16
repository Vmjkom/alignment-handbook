#!/bin/bash
#SBATCH --job-name=env_setup 
#SBATCH --partition=gputest     
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:00:50
#SBATCH --account=project_2007628
#SBATCH -o %x.out
#SBATCH -e %x.err

mkdir -p logs

# Load modules
module load pytorch #The latest pytorch module has issues with venv as of 25.3.2024


#Create venv
python -m venv .venv --system-site-packages

#Activate
source .venv/bin/activate

# Install pip packages
python -m pip install peft
python -m pip install trl