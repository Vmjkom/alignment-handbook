#!/bin/bash
#SBATCH --job-name=env_setup 
#SBATCH --partition=dev-g  
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1     
#SBATCH --gpus-per-node=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=7
#SBATCH --time=00:10:00
#SBATCH --account=project_462000319
#SBATCH -o %x.out
#SBATCH -e %x.err

mkdir -p logs

# Load modules
module load LUMI #Loads correct compilers for the accelerators, propably not needed
module use /appl/local/csc/modulefiles/ #Add the module path needed for csc modules in Lumi
module load pytorch


#Create venv
python -m venv .venv --system-site-packages

#Activate
source .venv/bin/activate

# Install pip packages
pip install transformers==4.39.3 #Latest version has some backwards compability issues with previous versions on checkpointing saving
