#!/bin/bash
#SBATCH --job-name=debug_sft_poro
#SBATCH --account=project_462000319
#SBATCH --partition=dev-g
#SBATCH --cpus-per-task=56
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p logs
rm -f logs/debug_latest.out logs/debug_latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/debug_latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/debug_latest.err


echo "PWD" $PWD

module use /appl/local/csc/modulefiles/
module load pytorch/2.1
echo "Transformers version"
pip show transformers
source /projappl/project_462000241/villekom/instruction-tuning/Instruction-tuning-experiments/.venv/bin/activate

export PYTHONPATH="/projappl/project_462000241/villekom/instruction-tuning/Instruction-tuning-experiments/.venv/lib/python3.10/site-packages/"

#Distributed variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
#export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))
#export RDZV_HOST=$(hostname)
#export RDZV_PORT=29400
#echo master_addr


#LOGGING/DEBUGGING 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
#HF_HUB_ENABLE_HF_TRANSFER=1
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO
#export AMD_LOG_LEVEL=2
export HIP_LAUNCH_BLOCKING=1
#export HSA_FORCE_FINE_GRAIN_PCIE=1 #Supposedly improves performance/prevents hanging
#export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false #Removes error involved with the FastTokenizer and rust/python parallelism. 
                                    #See more:https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996

ACCELERATE_CONFIG_FILE=recipes/accelerate_configs/deepspeed_zero3.yaml
CONFIG_FILE=recipes/poro/sft/debug.yaml

export CMD=" \
    scripts/run_sft.py $CONFIG_FILE
    "


#LAUNCHERS
export ACC_LAUNCHER="singularity_wrapper exec accelerate launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    --debug \
    "


SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
