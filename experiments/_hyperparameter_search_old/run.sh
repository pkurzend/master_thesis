#!/usr/bin/env bash
#SBATCH --job-name=nbeats
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kurzendo@uni-hildesheim.de

# ## FOR GPU USE:
#SBATCH --partition=STUD
##SBATCH --gres=gpu:1
#SBATCH--gpus=4





for CUDA in /usr/local/cuda-11.*; do
    PATH="$CUDA/bin${PATH:+:$PATH}"
    LD_LIBRARY_PATH="$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    LD_LIBRARY_PATH="$CUDA/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
done
export PATH
export LD_LIBRARY_PATH

# cd ~/thesis/master_thesis

source activate thesis

## Run the script
srun python main.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14}