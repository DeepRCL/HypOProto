#!/bin/bash

#SBATCH --job-name=LVFP_Vis
#SBATCH --time=1:00:00
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17                       # CPU cores per MPI process
#SBATCH --mem=180GB                                 # memory per node! 0 requests all
#SBATCH --gpus-per-node=1                       # number of GPUs per node
#SBATCH --output=Vis_%A_%a.out

################################################################################

# 1. Reload shell profile
#source ~/.bashrc

# 2. Initialize conda (once only)
#conda init bash

# 3. Reload AGAIN to apply init changes
source ~/.bashrc

conda activate echoprime_venv

# Navigate + Fix PYTHONPATH
cd /home/victoriawu/workspace/ProtoASNet
export PYTHONPATH="${PWD}"

# Run with full path
python src/utils/vis_prot_embd_space.py
