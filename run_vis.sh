#!/bin/bash

#SBATCH --job-name=LVFP
#SBATCH --time=2:00:00
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17                      # CPU cores per MPI process
#SBATCH --mem=200G                               # memory per node!  4 GB per GPU
#SBATCH --gres=gpu:b200_2g.45gb:1            # for full, gpu:b200_full:1,  for  45gb gpu:b200_2g.45gb:1,  for 23gb gpu:b200_1g.23gb:1
#SBATCH --output=Vis.out
#SBATCH --partition=mig

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
