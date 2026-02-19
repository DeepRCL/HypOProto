#!/bin/bash

#SBATCH --job-name=LVFP
#SBATCH --time=90:00:00
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17                       # CPU cores per MPI process
#SBATCH --mem=180GB                                 # memory per node! 0 requests all
#SBATCH --gpus-per-node=1                       # number of GPUs per node
#SBATCH --output=LVFP_%A_%a.out

################################################################################

# 1. Reload shell profile
#source ~/.bashrc

# 2. Initialize conda (once only)
#conda init bash

# 3. Reload AGAIN to apply init changes
source ~/.bashrc

conda activate echoprime_venv

cd /home/victoriawu/workspace/ProtoASNet/

CONFIG_YML="src/configs/ProtoFPNet_Video.yml"
NAME="ProtoFPNet_Full_02"
SAVE_DIR="logs/"$NAME

# python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$NAME --log_level="ERROR" 2>&1 | grep -v "GetObject.*object_info"

#### TEST ######
python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name="Test/"$NAME \
        --eval_only=True --eval_data_type='test'  --model.checkpoint_path=$SAVE_DIR"/model_best.pth" --wandb_mode="disabled"