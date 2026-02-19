#!/bin/bash

#SBATCH --job-name=LVFP
#SBATCH --time=20:00:00
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17                      # CPU cores per MPI process
#SBATCH --mem=68G                               # memory per node!  4 GB per GPU
#SBATCH --gres=gpu:b200_2g.45gb:1            # for full, gpu:b200_full:1,  for  45gb gpu:b200_2g.45gb:1,  for 23gb gpu:b200_1g.23gb:1
#SBATCH --output=LVFP_EchoCard-%A-%a.out
#SBATCH --error=LVFP_EchoCard-%A-%a.err
#SBATCH --partition=mig

################################################################################

# 1. Reload shell profile
#source ~/.bashrc

# 2. Initialize conda (once only)
#conda init bash

# 3. Reload AGAIN to apply init changes
source ~/.bashrc

conda activate echoprime_venv

cd /home/victoriawu/workspace/ProtoASNet/

CONFIG_YML="src/configs/Hyper_ProtoFPNet_Video_Dino.yml"
NAME="Hyper_PCA_Dino_01"
SAVE_DIR="logs/"$NAME

python main.py --seed=88 --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$NAME --log_level="ERROR" 2>&1 | grep -v "GetObject.*object_info"

#### TEST ######
python main.py --seed=88 --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name="Test/"$NAME \
        --eval_only=True --eval_data_type='test'  --model.checkpoint_path=$SAVE_DIR"/model_best.pth" --wandb_mode="disabled"