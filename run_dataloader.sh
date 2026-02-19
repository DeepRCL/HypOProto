source ~/.bashrc

conda activate echoprime_venv

# Navigate + Fix PYTHONPATH
cd /home/victoriawu/workspace/ProtoASNet
export PYTHONPATH="${PWD}"

# Run with full path
python src/data/dino_dataloader.py
