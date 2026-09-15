# HypOProto
Official repository for the paper:

> **HypOProto: Hyperbolic Ordinal Prototypes for Left Ventricular Filling Pressure Classification**              
> Victoria Wu, Nima Hashemi, Hooman Vaseli, Christina Luong, Purang Abolmaesumi, Teresa S. M. Tsang </br>
> [arXiv Link](https://arxiv.org/abs/2606.19804) 

--------------------------------------------------------------------------------------------------------
## Contents
- [Introduction](#Introduction)
- [Environment Setup](#Environment-Setup)
- [Train and Test](#Train-and-Test)
- [Description of Files and Folders](#Description-of-Files-and-Folders)
- [Acknowledgement](#Acknowledgement)
- [Citation](#Citation)


## Introduction 

This work aims to classify Left Ventricular Filling Pressure (LVFP) into normal vs. elevated
categories directly from echocardiography video, without relying on the Doppler-derived E/e'
ratio used in standard clinical practice, which is operator-dependent and often unavailable in
resource-limited settings.

HypOProto arranges learned prototypes in hyperbolic space along physiological scales, placing
borderline cases near the hyperboloid root and clearer diagnostic cases further outward, trained
with a Hyperbolic Prototype Angular Separation (HyperPAS) loss. This is, to our knowledge, the
first prototype-based interpretable framework applied to LVFP classification.
Due to privacy issues, we cannot share the private dataset on which we experimented on.


--------------------------------------------------------------------------------------------------------
## Environment Setup

1. Clone the repo

```bash
git clone https://github.com/DeepRCL/HypOProto.git
cd HypOProto
```
2. place your data in the `data` folder. For your private dataset, you need to prepare your own dataset class. The existing code in `src/data/` may be useful for your reference.  

3. If using Docker, it can be setup by running `docker_setup.sh` on your server. Change the parameters according to your needs:
   1. the name of the container `--name=your_container_name`  \
   2. Find the suitable pytorch image tag from https://hub.docker.com/r/pytorch/pytorch/tags based on your server.
   For example, we used: `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime`

4. Python library dependencies can be installed using:

```bash
pip install --upgrade pip
pip install torch torchvision  # if pytorch docker is not used
pip install pandas wandb tqdm seaborn torch-summary opencv-python jupyter jupyterlab imageio array2gif moviepy scikit-image scikit-learn torchmetrics termplotlib
pip install -e .
# sanity check 
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

--------------------------------------------------------------------------------------------------------
## Train and Test

To train the model `cd` to the project folder, then use the command `python main.py` with the arguments described below:

- `--config_path="src/configs/<config-name>.yml"`: yaml file containing hyper-parameters for model, experiment, loss objectives, **dataset**, and augmentations. all are stored in `src/configs`
- `--run_name="<your run name>"`: the name used by wandb to show the training results.
- `--save_dir="logs/<path-to-save>"` the folder to save all the trained model checkpoints, evaluations, and visualization of learned prototypes
- `--eval_only=True` a flag that evaluates the trained model
- `--eval_data_type="valid"` or  `--eval_data_type="test"` evaluates the model using valid or test dataset respectively. only applied when `--eval_only` flag is ON. 
- `--push_only=True` a flag to project (and then save the visualization of) the trained prototypes to the nearest relevant extracted features of training dataset. (this is done during training as well, but we can do it on any model checkpoint as standalone function using this flag)
- **Note:** You can modify any of the parameters included in the `config.yml` file on the fly by adding it as a parameter to python call in bash. For hierarchical parameters, the format is `--parent.child.child=value`
Examples for model checkpoint path:

  - `python main.py --config_path="src/configs/Ours_ProtoASNet_Video.yml" --run_name="ProtoASNet_test_run" --save_dir="logs/ProtoASNet/VideoBased_testrun_00" --model.checkpoint_path="logs/ProtoASNet/VideoBased_testrun_00/last.pth"`
  This bash command runs the last checkpoint saved in `VideoBased_testrun_00` folder.

### outputs 

the important content saved in save_dir folder are:

- `model_best.pth`: checkpoint of the best model based on a metric of interest (e.g. mean AUC or F1 score)
- `last.pth`: checkpoint of the model saved on the last epoch
- `<epoch_num>push_f1-<meanf1>.pth`: saved checkpoint after every prototype projection.

- `img/epoch-<epoch_num>_pushed`: folder containing:
  
  - visualization of projected prototypes

  - `prototypes_info.pickle`: stored dictionary containing:
    
    - `prototypes_filenames`: filenames of the source images
    - `prototypes_src_imgs`: source images in numpy
    - `prototypes_gts`: label of the source images
    - `prototypes_preds`: prediction of the source images (how model sees the source images)
    - `prototypes_occurrence_maps`: occurence map correpsonding to each prototype (where the model looks at for each prototype)
    - `prototypes_similarity_to_src_ROIs`: similarity score of the prototype vector before projection to the ROI it is projected to,

--------------------------------------------------------------------------------------------------------
## Description of files and folders

### logs
Once you run the system, it will contain the saved models, logs, and evaluation results (visualization of explanations, etc)

### pretrained_models
When training is done for the first time, pretrained backbone models are saved here.

### src
- `agents/`: folder containing agent classes for each of the architectures. contains the main framework for the training process
- `configs/`: folder containing the yaml files containing hyper-parameters for model, experiment, loss objectives, dataset, and augmentations.
- `data/`: folder for dataset and dataloader classes
- `loss/`: folder for loss functions
- `models/`: folders for model architectures
- `utils/`: folder for some utility scripts

--------------------------------------------------------------------------------------------------------
## Acknowledgement

Some code is borrowed from [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet), 
and we developed XprotoNet architecture based on their [paper](https://arxiv.org/abs/2103.10663).
This repository builds on our prior work, [ProtoASNet](https://github.com/hooman007/ProtoASNet)
(published at MICCAI 2023), which introduced dynamic prototypes for uncertainty-aware Aortic
Stenosis classification.

--------------------------------------------------------------------------------------------------------

## Citation
If you find this work useful in your research, please cite:
```
@misc{wu2026hypoprotohyperbolicordinalprototypes,
      title={HypOProto: Hyperbolic Ordinal Prototypes for Left Ventricular Filling Pressure Classification}, 
      author={Victoria Wu and Nima Hashemi and Hooman Vaseli and Christina Luong and Purang Abolmaesumi and Teresa S. M. Tsang},
      year={2026},
      eprint={2606.19804},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.19804}, 
}
```