"""
Main
-Process the yml config file
-Create an agent instance
-Run the agent
"""
from src.agents import *
from src.utils.utils import (
    updated_config,
    dict_print,
    create_save_loc,
    set_logger,
    backup_code,
    set_seed,
)
import wandb

import logging
import os

if __name__ == "__main__":
    # ############# handling the bash input arguments and yaml configuration file ###############
    config = updated_config()
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    # Suppress S3 GetObject spam
    logging.getLogger('s3fs').setLevel(logging.WARNING)
    logging.getLogger('fsspec').setLevel(logging.WARNING)
    os.environ['S3FS_LOG_LEVEL'] = 'WARNING'

    # create saving location and document config files
    create_save_loc(config)  # config['save_dir'] gets updated here!
    save_dir = config["save_dir"]

    # ############# handling the logistics of (seed), and (logging) ###############
    set_seed(config["train"]["seed"])
    config['log_level'] = 'ERROR'
    set_logger(save_dir, config["log_level"], "train", config["comment"])
    backup_code(save_dir)

    # printing the configuration
    dict_print(config)

    # ############# Wandb setup ###############
    wandb.init(
        project="lvfp", # TODO setup project name
        config=config,
        entity='rcl_stroke',
        name=None if config["run_name"] == "" else config["run_name"],
        mode=config["wandb_mode"],  # one of "online", "offline" or "disabled"
        notes=config["save_dir"],  # to know where the model is saved!
    )
    # Update config based on wandb sweep selected configs
    # config = wandb.config  # uncomment when using wandb sweep

    # ############# agent setup ###############
    # Create the Agent and pass all the configuration to it then run it.
    agent_class = globals()[config["agent"]]
    agent = agent_class(config)

    # ############# Run the system ###############
    if config["eval_dfr_only"]:
        # NEW: Eval DFR model only (no training)
        agent.load_checkpoint(config['pretrained_checkpoint'])  # Loads DFR-enabled model
        agent.run_epoch(0, mode=config["eval_data_type"])  # Uses DFR head
    elif config["dfr_only"]:
        # NEW: DFR-only mode - load pretrained + apply DFR
        pretrained_ckpt = config.get('pretrained_checkpoint', None)
        if pretrained_ckpt:
            agent.apply_dfr_from_checkpoint(pretrained_ckpt)
            agent.save_model(config['save_dir'], "model_dfr.pth")
            logging.info("✅ DFR applied & saved!")
        else:
            logging.error("dfr_only requires pretrained_checkpoint in config")
        agent.run_epoch(0, mode='test')
    elif config["eval_only"]:
        agent.evaluate(mode=config["eval_data_type"])
    elif config["push_only"]:
        agent.push(replace_prototypes=False)
    else:
        # Normal training (DFR auto-trains on first val if enabled)
        agent.run()

    agent.finalize()
