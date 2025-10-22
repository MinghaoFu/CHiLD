import torch
import random
import argparse
import numpy as np
import ipdb as pdb
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from CHiLD.modules.instantaneous import InstantaneousProcess
from CHiLD.tools.utils import load_yaml, setup_seed
from CHiLD.datasets.sim_dataset import StationaryDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# optuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback


import wandb

from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings('ignore')

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../CHiLD/configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    pl.seed_everything(args.seed)

    # REMEMBER TO DELETE THIS BEFORE SUBMIT
    data = StationaryDataset(args.data)
    print("dataset", args.data)
    # exit()

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['VAE']['VAL_BS'], 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    model = InstantaneousProcess(input_dim=cfg['VAE']['INPUT_DIM'],
                               z_dim=cfg['VAE']['LATENT_DIM'], 
                               z_dim_fix=cfg['VAE']['LATENT_DIM_FIX'],
                               z_dim_change=cfg['VAE']['LATENT_DIM_CHANGE'],
                               lag=cfg['VAE']['LAG'],
                               nclass=cfg['VAE']['NCLASS'],
                               hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                               embedding_dim=cfg['VAE']['EMBED_DIM'],
                               lr=cfg['VAE']['LR'],
                               beta=cfg['VAE']['BETA'],
                               gamma=cfg['VAE']['GAMMA'],
                               theta=cfg['VAE']['THETA'],
                               decoder_dist=cfg['VAE']['DEC']['DIST'],
                               correlation=cfg['MCC']['CORR'],
                               enable_flexible_sparsity=cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'],
                               w_hist=cfg['VAE']['FLEXIBLE_SPARTSITY']['HIST'] if cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'] else None,
                               w_inst=cfg['VAE']['FLEXIBLE_SPARTSITY']['INST'] if cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'] else None,
                               z_dim_list=cfg['VAE']['Z_DIM_LIST'],
                               )

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcc', 
                                          save_top_k=1, 
                                          mode='max')

    early_stop_callback = EarlyStopping(monitor="val_mcc", 
                                        min_delta=0.00, 
                                        patience=10, 
                                        verbose=False, 
                                        mode="max")

    logger = WandbLogger(project=cfg['WANDB']['PROJ_NAME'], name=f'1_4_{args.seed}_{args.data}')

    # trainer = pl.Trainer(default_root_dir=log_dir,
    #                      #accelerator="auto",
    #                      val_check_interval = cfg['MCC']['FREQ'],
    #                      max_epochs=cfg['VAE']['EPOCHS'],
    #                      callbacks=[checkpoint_callback],
    #                      logger=logger,
    #                      #strategy='ddp_find_unused_parameters_true'
    #                      )

    trainer = pl.Trainer(default_root_dir=log_dir,
                         accelerator='gpu',
                         devices=cfg['VAE']['GPU'],
                         val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'],
                         callbacks=[checkpoint_callback], #, early_stop_callback],
                         logger=logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str,
        default='instantaneous_stationary_link'
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )

    argparser.add_argument(
        '-d',
        '--data',
        type=str,
        default="A"
    )

    args = argparser.parse_args()
    main(args)
